import torch.nn as nn
import math
import models.common as common
import torch
import torch.nn.functional as F
from torch.autograd import Variable

def make_model():
    return RCAN()

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)  # 这个padding两边都会进行扩充

class BasicBlock(nn.Sequential):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, bias=False,
        bn=True, act=nn.ReLU(True)):

        m = [nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=(kernel_size//2), stride=stride, bias=bias)
        ]
        if bn: m.append(nn.BatchNorm2d(out_channels))
        if act is not None: m.append(act)
        super(BasicBlock, self).__init__(*m)

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if i == 0: m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feat, bn=False, act=False, bias=True):  # 这个conv是默认卷积

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?  # 恰好是2的指数倍才会为0
            for _ in range(int(math.log(scale, 2))):  # 如果是超过2，就进行多次放大
                m.append(conv(n_feat, 4 * n_feat, 3, bias))  # 先用一个卷积把图的通道进行扩大
                m.append(nn.PixelShuffle(2))  # 然后运行这个PixelShuffle
                if bn: m.append(nn.BatchNorm2d(n_feat))  # 默认没bn
                if act: m.append(act())  # 默认不激活
        elif scale == 3:
            m.append(conv(n_feat, 9 * n_feat, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if act: m.append(act())
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)  # 牛逼啊，直接继承Sequential，然后返回自己

## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 默认是1*1，如果传入元组就是对应形状
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),  # 得到那串缩小的值
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),  # 再卷积回之前的维度
                nn.Sigmoid()  # 又进行了一个激活操作
        )

    def forward(self, x):
        y = self.avg_pool(x)  # 先池化
        y = self.conv_du(y)  # 再获取那个attention
        return x * y  # 注意力通道

## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):  # 这个块基本上就是原论文那个块
    def __init__(
        self, conv, n_feat, kernel_size, reduction,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))  # 默认是没有bn的
            if i == 0: modules_body.append(act)  # 就加一个激活层，即在第一个卷积后加一个激活
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)  # 这个Sequential就是把所有的卷积层、ReLU，按放的顺序拼起来
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)  # 那一整套解包的
        #res = self.body(x).mul(self.res_scale)
        res += x  # 直接加上头
        return res

## Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = []
        modules_body = [
            RCAB(
                conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True), res_scale=1) \
            for _ in range(n_resblocks)]  # 循环这么多个模块
        modules_body.append(conv(n_feat, n_feat, kernel_size))  # 再加一个普通的卷积
        self.body = nn.Sequential(*modules_body)  # 对这个进行解压

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

## Residual Channel Attention Network (RCAN)
class RCAN(nn.Module):
    def __init__(self, conv=default_conv, isTrain=True, num_feature=64):  # 这个con是最最普通的那种卷积
        super(RCAN, self).__init__()
        
        n_resgroups = 10  # 残差组的数量
        n_resblocks = 20  # 残差块的数量
        n_feats = num_feature  # 特征的数量，即通道
        kernel_size = 3
        reduction = 16  # 论文中的那个r因子
        scale = 4  # 默认只进行一个因子的训练
        act = nn.ReLU(True)  # True就是进行原位运算
        
        # define head module
        modules_head = [conv(15, n_feats, 3)]  # n_colors使用的通道数

        # define body module
        modules_body = [
            ResidualGroup(  # act激活函数，残差缩放比例
                conv, n_feats, kernel_size, reduction, act=act, res_scale=1, n_resblocks=n_resblocks) \
            for _ in range(n_resgroups)]

        modules_body.append(conv(n_feats, n_feats, kernel_size))  # 主体尾巴那个小卷积层

        self.uv_up2 = common.Upsampler(conv, scale=2, n_feats=10, act=False)

        self.y_out_up4 = common.Upsampler(conv, scale=4, n_feats=n_feats, act=False)

        self.uv_out_up2 = common.Upsampler(conv, scale=2, n_feats=n_feats, act=False)

        self.y_res_up4 = nn.Sequential(
            common.Upsampler(conv, scale=4, n_feats=5, act=False),
            conv(5, n_feats, kernel_size)
        )

        self.yy_out = conv(n_feats, 1, kernel_size)

        self.uv_res_up4 = nn.Sequential(
            common.Upsampler(conv, scale=4, n_feats=10, act=False),
            conv(10, n_feats, kernel_size)
        )

        self.uuv_out = conv(n_feats, 2, kernel_size)

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.PS = nn.PixelShuffle(scale)
        self.isTrain = isTrain

    def forward(self, data_y, data_uv, label_y=None, label_uv=None):
        data_y -= 127.5
        data_y /= 127.5

        data_uv -= 127.5
        data_uv /= 127.5

        data_uv_2x = self.uv_up2(data_uv)
        data_yuv = torch.cat([data_y, data_uv_2x], dim=1)
        data_yuv = self.head(data_yuv)
        res = self.body(data_yuv)
        y_out = res + data_yuv

        y_out = self.y_out_up4(y_out) + self.y_res_up4(data_y)
        y_out = self.yy_out(y_out)
        y_out *= 127.5
        y_out += 127.5
        y_out = torch.clamp(y_out, 0., 255.)

        uv_out = self.uv_out_up2(res) + self.uv_res_up4(data_uv)
        uv_out = self.uuv_out(uv_out)
        uv_out *= 127.5
        uv_out += 127.5
        uv_out = torch.clamp(uv_out, 0., 255.)
        if self.isTrain:
            loss_y = F.mse_loss(y_out, label_y, reduction='mean')
            loss_uv = F.mse_loss(uv_out, label_uv, reduction='mean')
            return [y_out, uv_out, loss_y, loss_uv]

        return [y_out, uv_out]


def get_model(num_block=16, num_feature=128, scale=4, is_train=True):
    model = RCAN(isTrain=is_train, num_feature=num_feature)
    return model