import torch.nn as nn
import math

import torch
import torch.nn.functional as F
from config import args
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
    def __init__(self, conv=default_conv):  # 这个con是最最普通的那种卷积
        super(RCAN, self).__init__()
        
        n_resgroups = 10  # 残差组的数量
        n_resblocks = 20 # 残差块的数量
        n_feats = 64  # 特征的数量，即通道
        kernel_size = 3
        reduction = 16  # 论文中的那个r因子
        scale = args.scale  # 默认只进行一个因子的训练
        act = nn.ReLU(True)  # True就是进行原位运算
        
        # define head module
        modules_head = [conv(3, 64, 3)]  # n_colors使用的通道数

        # define body module
        modules_body = [
            ResidualGroup(  # act激活函数，残差缩放比例
                conv, n_feats, kernel_size, reduction, act=act, res_scale=1, n_resblocks=n_resblocks) \
            for _ in range(n_resgroups)]

        modules_body.append(conv(n_feats, n_feats, kernel_size))  # 主体尾巴那个小卷积层

        # define tail module
        modules_tail = [
            Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, 3, kernel_size)]

        # 总结的结果
        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)
        self.PS = nn.PixelShuffle(args.scale)

    def forward(self, x):
        # x *= 255.
        residual = x.view(3*x.shape[0], -1).repeat(1,args.scale**2).view(3*args.scale**2,-1).view(x.shape[0],3*args.scale**2,x.shape[2],x.shape[3])
        x = self.head(x)  # 一个小的卷积层

        res = self.body(x)  # 主要的那个循环结构
        res += x  # 残差

        x = self.tail(res)  # 残差结束后进行上采样
        # x /= 255.
        x = x + self.PS(residual)
        return x