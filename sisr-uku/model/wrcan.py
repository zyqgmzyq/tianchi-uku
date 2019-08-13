import torch
from torch.nn import init
import torch.nn as nn
import math


def make_model():
    return WRCAN()


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)  # 这个padding两边都会进行扩充


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


# Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_y = self.avg_pool(x)
        max_y = self.max_pool(x)
        avg_y = self.conv_du(avg_y)
        max_y = self.conv_du(max_y)
        y = self.sigmoid(avg_y + max_y)
        return x * y


# Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(
            self, conv, n_feat, block_feat, kernel_size, reduction,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        modules_body.append(conv(n_feat, block_feat, kernel_size, bias=bias))
        modules_body.append(act)
        modules_body.append(conv(block_feat, n_feat, kernel_size, bias=bias))
        modules_body.append(CALayer(n_feat, reduction))

        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        # res = self.body(x)
        res = self.body(x)
        res += x.mul(self.res_scale)
        return res


# Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, block_feat, kernel_size, reduction, act, res_scale, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = [
            RCAB(
                conv, n_feat, block_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True), res_scale=0.5) \
            for _ in range(n_resblocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        res += x.mul(self.res_scale)
        return res


# Residual Channel Attention Network (RCAN)
class WRCAN(nn.Module):
    def __init__(self, args, conv=default_conv):
        super(WRCAN, self).__init__()

        n_resgroups = 10
        n_resblocks = 20
        n_feats = 64
        block_feats = 128
        n_colors = 3
        print('G:', n_resblocks, 'R:', n_resgroups, 'BlockFeats:', block_feats, 'N_feats:', n_feats)

        kernel_size = 3
        reduction = 16
        scale = 4
        act = nn.ReLU(True)

        # define head module
        modules_head = [conv(n_colors, n_feats, kernel_size)]

        # define body module
        modules_body = [
            ResidualGroup(
                conv, n_feat=n_feats, block_feat=block_feats, kernel_size=kernel_size, reduction=reduction, act=act, res_scale=1,
                n_resblocks=n_resblocks) for _ in range(n_resgroups)]
        modules_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        modules_tail = [
            Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, 3, kernel_size)]

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)
        self.PS = nn.PixelShuffle(scale)

    def forward(self, x):
        residual = x.view(3 * x.shape[0], -1).repeat(1, 4 ** 2).view(3 * 4 ** 2, -1).view(x.shape[0],3*4**2,x.shape[2],x.shape[3])
        x = self.head(x)  # 一个小的卷积层
        res = self.body(x)  # 主要的那个循环结构
        res += x  # 残差

        x = self.tail(res)  # 残差结束后进行上采样
        # x /= 255.
        x = x + self.PS(residual)
        return x