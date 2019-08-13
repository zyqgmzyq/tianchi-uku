from model import common
import torch
import torch.nn.functional as F
import torch.nn as nn
import math


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
                if bn:
                    m.append(nn.BatchNorm2d(n_feat))  # 默认没bn
                if act:
                    m.append(act())  # 默认不激活
        elif scale == 3:
            m.append(conv(n_feat, 9 * n_feat, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feat))
            if act:
                m.append(act())
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
    def __init__(self, num_block, num_feature, scale, n_colors=15, res_scale=0.1, conv=default_conv,
                 isTrain=True):
        super(WRCAN, self).__init__()

        n_resgroups = num_block
        n_resblocks = 20
        n_feats = num_feature
        block_feats = 128
        n_colors = n_colors
        print('G:', n_resblocks, 'R:', n_resgroups, 'BlockFeats:', block_feats, 'N_feats:', n_feats)

        kernel_size = 3
        reduction = 16
        scale = scale
        act = nn.ReLU(True)

        # define head module
        modules_head = [conv(n_colors, n_feats, kernel_size)]

        # define body module
        modules_body = [
            ResidualGroup(
                conv, n_feat=n_feats, block_feat=block_feats, kernel_size=kernel_size, reduction=reduction, act=act, res_scale=res_scale,
                n_resblocks=n_resblocks) for _ in range(n_resgroups)]
        modules_body.append(conv(n_feats, n_feats, kernel_size))

        self.y_out_up4 = nn.Sequential(
            Upsampler(conv, scale=4, n_feats=n_feats, act=False),
            conv(n_feats, 1, kernel_size)
        )

        self.uv_out_up4 = nn.Sequential(
            Upsampler(conv, scale=4, n_feats=n_feats, act=False),
            conv(n_feats, 2, kernel_size)
        )

        self.uv_up2 = Upsampler(conv, scale=2, n_feats=10, act=False)
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
        res += data_yuv

        y_out = self.y_out_up4(res) + self.y_out_up4(data_y)
        uv_out = self.uv_out_up4(res) + self.y_out_up4(data_uv)

        y_out *= 127.5
        y_out += 127.5
        y_out = torch.clamp(y_out, 0., 255.)

        uv_out *= 127.5
        uv_out += 127.5
        uv_out = torch.clamp(uv_out, 0., 255.)

        if self.isTrain:
            loss_y = F.mse_loss(y_out, label_y, reduction='mean')
            loss_uv = F.mse_loss(uv_out, label_uv, reduction='mean')
            return [y_out, uv_out, loss_y, loss_uv]

        return [y_out, uv_out]


def get_model(num_block=16, num_feature=128, scale=4, is_train=True):
    model = WRCAN(num_block=num_block, num_feature=num_feature, scale=scale, isTrain=is_train)
    return model

