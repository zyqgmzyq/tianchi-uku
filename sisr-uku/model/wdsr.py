import torch
import math
import torch.nn as nn


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


class ResBlock(nn.Module):
    def __init__(self, n_feats, kernel_size, wn, act=nn.ReLU(True), res_scale=1):
        super(ResBlock, self).__init__()
        self.res_scale = res_scale
        body = []
        expand = 6
        linear = 0.8
        body.append(
            wn(default_conv(n_feats, n_feats*expand, 1)))
        body.append(act)
        body.append(
            wn(default_conv(n_feats*expand, int(n_feats*linear), 1)))
        body.append(
            wn(default_conv(int(n_feats*linear), n_feats, kernel_size)))

        self.body = nn.Sequential(*body)

    def forward(self, x):
        res = self.body(x) * self.res_scale
        res += x
        return res


def make_model(n_resblocks=32, n_feats=64):
    return WDSR(n_resblocks=n_resblocks, n_feats=n_feats)


class WDSR(nn.Module):
    def __init__(self, n_resblocks=16, n_feats=64):
        super(WDSR, self).__init__()
        res_scale = 1.0
        kernel_size = 3
        scale = 4
        n_colors = 3
        act = nn.ReLU(True)
        wn = lambda x: torch.nn.utils.weight_norm(x)

        # define head module
        m_head = []
        m_head.append(
            wn(default_conv(n_colors, n_feats, kernel_size)))

        # define body module
        m_body = []
        for i in range(n_resblocks):
            m_body.append(
                ResBlock(n_feats, kernel_size, act=act, res_scale=res_scale, wn=wn))

        # define tail module
        m_tail = []
        out_feats = scale * scale * n_colors
        m_tail.append(
            wn(default_conv(n_feats, out_feats, 3)))
        m_tail.append(nn.PixelShuffle(scale))
        m_skip = []
        m_skip.append(
            wn(default_conv(n_colors, out_feats, 5))
        )
        m_skip.append(nn.PixelShuffle(scale))

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)
        self.skip = nn.Sequential(*m_skip)

    def forward(self, x):
        s = self.skip(x)
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        x += s
        return x