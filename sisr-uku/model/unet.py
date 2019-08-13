import math
import torch.nn as nn
import torch


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


class double_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
        )

    def forward(self, x):
        x = x + self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=False, subpixel=True):
        super(up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        elif subpixel:
            self.up = Upsampler(default_conv, 2, out_ch, act=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)
        self.downshannel = default_conv(2 * in_ch, out_ch, 1)
        self.conv = double_conv(in_ch, out_ch)
        self.conv2 = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.downshannel(x)
        x = self.conv(x)
        x = self.conv2(x)
        return x


def make_model():
    return UNET()


class UNET(nn.Module):
    def __init__(self):
        super(UNET, self).__init__()
        n_colors = 3
        n_feats = 64
        kernel_size = 3
        scale = 4
        act = nn.ReLU(True)

        self.data_up4 = Upsampler(default_conv, scale=4, n_feats=n_colors, act=False)
        self.head = default_conv(n_colors, n_feats, kernel_size)
        self.inc = inconv(n_feats, n_feats)
        self.down1 = down(n_feats, n_feats)
        self.down2 = down(n_feats, n_feats)
        self.up1 = up(n_feats, n_feats)
        self.up2 = up(n_feats, n_feats)
        self.out_conv = default_conv(n_feats, n_colors, kernel_size)

        self.head_2 = default_conv(n_colors, n_feats, kernel_size)
        self.inc_2 = inconv(n_feats, n_feats)
        self.down1_2 = down(n_feats, n_feats)
        self.down2_2 = down(n_feats, n_feats)
        self.up1_2 = up(n_feats, n_feats)
        self.up2_2 = up(n_feats, n_feats)
        self.out_conv_2 = default_conv(n_feats, n_colors, kernel_size)

        self.head_3 = default_conv(n_colors, n_feats, kernel_size)
        self.inc_3 = inconv(n_feats, n_feats)
        self.down1_3 = down(n_feats, n_feats)
        self.down2_3 = down(n_feats, n_feats)
        self.up1_3 = up(n_feats, n_feats)
        self.up2_3 = up(n_feats, n_feats)
        self.out_conv_3 = default_conv(n_feats, n_colors, kernel_size)

    def forward(self, x):
        s = self.data_up4(x)
        x0 = self.head(s)
        x1 = self.inc(x0)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.up1(x3, x2)
        x5 = self.up2(x4, x1)
        x = self.out_conv(x5)
        x_1 = x + s

        x0_2 = self.head_2(x_1)
        x1_2 = self.inc_2(x0_2)
        x2_2 = self.down1_2(x1_2)
        x3_2 = self.down2_2(x2_2)
        x4_2 = self.up1_2(x3_2, x2_2)
        x5_2 = self.up2_2(x4_2, x1_2)
        x_2 = self.out_conv_2(x5_2)
        x_2 = x_2 + x

        x0_3 = self.head_3(x_2)
        x1_3 = self.inc_3(x0_3)
        x2_3 = self.down1_3(x1_3)
        x3_3 = self.down2_3(x2_3)
        x4_3 = self.up1_3(x3_3, x2_3)
        x5_3 = self.up2_3(x4_3, x1_3)
        x_3 = self.out_conv_3(x5_3)
        x_3 = x_3 + x_2

        # x0 = self.head(x_1)
        # x1 = self.inc(x0)
        # x2 = self.down1(x1)
        # x3 = self.down2(x2)
        # x4 = self.up1(x3, x2)
        # x5 = self.up2(x4, x1)
        # x = self.out_conv(x5)
        # x_2 = x + x_1
        #
        # x0 = self.head(x_2)
        # x1 = self.inc(x0)
        # x2 = self.down1(x1)
        # x3 = self.down2(x2)
        # x4 = self.up1(x3, x2)
        # x5 = self.up2(x4, x1)
        # x = self.out_conv(x5)
        # x_3 = x + x_2

        return [x_1, x_2, x_3]

