import os
import torch
import torch.nn as nn
import torch.optim as optim
from base_networks import ConvBlock, ResnetBlock, ResnetBlock_b
from torchvision.transforms import *


class Net(nn.Module):
    def __init__(self, num_channels, feat, scale, training='False'):
        super(Net, self).__init__()
        self.training = training
        kernel_size = 3
        # Initial Feature Extraction
        self.m_head = ConvBlock(num_channels, feat, 3, 1, 1, activation='relu', norm=None)
        self.res1 = ResnetBlock_b(feat, kernel_size, activation='relu', res_scale='0.5')
        self.res2 = ResnetBlock_b(feat, kernel_size, activation='relu', res_scale='0.5')
        self.res3 = ResnetBlock_b(feat, kernel_size, activation='relu', res_scale='0.5')
        self.res4 = ResnetBlock_b(feat, kernel_size, activation='relu', res_scale='0.5')
        self.res5 = ResnetBlock_b(feat, kernel_size, activation='relu', res_scale='0.5')
        self.res6 = ResnetBlock_b(feat, kernel_size, activation='relu', res_scale='0.5')
        self.res7 = ResnetBlock_b(feat, kernel_size, activation='relu', res_scale='0.5')
        self.res8 = ResnetBlock_b(feat, kernel_size, activation='relu', res_scale='0.5')

        out_feats = scale * scale * num_channels
        self.tail = ConvBlock(feat, out_feats, 3, 1, 1, activation='relu', norm=None)
        self.pixel = torch.nn.PixelShuffle(scale)
        self.skip = ConvBlock(num_channels, out_feats, 3, 1, 1, activation='relu', norm=None)
        self.pad = torch.nn.ReplicationPad2d(5//2)

    def forward(self, x):
        s = self.skip(x)
        s = self.pixel(s)
        x = self.m_head(x)
        if not self.training:
            x = self.pad(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        x = self.res6(x)
        x = self.res7(x)
        x = self.res8(x)
        x = self.tail(x)
        x = self.pixel(x)
        # print(x.size(), s.size())
        x += s
        return x
