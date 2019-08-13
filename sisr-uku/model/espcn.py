from torch import nn
import numpy as np
import torch
from config import args


def make_model():
    return ESPCN()

class ESPCN(nn.Module):
    def __init__(self):
        super(ESPCN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, padding=2),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 3*args.scale*args.scale, kernel_size=3, padding=1),
            nn.PixelShuffle(args.scale),
            nn.Tanh()

        )
        # self.shift = Shift(1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        return out