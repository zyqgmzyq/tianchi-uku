from torch import nn
import numpy as np
import torch
from config import args

import torch.nn.functional as F
import math


def make_model():
    return MYVGG()


class MYVGG(nn.Module):
    def __init__(self):
        super(MYVGG, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2)
        )
        layer = []
        for i in range(1,10):
            layer.append(nn.Conv2d(64, 64, kernel_size=3, padding=1))
            layer.append(nn.LeakyReLU(0.2))

        self.layer2 = nn.Sequential(*layer)

        if args.scale==4:
            self.layer3 = nn.Sequential(
                nn.ConvTranspose2d(64,32,3,stride=args.scale//2,bias=False,padding=1,output_padding=args.scale//2-1),
                nn.LeakyReLU(0.2),
                nn.ConvTranspose2d(32,8,3,stride=args.scale//2,bias=False,padding=1,output_padding=args.scale//2-1),
                nn.LeakyReLU(0.2),
            )
        else:
            self.layer3 = nn.Sequential(
                nn.ConvTranspose2d(64,8,3,stride=args.scale,bias=False,padding=1,output_padding=args.scale-1),
                nn.LeakyReLU(0.2)
            )
        self.layer4 = nn.Sequential(
                nn.Conv2d(8, 3, kernel_size=3, padding=1),
                nn.Tanh()
            )
        self.PS = nn.PixelShuffle(args.scale)

    def forward(self, x):
        residual = x.view(3*x.shape[0],-1).repeat(1,args.scale**2).view(3*args.scale**2,-1).view(x.shape[0],3*args.scale**2,x.shape[2],x.shape[3]) 
        out = self.layer1(x)
        temp = out
        out = self.layer2(out)
        out += temp
        out = self.layer3(out)
        out = self.layer4(out)

        return out + self.PS(residual)
