import torch
import torch.nn as nn
import torch.nn.functional as F
import models.common as common


class EDSR(nn.Module):
    def __init__(self, num_block, num_feature, scale, n_colors=15, res_scale=0.1, conv=common.default_conv, isTrain=True):
        super(EDSR, self).__init__()

        n_resblocks = num_block
        n_feats = num_feature
        scale = scale
        kernel_size = 3
        act = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # define head module

        # define body module
        m_body = [
            common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=res_scale
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        self.body = nn.Sequential(*m_body)

        self.down_x2 = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=2, padding=1),
            act
        )
        self.up_x2 = common.Upsampler(conv, scale=2, n_feats=n_feats, act=act)

        self.down_x4 = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=2, padding=1),
            act
        )
        self.up_x4 = common.Upsampler(conv, scale=2, n_feats=n_feats, act=act)

        self.upsampler_2 = common.Upsampler(conv, scale=2, n_feats=n_feats, act=act)
        self.upsampler_4 = common.Upsampler(conv, scale=4, n_feats=n_feats, act=act)

        self.head_y = conv(5, n_feats, kernel_size)
        self.head_uv = conv(10, n_feats, kernel_size)

        self.y_out = conv(n_feats, 1, kernel_size)

        self.uv_out = conv(n_feats, 2, kernel_size)

        self.isTrain = isTrain

    def forward(self, data_y, data_uv, label_y=None, label_uv=None):
        data_y -= 127.5
        data_y /= 127.5

        data_uv -= 127.5
        data_uv /= 127.5

        data_y = self.head_y(data_y)
        data_uv_2x = self.head_uv(F.interpolate(data_uv, scale_factor=2, mode='nearest'))

        data_yuv = data_y + data_uv_2x

        # Add padding to avoid div zero
        _, _, h, w = data_yuv.shape
        pad_h = 0 if h % 4 == 0 else h % 4
        pad_w = 0 if w % 4 == 0 else w % 4

        data_yuv = F.pad(data_yuv, pad=(pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), mode='constant', value=0)

        res2x = self.down_x2(data_yuv)
        res4x = self.down_x4(res2x)

        residual = res4x
        res4x = self.body(res4x)
        res4x += residual
        output_2x = self.up_x4(res4x)

        residual = res2x
        res2x = self.body(output_2x*0.2 + res2x)
        res2x += residual
        output = self.up_x2(res2x)

        residual = data_yuv
        output = self.body(output*0.2 + data_yuv)
        output += residual

        _, _, h, w = output.shape
        output = output[:, :, pad_h // 2:h - (pad_h - pad_h // 2), pad_w // 2: w - (pad_w - pad_w // 2)]

        data_y = self.upsampler_4(output)
        data_y = self.y_out(data_y)

        data_uv = self.upsampler_2(output)
        data_uv = self.uv_out(data_uv)

        data_y *= 127.5
        data_y += 127.5

        data_uv *= 127.5
        data_uv += 127.5

        if self.isTrain:
            loss_y = F.mse_loss(data_y, label_y, reduction='sum')
            loss_uv = F.mse_loss(data_uv, label_uv, reduction='sum')
            return [data_y, data_uv, loss_y, loss_uv]

        return [data_y, data_uv]

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == ~ -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))


def get_model(num_block=32, num_feature=160, scale=4, isTrain=True):
    model = EDSR(num_block=num_block, num_feature=num_feature, scale=scale, isTrain=isTrain)
    return model