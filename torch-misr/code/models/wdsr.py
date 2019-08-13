import torch
import torch.nn as nn
import torch.nn.functional as F
import models.common as common


class WDSR(nn.Module):
    def __init__(self, num_block, num_feature, scale, n_colors=15, res_scale=0.1, conv=common.default_conv, isTrain=True):
        super(WDSR, self).__init__()

        n_resblocks = num_block
        n_feats = num_feature
        scale = scale
        kernel_size = 3
        act = nn.ReLU(True)

        # define head module
        m_head = [conv(n_colors, n_feats, kernel_size)]

        # define body module
        m_body = [
            common.ResBlockb(
                conv, n_feats, kernel_size, act=act, res_scale=res_scale
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)

        self.uv_up2 = common.Upsampler(conv, scale=2, n_feats=10, act=False)

        self.y_out_up4 = common.Upsampler(conv, scale=4, n_feats=n_feats, act=False)

        self.uv_out_up2 = common.Upsampler(conv, scale=2, n_feats=n_feats, act=False)

        self.y_res_up4 = nn.Sequential(
            common.Upsampler(conv, scale=4, n_feats=5, act=False),
            conv(5, n_feats, kernel_size)
        )

        self.y_out = conv(n_feats, 1, kernel_size)

        self.uv_res_up4 = nn.Sequential(
            common.Upsampler(conv, scale=4, n_feats=10, act=False),
            conv(10, n_feats, kernel_size)
        )

        self.uv_out = conv(n_feats, 2, kernel_size)

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
        data_yuv = data_yuv + res

        data_y = self.y_out_up4(data_yuv) + self.y_res_up4(data_y)
        data_y = self.y_out(data_y)
        data_y *= 127.5
        data_y += 127.5
        data_y = torch.clamp(data_y, 0., 255.)

        data_uv = self.uv_out_up2(data_yuv) + self.uv_res_up4(data_uv)
        data_uv = self.uv_out(data_uv)
        data_uv *= 127.5
        data_uv += 127.5
        data_uv = torch.clamp(data_uv, 0., 255.)

        if self.isTrain:
            loss_y = F.mse_loss(data_y, label_y, reduction='mean')
            loss_uv = F.mse_loss(data_uv, label_uv, reduction='mean')
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
                        raise RuntimeError('While W the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == ~ -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))


def get_model(num_block=16, num_feature=128, scale=4, is_train=True):
    model = WDSR(num_block=num_block, num_feature=num_feature, scale=scale, isTrain=is_train)
    return model
