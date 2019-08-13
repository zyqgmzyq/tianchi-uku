import torch
import torch.nn as nn
import torch.nn.functional as F
import models.common as common


class UNET(nn.Module):
    def __init__(self, num_block, num_feature, scale, n_colors=15, res_scale=0.1, conv=common.default_conv, isTrain=True):
        super(UNET, self).__init__()

        n_resblocks = num_block
        n_feats = num_feature
        scale = scale
        kernel_size = 3
        act = nn.ReLU(True)

        self.head = common.default_conv(n_colors, n_feats, kernel_size)
        self.inc = common.inconv(num_feature, num_feature)
        self.down1 = common.down(num_feature, num_feature)
        self.down2 = common.down(num_feature, num_feature)
        self.up1 = common.up(num_feature, num_feature)
        self.up2 = common.up(num_feature, num_feature)

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

        data_yuv_head = self.head(data_yuv)
        data_yuv1_x1 = self.inc(data_yuv_head)
        data_yuv1_x2 = self.down1(data_yuv1_x1)
        data_yuv1_x3 = self.down2(data_yuv1_x2)
        data_yuv1 = self.up1(data_yuv1_x3, data_yuv1_x2)
        data_yuv1 = self.up2(data_yuv1, data_yuv1_x1)
        data_yuv1 += data_yuv1_x1

        data_y1 = self.y_out_up4(data_yuv1) + self.y_res_up4(data_y)
        data_y1 = self.y_out(data_y1)
        data_y1 *= 127.5
        data_y1 += 127.5
        data_y1 = torch.clamp(data_y1, 0., 255.)

        data_uv1 = self.uv_out_up2(data_yuv1) + self.uv_res_up4(data_uv)
        data_uv1 = self.uv_out(data_uv1)
        data_uv1 *= 127.5
        data_uv1 += 127.5
        data_uv1 = torch.clamp(data_uv1, 0., 255.)

        if self.isTrain:
            loss_y = F.mse_loss(data_y1, label_y, reduction='mean')
            loss_uv = F.mse_loss(data_uv1, label_uv, reduction='mean')
            return [data_y1, data_uv1, loss_y, loss_uv]

        return [data_y1, data_uv1]

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
    model = UNET(num_block=num_block, num_feature=num_feature, scale=scale, isTrain=is_train)
    return model
