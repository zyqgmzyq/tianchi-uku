from importlib import import_module
from torch import nn
import torch
from config import args

# net = getattr(model, args.model.upper())
torch.manual_seed(args.seed)


class Model(nn.Module):
    """docstring for Model"""
    def __init__(self, **m):
        super(Model, self).__init__()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        module = import_module('model.' + args.model.lower())
        self.model = module.make_model(**m).to(self.device)

    def forward(self, x):
        if args.ensemble:
            return self.forward_x2(x)
        else:
            return self.model(x)

    def print(self):
        print(self.model)
        print('parameters: ', sum(param.numel() for param in self.model.parameters()))
        # for name, parm in self.model.state_dict().items():
        #     print(name)
            # break

    def save(self, ckpt):
        torch.save(self.model.state_dict(), ckpt)

    def load(self, ckpt):

        if args.ckpt:
            ckpt = args.ckpt

        try:
            kwargs = {} if torch.cuda.is_available() else {'map_location': 'cpu'}
            self.model.load_state_dict(torch.load(ckpt, **kwargs), strict=False)
            print('load checkpoint file from "{}" succeeded!'.format(ckpt))
        except:
            print('try to load checkpoint file from "{}" failed!'.format(ckpt))
            exit(1)

    def forward_x2(self, x):
        def _transform(v, op):
            # if self.precision != 'single': v = v.float()

            v2np = v.data.cpu().numpy()
            if op == 'v':
                tfnp = v2np[:, :, :, ::-1].copy()
            elif op == 'h':
                tfnp = v2np[:, :, ::-1, :].copy()
            elif op == 't':
                tfnp = v2np.transpose((0, 1, 3, 2)).copy()

            ret = torch.Tensor(tfnp).to(self.device)
            # if self.precision == 'half': ret = ret.half()

            return ret

        lr_list = [x]  # 1 3 h w

        for tf in 'v':
            lr_list.extend([_transform(t, tf) for t in lr_list])

        lr_list = [aug for aug in lr_list]
        lr_batch1 = torch.cat(lr_list, dim=0)
        # lr_batch2 = torch.cat(lr_list[4:], dim=0)
        sr_batch1 = self.model.forward(lr_batch1)
        # sr_batch2 = self.model.forward(lr_batch2)
        sr_list = []
        for i in range(len(sr_batch1)):
            sr_list.append(sr_batch1[i:i + 1])

        # for i in range(len(sr_batch2)):
        #     sr_list.append(sr_batch2[i:i+1])

        # sr_list = [sr_batch[i:i+1] for i in range(len(sr_batch))]
        # sr_list = [self.model.forward(aug) for aug in lr_list]
        # for i in range(len(sr_list)):
        #     if i > 3:
        #         sr_list[i] = _transform(sr_list[i], 't')
        #     if i % 4 > 1:
        #         sr_list[i] = _transform(sr_list[i], 'h')
        #     if (i % 4) % 2 == 1:
        #         sr_list[i] = _transform(sr_list[i], 'v')

        sr_list[-1] = _transform(sr_list[-1], 'v')

        output_cat = torch.cat(sr_list, dim=0)
        output = output_cat.mean(dim=0, keepdim=True)

        return output

    def forward_x8(self, x):
        def _transform(v, op):
            # if self.precision != 'single': v = v.float()

            v2np = v.data.cpu().numpy()
            if op == 'v':
                tfnp = v2np[:, :, :, ::-1].copy()
            elif op == 'h':
                tfnp = v2np[:, :, ::-1, :].copy()
            elif op == 't':
                tfnp = v2np.transpose((0, 1, 3, 2)).copy()

            ret = torch.Tensor(tfnp).to(self.device)
            # if self.precision == 'half': ret = ret.half()

            return ret

        lr_list = [x]  # 1 3 h w

        for tf in 'v', 'h', 't':
            lr_list.extend([_transform(t, tf) for t in lr_list])

        lr_list = [aug for aug in lr_list]
        lr_batch1 = torch.cat(lr_list[:4], dim=0)
        lr_batch2 = torch.cat(lr_list[4:], dim=0)
        sr_batch1 = self.model.forward(lr_batch1)
        sr_batch2 = self.model.forward(lr_batch2)
        sr_list = []
        for i in range(len(sr_batch1)):
            sr_list.append(sr_batch1[i:i+1])

        for i in range(len(sr_batch2)):
            sr_list.append(sr_batch2[i:i+1])

        # sr_list = [sr_batch[i:i+1] for i in range(len(sr_batch))]
        # sr_list = [self.model.forward(aug) for aug in lr_list]
        for i in range(len(sr_list)):
            if i > 3:
                sr_list[i] = _transform(sr_list[i], 't')
            if i % 4 > 1:
                sr_list[i] = _transform(sr_list[i], 'h')
            if (i % 4) % 2 == 1:
                sr_list[i] = _transform(sr_list[i], 'v')

        output_cat = torch.cat(sr_list, dim=0)
        output = output_cat.mean(dim=0, keepdim=True)

        return output


class L1_Charbonnier_loss(nn.Module):
    """L1 Charbonnierloss."""
    def __init__(self, reduction='mean'):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-6
        self.reduction = reduction

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt( diff * diff + self.eps )
        if self.reduction == 'mean':
            loss = torch.mean(error)
        else:
            loss = torch.sum(error)
        return loss


class YUV_loss(nn.Module):
    """L1 Charbonnierloss."""
    def __init__(self, reduction='mean'):
        super(YUV_loss, self).__init__()
        self.eps = 1e-6
        self.reduction = reduction

    def forward(self, X, Y):
        diff_uv = 0
        diff_y = torch.add(X[:, 0:1, :, :], -Y[:, 0:1, :, :])
        diff_uv_x2 = torch.add(X[:, 1:, :, :], -Y[:, 1:, :, :])
        diff_uv += diff_uv_x2[:, :, 0::2, 0::2]
        diff_uv += diff_uv_x2[:, :, 1::2, 0::2]
        diff_uv += diff_uv_x2[:, :, 0::2, 1::2]
        diff_uv += diff_uv_x2[:, :, 1::2, 1::2]
        diff_uv = torch.mul(diff_uv, 1/4.)

        error_y = torch.sqrt(diff_y * diff_y + self.eps)
        error_uv = torch.sqrt(diff_uv * diff_uv + self.eps)

        if self.reduction == 'mean':
            loss_y = torch.mean(error_y)
            loss_uv = torch.mean(error_uv)
        else:
            loss_y = torch.sum(error_y)
            loss_uv = torch.sum(error_uv)

        loss = loss_y + loss_uv
        return loss