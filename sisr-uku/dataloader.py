import os
import glob
from torch.utils.data.dataset import Dataset
import torch
import scipy.misc as misc
import random
import numpy as np
from config import args
from util import logging

flag = 's{}_{}{}_ps{}_bs{}_loss{}'.format(args.scale, args.model.lower(), args.mark, args.patch_size, args.batch_size,
                                          args.loss.upper())
logger = 'logs/{}.txt'.format(flag)

random.seed(args.seed)


def augment(l, hflip=True, rot=True):
    hflip = hflip and random.random() < 0.5  # 水平翻转是随便的，
    vflip = rot and random.random() < 0.5  # 垂直翻转是认真的
    rot90 = rot and random.random() < 0.5

    def _augment(img):
        if hflip: img = img[:, ::-1, :].copy()  # 左右颠倒
        if vflip: img = img[::-1, :, :].copy()  # 上下颠倒
        if rot90: img = img.transpose(1, 0, 2).copy()  # 直接进行了逆时针90和垂直翻转
        return img

    return [_augment(_l) for _l in l]


def get_patch(img_in, img_tar):
    ih, iw = img_in.shape[:2]  # 获取高宽 h w

    ip = args.patch_size
    tp = args.patch_size * args.scale
    ix = random.randrange(0, iw - ip + 1)  # 索引值不能超过放不下一个patch，这是输入的索引
    iy = random.randrange(0, ih - ip + 1)  # 高也是一样，图片不能超过范围
    tx, ty = args.scale * ix, args.scale * iy  # 这是输出的索引

    img_in = img_in[iy:iy + ip, ix:ix + ip, :]  # 输入进行取块
    img_tar = img_tar[ty:ty + tp, tx:tx + tp, :]  # 输出进行取块，这是可截取的范围

    return img_in, img_tar  # 返回取块后的结果


def hr_transform(img):
    img = img.transpose((2, 0, 1))
    img = img.astype(np.float32) /args.sd - args.mean
    img = torch.from_numpy(img).float()
    return img


def lr_transform(img):
    img = img.transpose((2, 0, 1))
    img = img.astype(np.float32) /args.sd - args.mean
    img = torch.from_numpy(img).float()
    return img


def read_img(img_path, dformate='numpy'):
    if dformate == 'numpy':
        img = np.load(img_path)  # ycbcr
    else:
        img = misc.imread(img_path)  # bmp for rgb
    return img


class TrainDataset(Dataset):
    def __init__(self, train_path):
        super(TrainDataset, self).__init__()
        if args.debug:
            args.dN = 2
        self.list_hr = sorted(glob.glob(os.path.join(train_path, '*_h_GT_*.npy')))[:-args.dN:args.dR]
        self.list_lr = [hr.replace('h_GT', 'l') for hr in self.list_hr]
        print('1', self.list_lr)
        if args.debug:
            self.list_lr = self.list_lr[:32]
            self.list_hr = self.list_hr[:32]
        print('data lenght: ', len(self.list_lr) * args.repeat)
        logging(logger, 'data lenght: {}'.format(len(self.list_lr)))
        if args.load_mem == 'all':
            self.lr_np = [read_img(lr) for lr in self.list_lr]
            self.hr_np = [read_img(hr) for hr in self.list_hr]
        elif args.load_mem == 'lr':
            self.lr_np = [read_img(lr) for lr in self.list_lr]
            self.hr_np = []

    def __getitem__(self, index):
        index = index % len(self.list_lr)

        if args.load_mem == 'all':
            lr = self.lr_np[index]
            hr = self.hr_np[index]
        elif args.load_mem == 'lr':
            lr = self.lr_np[index]
            hr = read_img(self.list_hr[index])
        else:
            lr = read_img(self.list_lr[index])
            hr = read_img(self.list_hr[index])
        lr, hr = get_patch(lr, hr)  # 可以加多一个计数器来进行尺寸变化

        lr, hr = augment([lr, hr])  # h, w, c

        lr = lr_transform(lr)
        hr = hr_transform(hr)

        return lr, hr

    def __len__(self):
        return len(self.list_lr) * args.repeat


class ValDataset(Dataset):
    def __init__(self, val_path):
        super(ValDataset, self).__init__()
        self.list_hr = sorted(glob.glob(os.path.join(val_path, '*_h_GT_*.npy')))[-args.dN::args.dS]
        self.list_lr = [hr.replace('h_GT', 'l') for hr in self.list_hr]

        if args.debug:
            self.list_lr = self.list_lr[:10]
            self.list_hr = self.list_hr[:10]

        self.lr_np = [read_img(lr) for lr in self.list_lr]  # 直接加载到内存了，反正一直要用
        self.hr_np = [read_img(hr) for hr in self.list_hr]

    def __getitem__(self, index):
        lr = self.lr_np[index]
        hr = self.hr_np[index]
        # lr, hr = get_patch(lr, hr)  # 直接分片省时间
        lr = lr_transform(lr)  # 会发生归一化

        return lr, hr

    def __len__(self):
        return len(self.list_lr)


class TestDataset(object):
    def __init__(self, infer_path):
        super(TestDataset, self).__init__()
        if args.output_path == '':
            # self.list_lr = sorted(glob.glob(os.path.join(infer_path, 'lr', '*.npy')))[-args.dN::args.dS]
            # self.list_hr = sorted(glob.glob(os.path.join(infer_path, 'hr', '*.npy')))[-args.dN::args.dS]

            self.list_hr = sorted(glob.glob(os.path.join(infer_path, '*_h_GT_*.npy')))[:10]
            self.list_lr = [hr.replace('h_GT', 'l') for hr in self.list_hr]
        else:
            self.list_lr = sorted(glob.glob(os.path.join(infer_path, '*_l_*.npy')))
            self.list_hr = self.list_lr

    def __getitem__(self, index):
        lr = read_img(self.list_lr[index])
        hr = read_img(self.list_hr[index])

        lr = lr_transform(lr)  # 会发生归一化
        return lr, hr, self.list_lr[index]

    def __len__(self):
        return len(self.list_lr)
