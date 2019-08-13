import time
import sys
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
from torch.utils.data import DataLoader

import model
from config import args
from torch.backends import cudnn
from util import logging, mkdirs
from util.psnr_torch import psnr_torch
from dataloader1 import TrainDataset, ValDataset
from model import L1_Charbonnier_loss
from model import YUV_loss

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
flag = 's{}_{}{}_ps{}_bs{}_loss{}'.format(args.scale, args.model.lower(), args.mark, args.patch_size, args.batch_size,
                                          args.loss.upper())
logger = 'logs/{}.txt'.format(flag)
ckpt = 'checkpoint/{}.ckpt'.format(flag)
mkdirs(logger.split('/')[0])
mkdirs(ckpt.split('/')[0])

mean_numpy = np.array(args.mean).reshape((1, 1, -1))
mean_torch = torch.from_numpy(mean_numpy).float().to(device)

if sys.platform == 'win32':
    num_workers = 0  # @div2k:0个worker 96s, 1个worker 97s, 2个worker 49s, 3个worker 31s, 4个worker 30s
    var_bs = 1
else:
    num_workers = 4
    var_bs = 1

train_set = TrainDataset(args.train_path)
val_set = ValDataset(args.val_path)
train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True, num_workers=num_workers)
val_loader = DataLoader(dataset=val_set, batch_size=var_bs, shuffle=False, num_workers=num_workers // 2)

if args.model.lower() == 'edsr' or args.model.lower() == 'wdsr':
    net = model.Model(n_resblocks=args.n_resblocks, n_feats=args.n_feats)
else:
    net = model.Model()
net.print()


if args.loss.lower() == 'l1':
    criterion = nn.L1Loss().to(device)
elif args.loss.lower() == 'yuv':
    criterion = YUV_loss().to(device)
elif args.loss.lower() == 'char':
    criterion = L1_Charbonnier_loss().to(device)
else:
    criterion = nn.MSELoss().to(device)

criterion2 = nn.L1Loss(reduction='none').to(device)
optimizer = optim.Adam(net.parameters(), lr=args.lr)  # weight_decay=1e-4, rcan default=0

# scheduler = lrs.StepLR(
#     optimizer,
#     step_size=20,  # 每经过这么多个epoch（默认200）就乘上gamma倍
#     gamma=0.5
# )

milestones = [int(e) for e in args.milestones.split('-')]  # 10-20-30-50 -> [10, 20, 30, 50]
scheduler = lrs.MultiStepLR(
    optimizer,
    # 固定迭代次数，达到每次乘下面的gamma  # hdnet用小迭代，dbcn用大迭代
    milestones=milestones,
    gamma=0.5
)

if args.ckpt:
    net.load(args.ckpt)
    for i in range(args.repoch):
        scheduler.step()
    print('restored ckpt from {} at step: {}'.format(args.ckpt if args.ckpt else ckpt, args.repoch))

now = time.strftime('%Y.%m.%d %H:%M:%S\n', time.localtime(time.time()))
logging(logger, args.message + now)

psnr_best = 0

print('begin training')
for epoch in range(args.repoch, args.epochs):
    scheduler.step()
    start_train = time.time()
    total_loss = 0
    cnt = 0
    for lr, hr in train_loader:
        # hr_temp =
        sr = net(lr.to(device))
        loss = criterion(sr, hr.to(device))
        total_loss += loss.data.item()
        cnt += 1
        # loss_each = criterion2(sr, hr.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # loss_each = loss_each.cpu().detach().numpy()
        # loss_each = np.mean(loss_each, axis=(1, 2, 3))
        # print(loss_each)
    end_train = time.time()

    start_val = time.time()
    with torch.no_grad():
        psnr_index = []  # 一个的
        for tlrs, thrs in val_loader:
            tsrs = net(tlrs.to(device)).permute(0, 2, 3, 1)
            for tsr, thr in zip(tsrs, thrs):
                tsr = torch.clamp((tsr.float() + mean_torch) * args.sd, 0, args.sd).round()  # 这个存的位数和numpy似乎有点不一样
                psnr_index.append(psnr_torch(tsr[:, :, 0], thr.to(device)[:, :, 0], args.scale))
        psnr_single = sum(psnr_index) / len(psnr_index) if psnr_index else -1
    end_val = time.time()

    message = 'epoch:{}, lr: {:.2e}, loss:{:.2e}, psnr:{:.3f} train time:{:.0f}s val time:{:.2f}s\n'.\
        format(epoch, scheduler.get_lr()[0], total_loss / cnt, psnr_single, end_train - start_train, end_val - start_val)

    logging(logger, message)
    print(message.strip())

    if psnr_single > psnr_best:
        psnr_best = psnr_single
        net.save(ckpt)  # save highest scored model
        model_save_message = 'model saved at epoch {}, psnr_best is {:.4f}\n'.format(epoch, psnr_best)
        logging(logger, model_save_message)
        print(model_save_message, end='')

    net.save('checkpoint/{}{}_newest.ckpt'.format(args.model.lower(), args.mark))  # save newest model
