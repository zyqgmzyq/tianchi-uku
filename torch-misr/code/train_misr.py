from comet_ml import Experiment
import os
import sys
import cv2
import os.path
import logging
import argparse
import math
import numpy as np
import random as rd
from PIL import Image
import importlib
import argparse

import torch
import torch.nn as nn
from torch.backends import cudnn
import torch.optim as optim
from importlib import import_module
from torch.utils.data import DataLoader
from data import DLData
from utils.utils import init_weights, save_checkpoint
from tqdm import tqdm


def trainMISRConfig():
    parser = argparse.ArgumentParser(description='Train a sisr network')
    
    parser.add_argument('--batchSize', dest='batchSize', help='batchSize', default=16, type=int)

    parser.add_argument("--threads", type=int, default=8, help="threads for data loader to use. Default=8")

    parser.add_argument("--resume", default="", type=str, help="path to checkpoint (default: none)")
    
    parser.add_argument('--network', dest='network', help='network file name', default="rcan", type=str)
    
    parser.add_argument('--load_prefix', dest='load_prefix', help='load_prefix', default='', type=str)
    
    parser.add_argument('--load_epoch', dest='load_epoch', help='load_epoch', default=-1, type=int)

    parser.add_argument("--checkpoint", required=True, type=str, help="path to save checkpoints")

    parser.add_argument('--numEpoch', dest='numEpoch', type=int, default=1000, help='numEpoch')

    parser.add_argument('--lr', dest='lr', type=float, default=1e-4, help='lr')
       
    parser.add_argument('--opt', dest='opt', type=str, default='adam', help='optmiser,RMSProp,Nadam,adam,sgd')
    
    parser.add_argument('--data_root', dest='data_root', type=str, default='../data/', help='data_root')
    
    parser.add_argument('--patch_size', dest='patch_size', type=str, default='64,64', help='patch_size')
    
    parser.add_argument('--scale', dest='scale', type=int, default=4, help='scale')
    
    parser.add_argument('--shave', dest='shave', type=int, default=4, help='shave')

    parser.add_argument('--numblock', dest='numblock', type=int, default=16, help='numblock')
    
    parser.add_argument('--numfeature', dest='numfeature', type=int, default=128, help='numfeature')
    
    parser.add_argument('--rotate', dest='rotate', type=int, default=1, help='is rotate sample')
    
    parser.add_argument('--N', dest='N', type=int, default=2, help='2*N+1 is the length of video')
    
    parser.add_argument('--prefix', dest='prefix', type=str, default='checkpoints/', help='prefix')
    
    parser.add_argument('--wd', dest='wd', type=float, default=0.00001, help='weight decay')

    parser.add_argument("--cuda", default=True, action="store_true", help="Use cuda?")

    parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number. Default=1")

    parser.add_argument("--nEpochs", type=int, default=2000, help="number of epochs to train. Default=2000")

    args = parser.parse_args()
    return args
###########################################################################################


experiment = Experiment(api_key="1EF8pAM91tS7x38HfhXcFw2d9",
                        project_name="general", workspace="zyqgmzyq", auto_output_logging=None)

KWAI_SEED = 666

config = trainMISRConfig()
print(config)
patch_size = config.patch_size.split(',')
config.patch_size = (int(patch_size[0]), int(patch_size[1]))

cuda = config.cuda
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

if cuda:
    torch.cuda.manual_seed(KWAI_SEED)

cudnn.benchmark = True

device = 'cuda' if torch.cuda.is_available() else 'cpu'


model_name = config.network+"_b"+str(config.numblock)+"_f"+str(config.numfeature)+"_x"+str(config.scale)
if config.rotate:
    model_name = model_name+'_rotate'
if not os.path.exists(config.prefix+'/misr_'+model_name):
    os.mkdir(config.prefix+'/misr_'+model_name)
config.save_prefix = config.prefix+'/misr_'+model_name+'/misr_'+model_name
head = '%(asctime)-15s Node[0] %(message)s'


opt = config.opt
batchSize = config.batchSize
get_model = import_module('models.' + config.network.lower()).get_model
net = get_model(num_block=config.numblock, num_feature=config.numfeature, scale=config.scale, is_train=True)
init_weights(net, 'normal')
print(net)

dataIter = DLData(config.data_root+"train/trainPair2.txt",
                  data_name=['data_y', 'data_uv'],
                  label_name=['label_y', 'label_uv'],
                  patch_size=config.patch_size,
                  frames=config.N,
                  scale=config.scale,
                  isRotate=config.rotate)

testIter = DLData(config.data_root+"train/testPair2.txt",
                  data_name=['data_y', 'data_uv'],
                  label_name=['label_y', 'label_uv'],
                  patch_size=config.patch_size,
                  frames=config.N,
                  scale=config.scale,
                  train=True,
                  isRotate=0)

training_data_loader = DataLoader(
    dataset=dataIter,
    batch_size=config.batchSize,
    pin_memory=True,
    shuffle=True,
    num_workers=int(config.threads)
)

test_data_loader = DataLoader(
    dataset=testIter,
    batch_size=1,
    num_workers=1
)

if config.resume:
    if os.path.isfile(config.resume):
        print("======> loading checkpoint at '{}'".format(config.resume))
        checkpoint = torch.load(config.resume)
        net.load_state_dict(checkpoint["state_dict_model"])
    else:
        print("======> founding no checkpoint at '{}'".format(config.resume))

if config.cuda:
    net = net.cuda()
    net = nn.DataParallel(net)

optimizer = optim.Adam(net.parameters(), lr=config.lr, betas=(0.5, 0.999))

with experiment.train():
    for epoch in range(config.start_epoch, config.nEpochs + 1):
        print("epoch =", epoch, "lr =", optimizer.param_groups[0]["lr"])
        net.train()

        for iteration, batch in enumerate(training_data_loader):
            steps = len(training_data_loader) * (epoch - 1) + iteration
            data_y = batch[0]
            label_y = batch[1]
            data_uv = batch[2]
            label_uv = batch[3]

            data_y = data_y.to(device)
            data_uv = data_uv.to(device)
            label_y = label_y.to(device)
            label_uv = label_uv.to(device)

            net.zero_grad()
            data_y, data_uv, loss_y, loss_uv = net(data_y, data_uv, label_y, label_uv)
            loss = loss_y + loss_uv
            loss.sum().backward()
            optimizer.step()

            loss = loss_y + loss_uv

            if iteration % 20 == 0:
                print("Epoch[{}]({}/{}): loss {:.4f} loss_y {:.4f} loss_uv {:.4f}".format(
                    epoch, iteration, len(training_data_loader), loss.mean().item(), loss_y.mean().item(), loss_uv.mean().item())
                )
                experiment.log_metrics({'loss': loss.mean().item(),
                                        'loss_y': loss_y.mean().item(),
                                        'loss_uv': loss_uv.mean().item()}, step=steps)

        if epoch % 2 == 0:
            save_checkpoint(net, None, epoch, config.checkpoint)
            mean_psnr_y = 0
            mean_psnr_yuv = 0
            net.eval()
            for iteration, batch in enumerate(test_data_loader, 1):
                data_y = batch[0]
                label_y = batch[1]
                data_uv = batch[2]
                label_uv = batch[3]

                data_y = data_y.to(device)
                data_uv = data_uv.to(device)
                label_y = label_y.to(device)
                label_uv = label_uv.to(device)

                with torch.no_grad():
                    sr_y, sr_uv, loss_y, loss_uv = net(data_y, data_uv, label_y, label_uv)
                diff_y = (sr_y - label_y) / 255.
                diff_uv = (sr_uv - label_uv) / 255.
                diff_y = diff_y.squeeze(0).cpu()
                diff_uv = diff_uv.squeeze(0).cpu()
                diff_y = diff_y[..., config.shave:-config.shave, config.shave:-config.shave]
                diff_uv = diff_uv[..., config.shave:-config.shave, config.shave:-config.shave]
                mse_y = diff_y.pow(2).mean()
                mse_yuv = (mse_y + diff_uv.pow(2).mean()*0.5) / 1.5
                psnr_y = -10 * np.log10(mse_y)
                psnr_yuv = -10 * np.log10(mse_yuv)
                mean_psnr_y += psnr_y
                mean_psnr_yuv += psnr_yuv
            mean_psnr_y /= len(test_data_loader)
            mean_psnr_yuv /= len(test_data_loader)
            print("Epoch {} for psnr_y {:.4f} psnr_yuv {:.4f}".format(epoch, mean_psnr_y, mean_psnr_yuv))
            experiment.log_metrics({'test_psnr_y': mean_psnr_y.item(),
                                    'test_psnr_yuv': mean_psnr_yuv.item()}, step=epoch)
