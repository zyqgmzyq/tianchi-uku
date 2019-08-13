from __future__ import print_function
import argparse
import math
from math import log10
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from utils import tensor2img, calculate_psnr
from torch.utils.data import DataLoader
from net import Net as WDSR
from data import get_training_set
from data import get_eval_set
import pdb
import socket
import time
# from test import chop_forward, save_img, x8_forward

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--upscale_factor', type=int, default=4, help="super resolution upscale factor")
parser.add_argument('--batchSize', type=int, default=4, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--nEpochs', type=int, default=200, help='number of epochs to train for')
parser.add_argument('--snapshots', type=int, default=5, help='Snapshots')
parser.add_argument('--start_iter', type=int, default=1, help='Starting Epoch')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning Rate. Default=0.01')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--threads', type=int, default=1, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--gpus', default=1, type=int, help='number of gpu')
parser.add_argument('--data_dir', type=str, default='../dataset/')
parser.add_argument('--data_augmentation', type=bool, default=True)
parser.add_argument('--hr_train_dataset', type=str, default='train_gt_bmp')
parser.add_argument('--lr_train_dataset', type=str, default='train_low_bmp')
parser.add_argument('--model_type', type=str, default='WDSR')
parser.add_argument('--patch_size', type=int, default=64, help='Size of cropped HR image')
parser.add_argument('--pretrained_sr', default='MIX2K_LR_aug_x4dl10DBPNITERtpami_epoch_399.pth', help='sr pretrained base model')
parser.add_argument('--pretrained', type=bool, default=False)
parser.add_argument('--save_folder', default='weights/', help='Location to save checkpoint models')
parser.add_argument('--prefix', default='tpami_residual_filter8', help='Location to save checkpoint models')
parser.add_argument('--eval_dir', type=str, default='Input')
parser.add_argument('--eval_output', default='Results/', help='Location to save checkpoint models')
parser.add_argument('--hr_eval_dataset', type=str, default='val_gt_bmp')
parser.add_argument('--lr_eval_dataset', type=str, default='val_low_bmp')

opt = parser.parse_args()
gpus_list = range(opt.gpus)
hostname = str(socket.gethostname())
cudnn.benchmark = True
print(opt)


def train(epoch):
    epoch_loss = 0
    total_psnr = 0.0
    model.train()
    for iteration, batch in enumerate(training_data_loader, 1):
        input, target, bicubic = Variable(batch[0]), Variable(batch[1]), Variable(batch[2])
        if cuda:
            input = input.cuda(gpus_list[0])
            target = target.cuda(gpus_list[0])
            bicubic = bicubic.cuda(gpus_list[0])

        optimizer.zero_grad()
        t0 = time.time()
        prediction = model(input)

        loss = criterion(prediction, target)
        # prediction1 = tensor2img(prediction)/255.
        # target1 = tensor2img(target)/255.
        # psnr = calculate_psnr(prediction1 * 255, target1 * 255)
        # total_psnr += psnr

        t1 = time.time()
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()

        print("===> Epoch[{}]({}/{}): Loss: {:.4f} || Timer: {:.4f} sec."
              .format(epoch, iteration, len(training_data_loader), loss.data, (t1 - t0)))
        # print("psnr: {:.4f}".format(psnr))

    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(training_data_loader)))
    # print("avg_psnr: {:.4f}".format(total_psnr/len(training_data_loader)))


# def test():
#     model.eval()
#     avg_psnr = 0
#     for batch in eval_data_loader:
#         with torch.no_grad():
#             input, bicubic, name = Variable(batch[0]), Variable(batch[1]), batch[2]
#         if cuda:
#             input = input.cuda(gpus_list[0])
#             bicubic = bicubic.cuda(gpus_list[0])
#
#         t0 = time.time()
#         if opt.chop_forward:
#             with torch.no_grad():
#                 prediction = chop_forward(input, model, opt.upscale_factor)
#         else:
#             if opt.self_ensemble:
#                 with torch.no_grad():
#                     prediction = x8_forward(input, model)
#             else:
#                 with torch.no_grad():
#                     prediction = model(input)
#
#         criterion = nn.L1Loss()
#         mse = criterion(prediction, bicubic)
#         psnr = 10 * log10(1 / mse.item())
#         avg_psnr += psnr
#         t1 = time.time()
#         print("===> Processing: %s || Timer: %.4f sec." % (name[0], (t1 - t0)))
#         save_img(prediction.cpu().data, name[0])
#     print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(eval_data_loader)))


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


def checkpoint(epoch):
    model_out_path = opt.save_folder+opt.hr_train_dataset+opt.model_type+"_epoch_{}.pth".format(epoch)
    torch.save(model.state_dict(), model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


if __name__ == '__main__':
    cuda = opt.gpu_mode
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)

    print('===> Loading datasets')
    train_set = get_training_set(opt.data_dir, opt.hr_train_dataset, opt.lr_train_dataset,
                                 opt.upscale_factor, opt.patch_size, opt.data_augmentation)
    training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads,
                                      batch_size=opt.batchSize, shuffle=True)

    # eval_set = get_eval_set(opt.eval_dir, opt.hr_eval_dataset, opt.lr_eval_dataset, opt.upscale_factor)
    # eval_data_loader = DataLoader(dataset=eval_set, num_workers=opt.threads, batch_size=opt.testBatchSize,
    #                                  shuffle=False)

    print('===> Building model ', opt.model_type)
    if opt.model_type == 'WDSR':
        model = WDSR(num_channels=3, feat=64, scale=opt.upscale_factor, training='True')

    model = torch.nn.DataParallel(model, device_ids=gpus_list)
    criterion = nn.L1Loss()

    print('---------- Networks architecture -------------')
    print_network(model)
    print('----------------------------------------------')

    if opt.pretrained:
        model_name = os.path.join(opt.save_folder + opt.pretrained_sr)
        if os.path.exists(model_name):
            # model= torch.load(model_name, map_location=lambda storage, loc: storage)
            model.load_state_dict(torch.load(model_name, map_location=lambda storage, loc: storage))
            print('Pre-trained SR model is loaded.')

    if cuda:
        model = model.cuda(gpus_list[0])
        criterion = criterion.cuda(gpus_list[0])

    optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-8)

    for epoch in range(opt.start_iter, opt.nEpochs + 1):
        train(epoch)

        # learning rate is decayed by a factor of 10 every half of total epochs
        if (epoch+1) % (opt.nEpochs/2) == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] /= 10.0
            print('Learning rate decay: lr={}'.format(optimizer.param_groups[0]['lr']))

        if (epoch+1) % (opt.snapshots) == 0:
            checkpoint(epoch)


