import torchvision.models as models
import torch.nn as nn
import argparse
from data import ClsData
from torch.backends import cudnn
from torch.utils.data import DataLoader
import torch
import numpy as np
from importlib import import_module
import torch.optim as optim
from utils.utils import init_weights, save_checkpoint


def trainClsConfig():
    parser = argparse.ArgumentParser(description='Train a cls network')
    parser.add_argument('--batchSize', dest='batchSize', help='batchSize', default=16, type=int)
    parser.add_argument("--threads", type=int, default=8, help="threads for data loader to use. Default=8")
    parser.add_argument("--resume", default="", type=str, help="path to checkpoint (default: none)")
    parser.add_argument('--network', dest='network', help='network file name', default="wdsr", type=str)
    parser.add_argument('--load_prefix', dest='load_prefix', help='load_prefix', default='', type=str)
    parser.add_argument('--load_epoch', dest='load_epoch', help='load_epoch', default=-1, type=int)
    parser.add_argument("--checkpoint", required=True, type=str, help="path to save checkpoints")
    parser.add_argument('--numEpoch', dest='numEpoch', type=int, default=1000, help='numEpoch')
    parser.add_argument('--lr', dest='lr', type=float, default=1e-4, help='lr')
    parser.add_argument('--opt', dest='opt', type=str, default='adam', help='optmiser,RMSProp,Nadam,adam,sgd')
    parser.add_argument('--data_root', dest='data_root', type=str, default='../data/', help='data_root')
    parser.add_argument('--patch_size', dest='patch_size', type=str, default='128,128', help='patch_size')
    parser.add_argument('--scale', dest='scale', type=int, default=4, help='scale')
    parser.add_argument('--shave', dest='shave', type=int, default=4, help='shave')
    parser.add_argument('--numblock', dest='numblock', type=int, default=16, help='numblock')
    parser.add_argument('--numfeature', dest='numfeature', type=int, default=128, help='numfeature')
    parser.add_argument('--rotate', dest='rotate', type=int, default=1, help='is rotate sample')
    parser.add_argument('--N', dest='N', type=int, default=2, help='2*N+1 is the length of video')
    parser.add_argument('--prefix', dest='prefix', type=str, default='models/', help='prefix')
    parser.add_argument('--wd', dest='wd', type=float, default=0.00001, help='weight decay')
    parser.add_argument("--cuda", default=True, action="store_true", help="Use cuda?")
    parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number. Default=1")
    parser.add_argument("--nEpochs", type=int, default=2000, help="number of epochs to train. Default=2000")

    args = parser.parse_args()
    return args


KWAI_SEED = 666

config = trainClsConfig()
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
batchSize = config.batchSize
cls_net = import_module('models.resnet').get_model()
init_weights(cls_net, 'normal')
print(cls_net)

dataIter = ClsData(config.data_root+"train/trainPair2.txt",
                  data_name=['data_y', 'data_uv'],
                  label_name=['label_y', 'label_uv', 'label_cls'],
                  patch_size=config.patch_size,
                  frames=config.N,
                  scale=config.scale,
                  isRotate=config.rotate)

testIter = ClsData(config.data_root+"train/testPair2.txt",
                  data_name=['data_y', 'data_uv'],
                  label_name=['label_y', 'label_uv', 'label_cls'],
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


if config.cuda:
    cls_net = cls_net.cuda()
    cls_net = nn.DataParallel(cls_net)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(cls_net.parameters(), lr=config.lr, betas=(0.5, 0.999))

total_step = len(training_data_loader)

for epoch in range(config.start_epoch, config.nEpochs + 1):
    print("epoch =", epoch, "lr =", optimizer.param_groups[0]["lr"])
    cls_net.train()

    for iteration, batch in enumerate(training_data_loader):
        steps = len(training_data_loader) * (epoch - 1) + iteration
        data_y = batch[0]
        data_uv = batch[2]
        cls_label = batch[4]

        data_y = data_y.to(device)
        data_uv = data_uv.to(device)
        cls_label = cls_label.to(device)

        outputs = cls_net(data_y, data_uv)
        # print(len(outputs), len(cls_label))
        # print(outputs.size(), cls_label.squeeze().size())
        loss = criterion(outputs, cls_label.squeeze())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iteration % 20 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch, config.numEpoch, iteration + 1, total_step, loss.item()))

    if epoch % 2 == 0:
        save_checkpoint(cls_net, None, epoch, config.checkpoint)
        correct = 0
        total = 0
        cls_net.eval()
        for iteration, batch in enumerate(test_data_loader, 1):
            data_y = batch[0]
            data_uv = batch[2]
            cls_label = batch[4]

            data_y = data_y.to(device)
            data_uv = data_uv.to(device)
            cls_label = cls_label.to(device)

            with torch.no_grad():
                outputs = cls_net(data_y, data_uv)
            _, predicted = torch.max(outputs.data, 1)
            total += cls_label.size(0)
            correct += (predicted == cls_label).sum().item()

        print('Test Accuracy  on the test images: {} %'.format(100 * correct / total))

