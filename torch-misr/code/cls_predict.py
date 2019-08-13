import os
import torch
import argparse
from importlib import import_module
from six.moves.queue import Queue
from threading import Thread
import pickle
import numpy as np
import math
import torch
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Test Script")
parser.add_argument("--output", type=str, required=True, help="path to save output images")
parser.add_argument("--checkpoint", type=str, required=True, help="path to load model checkpoint")
parser.add_argument('--network', dest='network', help='network file name', default="wdsr", type=str)
parser.add_argument('--numblock', dest='numblock', type=int, default=16, help='numblock')
parser.add_argument('--numfeature', dest='numfeature', type=int, default=128, help='numfeature')
parser.add_argument('--scale', dest='scale', type=int, default=4, help='scale')
parser.add_argument('--data_root', dest='data_root', type=str, default='../data/test2/yuvPickle', help='data_root')
parser.add_argument('--N', dest='N', type=int, default=2, help='2*N+1 is the length of video')
parser.add_argument("--save_result", default=True, action="store_true", help="save result?")
config = parser.parse_args()
print(config)


def eachFile(filepath):
    pathDir = os.listdir(filepath)
    subfile = []
    for allDir in pathDir:
        child = os.path.join('%s/%s' % (filepath, allDir))
        subfile.append(child)
    return subfile


def loadimg(path_queue, data_queue, threadID, N=2):
    print('activate thread ' + str(threadID) + ' loading imgs')
    while True:
        path = path_queue.get()
        # print(path)
        if len(path) == 1:
            lr_center_path = path[0]
        else:
            lr_center_path = path[1]

        lr_y = []
        lr_uv = []
        for n in range(N * 2 + 1):
            ind = n - N
            frameID = int(lr_center_path.split('/')[-1][-11:-7])
            frameID += ind
            id_str = str(frameID)
            id_str = '0' * (4 - len(id_str)) + id_str
            lr_path = lr_center_path[:-11] + id_str + '.pickle'
            while (not os.path.exists(lr_path)):
                if ind > 0:
                    frameID -= 1
                    id_str = str(frameID)
                    id_str = '0' * (4 - len(id_str)) + id_str
                    lr_path = lr_center_path[:-11] + id_str + '.pickle'
                if ind < 0:
                    frameID += 1
                    id_str = str(frameID)
                    id_str = '0' * (4 - len(id_str)) + id_str
                    lr_path = lr_center_path[:-11] + id_str + '.pickle'
                if ind == 0:
                    print("error " + lr_path)

            lr_yuv_fr = open(lr_path, 'rb')
            lr_yuv = pickle.load(lr_yuv_fr)
            lr_y.append(lr_yuv[0])
            lr_uv.append(lr_yuv[1])
            lr_uv.append(lr_yuv[2])
            lr_yuv_fr.close()

        if len(path) == 1:  # predict data
            data_queue.put([[lr_y, lr_uv]] + path)
        else:  # val data
            hr_yuv_fr = open(path[0], 'rb')
            hr_yuv = pickle.load(hr_yuv_fr)
            data_queue.put([hr_yuv, [lr_y, lr_uv]] + path)
            hr_yuv_fr.close()


cls_net = import_module('models.resnet').get_model()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

checkpoint = torch.load(config.checkpoint)
cls_net.load_state_dict(checkpoint["state_dict_model"])

cls_net = cls_net.to(device)
cls_net = torch.nn.DataParallel(cls_net)
cls_net = cls_net.eval()

get_model = import_module('models.' + config.network.lower()).get_model
net = get_model(num_block=config.numblock, num_feature=config.numfeature, scale=config.scale, is_train=False)
net2 = get_model(num_block=config.numblock, num_feature=config.numfeature, scale=config.scale, is_train=False)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

checkpoint = torch.load("./model1/290.pth")
checkpoint2 = torch.load("./model2/320.pth")
net.load_state_dict(checkpoint["state_dict_model"])
net.load_state_dict(checkpoint2["state_dict_model"])

net = net.to(device)
net2 = net2.to(device)
net = torch.nn.DataParallel(net)
net2 = torch.nn.DataParallel(net2)
net = net.eval()
net2 = net2.eval()


testvid = [850, 899]
imgList = []
files = eachFile(config.data_root)
for f in files:
    if '_l_' not in f:
        continue
    name = f.split('/')[-1]
    video_ind = int(name.split('_')[1])
    if testvid[0] <= video_ind <= testvid[1]:
        if os.path.exists(f.replace('_l_', '_h_GT_')):
            imgList.append([f.replace('_l_', '_h_GT_'), f])
        else:
            imgList.append([f])

print("Load image list len:", len(imgList))

path_queue = Queue()
data_queue = Queue()
result_queue = Queue()

for i in range(len(imgList)):
    path_queue.put(imgList[i])

loadThreads = 16
loaders = [Thread(target=loadimg, args=[path_queue, data_queue, i, config.N]) for i in range(loadThreads)]

for loader in loaders:
    loader.daemon = True
    loader.start()

count = 0
datas = []
while count != len(imgList):
    data = data_queue.get()
    datas.append(data)
    count += 1

cls_dict = {}
for data in tqdm(datas):
    results = []
    if len(data) == 2:
        lr_yuv = data[0]
    else:
        lr_yuv = data[1]
    y = np.array([lr_yuv[0]], np.uint8, copy=False)
    uv = np.array([lr_yuv[1]], np.uint8, copy=False)

    data_lr_y_np = torch.from_numpy(np.ascontiguousarray(y)).float()
    data_lr_uv_np = torch.from_numpy(np.ascontiguousarray(uv)).float()

    if not os.path.exists(config.output):
        os.makedirs(config.output)

    with torch.no_grad():
        outputs = cls_net(data_lr_y_np, data_lr_uv_np)
        _, predicted = torch.max(outputs.data, 1)
        lr, lr_path = data
        frame = lr_path.split('_l_')[0][-5:]
        print(frame)
        if frame not in cls_dict:
            cls_dict[frame] = 0
        if predicted.item() == 0:
            cls_dict[frame] -= 1
        else:
            cls_dict[frame] += 1
print(cls_dict)

        # if predicted.item() == 0:
        #     data_y, data_uv = net(data_lr_y_np, data_lr_uv_np)
        # else:
        #     data_y, data_uv = net2(data_lr_y_np, data_lr_uv_np)
        # sr = [data_y.cpu().numpy().squeeze(0), data_uv.cpu().numpy().squeeze(0)]
        # results.append([sr] + data)
        # result = results[0]
        # sr, lr, lr_path = result
        # if config.save_result:
        #     result_path = lr_path.replace('_l_', '_sr_')
        #     result_path = result_path.split('/')[-1]
        #     result_path = config.output + '/' + result_path
        #     fw = open(result_path, 'wb')
        #     pickle.dump(sr, fw, pickle.HIGHEST_PROTOCOL)
        #     print('save as ' + result_path)
        #     fw.close()

# python cls_predict.py --checkpoint ./cls/554.pth --output ../submit/yuvPickle
