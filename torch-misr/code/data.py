import os
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import pickle
import random


def readImg(listPath, index):
    data = open(listPath, 'r')
    imgPair = []
    datas = data.readlines()
    if len(index) == 2:
        datas = datas[index[0]:index[1]]
    print('loading '+str(len(datas))+' imgs from '+listPath)
    for d in datas:
        d = d.strip("\n")
        d = d.strip("\000")
        d = d.strip("\r")
        d = d.split(" ")
        imgPair.append([d[0], d[1]])
    data.close()
    return imgPair


class DLData(data.Dataset):
    def __init__(self,
                 imgPairList,
                 data_name=['data_y', 'data_uv'],
                 label_name=['label_y', 'label_uv'],
                 patch_size=(96, 96),
                 frames=2,
                 scale=4,
                 isRotate=1,
                 train=True,
                 index=[]):
        self.imgPairList = []
        _imgPairList = readImg(imgPairList, index)
        for i, p_l in enumerate(_imgPairList):
            if i % 5 == 0:
                self.imgPairList.append(p_l)

        self.data_name = data_name
        self.label_name = label_name

        self.patch_size = patch_size
        self.frames = frames
        self.scale = scale

        self.isRotate = isRotate

        self.train = train

    def __getitem__(self, idx):
        imgPair = self.imgPairList[idx]
        data_hr_y = []
        data_hr_uv = []
        data_lr_y = []
        data_lr_uv = []
        ix = -999
        iy = -999

        for n in range(self.frames*2+1):
            ind = n-self.frames
            frameID = int(imgPair[0].split('/')[-1][-11:-7])
            frameID += ind
            id_str = str(frameID)
            id_str = '0'*(4-len(id_str))+id_str
            hr_path = imgPair[0][3:-11] + id_str + '.pickle'
            lr_path = imgPair[1][3:-11] + id_str + '.pickle'

            while(not os.path.exists(hr_path) or not os.path.exists(lr_path)):
                if ind > 0:
                    frameID -= 1
                    id_str = str(frameID)
                    id_str = '0'*(4-len(id_str))+id_str
                    hr_path = imgPair[0][3:-11] + id_str + '.pickle'
                    lr_path = imgPair[1][3:-11] + id_str + '.pickle'
                if ind < 0:
                    frameID += 1
                    id_str = str(frameID)
                    id_str = '0'*(4-len(id_str))+id_str
                    hr_path = imgPair[0][3:-11] + id_str + '.pickle'
                    lr_path = imgPair[1][3:-11] + id_str + '.pickle'
                if ind == 0:
                    print("error "+hr_path)

            lr_yuv_fr = open(lr_path, 'rb')
            lr_yuv = pickle.load(lr_yuv_fr)

            scale = self.scale
            ih = lr_yuv[1].shape[0]
            iw = lr_yuv[1].shape[1]

            ph = self.patch_size[0]
            pw = self.patch_size[1]

            if ix == -999 or iy == -999:
                ix = random.randrange(1, iw - ph - 1)
                iy = random.randrange(1, ih - pw - 1)

            lr_y_crop = lr_yuv[0][iy*2:(iy+pw)*2, ix*2:(ix+ph)*2].copy() if self.train else lr_yuv[0].copy()
            lr_u_crop = lr_yuv[1][iy:(iy+pw), ix:(ix+ph)].copy() if self.train else lr_yuv[1].copy()
            lr_v_crop = lr_yuv[2][iy:(iy+pw), ix:(ix+ph)].copy() if self.train else lr_yuv[2].copy()
            lr_yuv_fr.close()

            if ind == 0:
                hr_yuv_fr = open(hr_path, 'rb')
                hr_yuv = pickle.load(hr_yuv_fr)
                hr_y_crop = hr_yuv[0][iy*scale*2:(iy+pw)*scale*2, ix*scale*2:(ix+ph)*scale*2].copy() if self.train else hr_yuv[0].copy()
                hr_u_crop = hr_yuv[1][iy*scale:(iy+pw)*scale, ix*scale:(ix+ph)*scale].copy() if self.train else hr_yuv[1].copy()
                hr_v_crop = hr_yuv[2][iy*scale:(iy+pw)*scale, ix*scale:(ix+ph)*scale].copy() if self.train else hr_yuv[2].copy()
                data_hr_y.append(hr_y_crop)
                data_hr_uv.append(hr_u_crop)
                data_hr_uv.append(hr_v_crop)
                hr_yuv_fr.close()

            data_lr_y.append(lr_y_crop)
            data_lr_uv.append(lr_u_crop)
            data_lr_uv.append(lr_v_crop)

        data_lr_y_np = np.array(data_lr_y, np.uint8, copy=False)
        data_hr_y_np = np.array(data_hr_y, np.uint8, copy=False)
        data_lr_uv_np = np.array(data_lr_uv, np.uint8, copy=False)
        data_hr_uv_np = np.array(data_hr_uv, np.uint8, copy=False)

        if self.isRotate:
            k = random.randint(0, 3)
            data_lr_y_np = np.rot90(data_lr_y_np, k=k, axes=(1, 2))
            data_hr_y_np = np.rot90(data_hr_y_np, k=k, axes=(1, 2))
            data_lr_uv_np = np.rot90(data_lr_uv_np, k=k, axes=(1, 2))
            data_hr_uv_np = np.rot90(data_hr_uv_np, k=k, axes=(1, 2))
            if random.randint(0, 1):
                data_lr_y_np = data_lr_y_np.transpose((0, 2, 1))
                data_hr_y_np = data_hr_y_np.transpose((0, 2, 1))
                data_lr_uv_np = data_lr_uv_np.transpose((0, 2, 1))
                data_hr_uv_np = data_hr_uv_np.transpose((0, 2, 1))
        data_lr_y_np = torch.from_numpy(np.ascontiguousarray(data_lr_y_np))
        data_hr_y_np = torch.from_numpy(np.ascontiguousarray(data_hr_y_np))
        data_lr_uv_np = torch.from_numpy(np.ascontiguousarray(data_lr_uv_np))
        data_hr_uv_np = torch.from_numpy(np.ascontiguousarray(data_hr_uv_np))
        return data_lr_y_np.float(), data_hr_y_np.float(), data_lr_uv_np.float(), data_hr_uv_np.float()

    def __len__(self):
        return len(self.imgPairList)


type_01 = []
type_23 = []


f = open("./type_label.txt", 'r')
for line in f.readlines():
    if int(line.split(' ')[1]) == 1 or int(line.split(' ')[1]) == 0:
        id_str = line.split(' ')[0]
        id_str = '0' * (5 - len(id_str)) + id_str
        type_01.append(id_str)
    else:
        id_str = line.split(' ')[0]
        id_str = '0' * (5 - len(id_str)) + id_str
        type_23.append(id_str)

print("01:", type_01)
print("23:", type_23)


class ClsData(data.Dataset):
    def __init__(self,
                 imgPairList,
                 data_name=['data_y', 'data_uv'],
                 label_name=['label_y', 'label_uv', 'label_cls'],
                 patch_size=(96, 96),
                 frames=2,
                 scale=4,
                 isRotate=1,
                 train=True,
                 index=[]):
        self.imgPairList = []
        _imgPairList = readImg(imgPairList, index)
        for i, p_l in enumerate(_imgPairList):
            if i % 5 == 0:
                self.imgPairList.append(p_l)

        self.data_name = data_name
        self.label_name = label_name

        self.patch_size = patch_size
        self.frames = frames
        self.scale = scale

        self.isRotate = isRotate

        self.train = train

    def __getitem__(self, idx):
        imgPair = self.imgPairList[idx]
        data_hr_y = []
        data_hr_uv = []
        data_lr_y = []
        data_lr_uv = []
        cls_label = []
        ix = -999
        iy = -999

        for n in range(self.frames*2+1):
            ind = n-self.frames
            frameID = int(imgPair[0].split('/')[-1][-11:-7])
            frameID += ind
            id_str = str(frameID)
            id_str = '0'*(4-len(id_str))+id_str
            hr_path = imgPair[0][3:-11] + id_str + '.pickle'
            lr_path = imgPair[1][3:-11] + id_str + '.pickle'

            while(not os.path.exists(hr_path) or not os.path.exists(lr_path)):
                if ind > 0:
                    frameID -= 1
                    id_str = str(frameID)
                    id_str = '0'*(4-len(id_str))+id_str
                    hr_path = imgPair[0][3:-11] + id_str + '.pickle'
                    lr_path = imgPair[1][3:-11] + id_str + '.pickle'
                if ind < 0:
                    frameID += 1
                    id_str = str(frameID)
                    id_str = '0'*(4-len(id_str))+id_str
                    hr_path = imgPair[0][3:-11] + id_str + '.pickle'
                    lr_path = imgPair[1][3:-11] + id_str + '.pickle'
                if ind == 0:
                    print("error "+hr_path)

            lr_yuv_fr = open(lr_path, 'rb')
            lr_yuv = pickle.load(lr_yuv_fr)

            scale = self.scale
            ih = lr_yuv[1].shape[0]
            iw = lr_yuv[1].shape[1]

            ph = self.patch_size[0]
            pw = self.patch_size[1]

            if ix == -999 or iy == -999:
                ix = random.randrange(1, iw - ph - 1)
                iy = random.randrange(1, ih - pw - 1)

            lr_y_crop = lr_yuv[0][iy*2:(iy+pw)*2, ix*2:(ix+ph)*2].copy() if self.train else lr_yuv[0].copy()
            lr_u_crop = lr_yuv[1][iy:(iy+pw), ix:(ix+ph)].copy() if self.train else lr_yuv[1].copy()
            lr_v_crop = lr_yuv[2][iy:(iy+pw), ix:(ix+ph)].copy() if self.train else lr_yuv[2].copy()
            lr_yuv_fr.close()

            if ind == 0:
                hr_yuv_fr = open(hr_path, 'rb')
                hr_yuv = pickle.load(hr_yuv_fr)
                hr_y_crop = hr_yuv[0][iy*scale*2:(iy+pw)*scale*2, ix*scale*2:(ix+ph)*scale*2].copy() if self.train else hr_yuv[0].copy()
                hr_u_crop = hr_yuv[1][iy*scale:(iy+pw)*scale, ix*scale:(ix+ph)*scale].copy() if self.train else hr_yuv[1].copy()
                hr_v_crop = hr_yuv[2][iy*scale:(iy+pw)*scale, ix*scale:(ix+ph)*scale].copy() if self.train else hr_yuv[2].copy()
                data_hr_y.append(hr_y_crop)
                data_hr_uv.append(hr_u_crop)
                data_hr_uv.append(hr_v_crop)
                frame_id = hr_path.split('/')[-1].split('_')[1][:5]
                if type_01.__contains__(frame_id):
                    cls_label.append(0)
                else:
                    cls_label.append(1)
                hr_yuv_fr.close()

            data_lr_y.append(lr_y_crop)
            data_lr_uv.append(lr_u_crop)
            data_lr_uv.append(lr_v_crop)

        data_lr_y_np = np.array(data_lr_y, np.uint8, copy=False)
        data_hr_y_np = np.array(data_hr_y, np.uint8, copy=False)
        data_lr_uv_np = np.array(data_lr_uv, np.uint8, copy=False)
        data_hr_uv_np = np.array(data_hr_uv, np.uint8, copy=False)
        cls_label_np = np.array(cls_label, np.int, copy=False)

        if self.isRotate:
            k = random.randint(0, 3)
            data_lr_y_np = np.rot90(data_lr_y_np, k=k, axes=(1, 2))
            data_hr_y_np = np.rot90(data_hr_y_np, k=k, axes=(1, 2))
            data_lr_uv_np = np.rot90(data_lr_uv_np, k=k, axes=(1, 2))
            data_hr_uv_np = np.rot90(data_hr_uv_np, k=k, axes=(1, 2))
            if random.randint(0, 1):
                data_lr_y_np = data_lr_y_np.transpose((0, 2, 1))
                data_hr_y_np = data_hr_y_np.transpose((0, 2, 1))
                data_lr_uv_np = data_lr_uv_np.transpose((0, 2, 1))
                data_hr_uv_np = data_hr_uv_np.transpose((0, 2, 1))
        data_lr_y_np = torch.from_numpy(np.ascontiguousarray(data_lr_y_np))
        data_hr_y_np = torch.from_numpy(np.ascontiguousarray(data_hr_y_np))
        data_lr_uv_np = torch.from_numpy(np.ascontiguousarray(data_lr_uv_np))
        data_hr_uv_np = torch.from_numpy(np.ascontiguousarray(data_hr_uv_np))
        cls_label_np = torch.from_numpy(cls_label_np)
        return data_lr_y_np.float(), data_hr_y_np.float(), data_lr_uv_np.float(), \
               data_hr_uv_np.float(), cls_label_np.long()

    def __len__(self):
        return len(self.imgPairList)
