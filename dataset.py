import random
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
from os.path import join
import os,sys
import cv2
from torchvision import transforms 
#from utils import prepare_image_cv2,tensor_to_np
import util
import torchvision


def perpare_image_cv2(im):
    im = np.array(im).astype(np.float32)
    im -= np.array((122.67891434,116.66876762,104.00698793))
    im = np.transpose(im,(2,0,1))
    return im


def LR_edge_generate(LR):
    gray = cv2.cvtColor(LR,cv2.COLOR_RGB2GRAY)
    LR_edge = cv2.Canny(gray,50,150)
    return LR_edge


class DIV2K_dataset(Dataset):
    def __init__(self,root_dir,patch_size = 128,scale=4,need_edge = False):

        self.scale = scale
        self.patch_size = patch_size
        self.HR_dir = join(root_dir,"dataset/train/youku_00000_00049_l")
        self.LR_dir = join(root_dir,"DIV2K_LR_bicubic/X%d"%self.scale)
        self.filelist = os.listdir(self.HR_dir)
        self.need_edge = need_edge
       

    def __getitem__(self,idx):
        HR_image = cv2.imread(join(self.HR_dir,self.filelist[idx]))
        LR_image = cv2.imread(join(self.LR_dir,self.filelist[idx]))

        HR_patch = self.patch_size
        LR_patch = self.patch_size // self.scale


        H,W,C = LR_image.shape
        
        #random crop
        rand_h = random.randint(0,max(0,H - LR_patch))
        rand_w = random.randint(0,max(0,W - LR_patch))
        img_LR = LR_image[rand_h:rand_h + LR_patch, rand_w:rand_w + LR_patch, : ]
        ran_h_HR,ran_w_HR = int(rand_h * self.scale), int(rand_w * self.scale)
        img_HR = HR_image[ran_h_HR:ran_h_HR + HR_patch, ran_w_HR:ran_w_HR + HR_patch, :] 
        
        img_LR,img_HR = util.augment([img_LR,img_HR],True,True)


        if img_HR.shape[2] == 3:
                # bgr -> rgb
                img_HR = img_HR[:, :, [2, 1, 0]]
                img_LR = img_LR[:, :, [2, 1, 0]]
        
        if self.need_edge:
            LR_edge = LR_edge_generate(img_LR)
        # to-do:归一化
        img_HR = torch.from_numpy(np.ascontiguousarray(perpare_image_cv2(img_HR)))
        img_LR = torch.from_numpy(np.ascontiguousarray(perpare_image_cv2(img_LR)))

        if self.need_edge:
            LR_edge = torch.from_numpy(np.ascontiguousarray(LR_edge)).float()
        # edge generate
        # LR edge
        
        if self.need_edge:
            return img_HR,img_LR,LR_edge
        else:
            return img_HR,img_LR


    def __len__(self):
        return len(self.filelist)


if __name__ == "__main__":
    dataset = DIV2K_dataset(root_dir = "./DIV2K",need_edge=True)
    dataloader = DataLoader(dataset,batch_size=1,shuffle=True)

    for idx,(HR,LR,LR_edge) in enumerate(dataloader):
        if idx <= 3:

            print(type(HR))
            torchvision.utils.save_image(HR,"test_{}_HR.jpg".format(idx))
            torchvision.utils.save_image(LR,"test_{}_LR.jpg".format(idx))
            torchvision.utils.save_image(LR_edge,"LR_edge_{}.jpg".format(idx))
        else:
            print("test complete")
            break


