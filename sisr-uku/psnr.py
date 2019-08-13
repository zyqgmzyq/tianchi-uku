import cv2
import numpy as np
import pickle
import math
import os


def loadimg(path):
    lr_yuv_fr = open(path, 'rb')
    lr_yuv = pickle.load(lr_yuv_fr)
    lr_yuv_fr.close()
    return lr_yuv


def psnr_yuv(img1_y, img2_y, img1_uv, img2_uv):
    sum_psnr = 0
    diff_y = (img1_y - img2_y) / 255.0
    diff_uv = (img1_uv - img2_uv) / 255.0
    valid_y = diff_y ** 2
    valid_uv = diff_uv ** 2
    # mse = valid_y.mean() + valid_uv.mean() * 0.5
    # mse /= 1.5
    mse = valid_uv.mean()
    sum_psnr += -10 * math.log10(mse)
    return sum_psnr


def psnr_y(img1, img2):
    sum_psnr = 0
    diff_y = (img1 - img2) / 255.0
    valid_y = diff_y ** 2
    mse = valid_y.mean()
    sum_psnr += -10 * math.log10(mse)
    return sum_psnr


def cal_psnr(imgs_path):
    img_list = [os.path.join(imgs_path, i) for i in os.listdir(imgs_path)]
    sum_psnr_y = 0
    sum_psnr_yuv = 0
    s = imgs_path.split('/')[-2]
    for i in range(len(img_list)):
        img1 = loadimg(img_list[i])
        img1_y = img1[0]
        img1_uv = img1[1]
        img2 = loadimg(img_list[i].replace(s, "val"))
        img2_y = img2[0].squeeze()
        img2_uv = img2[1].squeeze()
        for j in range(len(img1_y)):
            x = np.mean(img1_y[j])
            print(j, x)
        exit()
        sum_psnr_y += psnr_y(img1_y, img2_y)
        sum_psnr_yuv += psnr_yuv(img1_y, img2_y, img1_uv, img2_uv)
    return sum_psnr_y, sum_psnr_yuv


def cal_psnr2(imgs_path):
    img_list = [os.path.join(imgs_path, i) for i in os.listdir(imgs_path)]
    sum_psnr_y = 0
    sum_psnr_yuv = 0
    s = imgs_path.split('/')[-2]
    for i in range(len(img_list)):
        img1 = loadimg(img_list[i])
        img1_y = img1[0]
        img1_uv = img1[1]
        # print(img1_uv.shape)
        # for j in range(len(img1_uv)):
        #     # print(img1_uv[j])
        #     x = np.mean(img1_uv[j])
        #     print(j, x)
        # exit()
        img2 = loadimg(img_list[i].replace(s, "val"))
        # print(img2.shape)
        # exit()
        img2_y = img2[0].squeeze()
        img2_uv = img2[1].squeeze()
        img2_y[0:130][:] = 16
        img2_y[949:1079][:] = 16
        img2_uv[0:60][:] = 128
        img2_uv[480:539][:] = 128

        sum_psnr_y += psnr_y(img1_y, img2_y)
        sum_psnr_yuv += psnr_yuv(img1_y, img2_y, img1_uv, img2_uv)
    return sum_psnr_y, sum_psnr_yuv


# def get_top_bottom(imgpath):
#     img = cv2.imread(imgpath)

# imgs_path = "../y4m/gt1/"
# psnr1_y, psnr1_yuv = cal_psnr(imgs_path)
# psnr2_y, psnr2_yuv = cal_psnr2(imgs_path)
# print("y:", psnr1_y/100, "  ", psnr2_y/100)
# print("uv:", psnr1_yuv/100, "  ", psnr2_yuv/100)
# print("--------------------------------")
#
# imgs_path = "../y4m/gt2/"
# psnr1_y, psnr1_yuv = cal_psnr(imgs_path)
# psnr2_y, psnr2_yuv = cal_psnr2(imgs_path)
# print("y:", psnr1_y/100, "  ", psnr2_y/100)
# print("uv:", psnr1_yuv/100, "  ", psnr2_yuv/100)
# print("--------------------------------")



# imgs_path = "../y4m/gt3/"
# psnr1_y, psnr1_yuv = cal_psnr(imgs_path)
# psnr2_y, psnr2_yuv = cal_psnr2(imgs_path)
# print("y:", psnr1_y/100, "  ", psnr2_y/100)
# print("uv:", psnr1_yuv/100, "  ", psnr2_yuv/100)
# print("--------------------------------")


# img = cv2.imread("./019.bmp")
lr_yuv_fr = open('Youku_00250_h_GT_0001.pickle', 'rb')
img = pickle.load(lr_yuv_fr)[1]
img = np.array(img)
img = np.transpose(img)
for i in range(len(img)):
    x = np.mean(img[i])
    print(i, x)
    print(img[i])
    print(len(img[i]), "------------------------------------")
lr_yuv_fr.close()


# print(len(img))
# print(len(img[0]))
# print(len(img[0][0]))
# print(img.shape)
