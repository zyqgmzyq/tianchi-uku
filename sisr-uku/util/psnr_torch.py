import numpy as np 
import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# def psnr_torch(im1, im2, scale):  # Mine
#     '''only calc Y channel'''
#     im1 = im1.squeeze().float()
#     im2 = im2.squeeze().float()

#     if len(im1.shape) == 3:
#         im1 = rgb2ycbcr(im1)
#     if len(im2.shape) == 3:
#         im2 = rgb2ycbcr(im2)
#     diff = im1 - im2
#     diff = diff[scale:-scale, scale:-scale]
#     rmse = diff.pow(2).mean().sqrt().item()
#     return 20.*np.log10(255.0/rmse)
#     # return 20*torch.log10(255.0/rmse)

# def rgb2ycbcr(im):  # Mine
#     '''only return Y channel - PyTorch'''
#     # im = im.squeeze()
#     # convert = np.array([[[65.481/255, 128.553/255, 24.966/255]]]).astype(np.float32)  # Matlab
#     convert = np.array([[[65.738/256, 129.057/256, 25.064/256]]]).astype(np.float32)  # RCAN
#     convert = torch.from_numpy(convert).to(device)
#     im = im.float().to(device) * convert
#     im = torch.round(torch.sum(im, (2), keepdim = True, dtype=torch.float32)) + 16
#     if len(im.shape) == 3:
#         im = im[:,:,0]
#     return im


def psnr_torch(im1, im2, scale):  # 我直接传一维的y进来
    '''only calc Y channel'''
    im1 = im1.squeeze().float()
    im2 = im2.squeeze().float()

    if len(im1.shape) == 3:
        im1 = rgb2ycbcr(im1)
    if len(im2.shape) == 3:
        im2 = rgb2ycbcr(im2)
    diff = im1 - im2
    diff = diff[scale:-scale, scale:-scale]
    # rmse = diff.pow(2).mean().sqrt().item()
    # return 20.*np.log10(256.0/rmse)
    rmse = diff.div_(256).pow(2).mean().item()
    return -10.*np.log10(rmse)
    # return 20*torch.log10(255.0/rmse)

def rgb2ycbcr(im):
    '''only return Y channel - PyTorch'''
    # im = im.squeeze()
    # convert = np.array([[[65.481/255, 128.553/255, 24.966/255]]]).astype(np.float32)  # Matlab
    convert = np.array([[[65.738/256, 129.057/256, 25.064/256]]]).astype(np.float32)  # RCAN
    convert = torch.from_numpy(convert).to(device)
    im = im.float().to(device) * convert
    im = torch.round(torch.sum(im, (2), keepdim = True, dtype=torch.float32)) + 16
    if len(im.shape) == 3:
        im = im[:,:,0]
    return im