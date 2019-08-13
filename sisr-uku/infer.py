import time
import numpy as np
import torch
import os
import scipy.misc as misc
from config import args
from util import mkdirs
from concurrent.futures import ThreadPoolExecutor, wait
import model
import glob
from util import logging

def frames2y4m(frames, save_path):
    with open(save_path, 'wb') as fw:
        if args.debug:
            header = b"YUV4MPEG2 W192 H256 F24:1 Ip A0:0 C420mpeg2 XYSCSS=420MPEG2\n"
        else:
            header = b"YUV4MPEG2 W1920 H1080 F24:1 Ip A0:0 C420mpeg2 XYSCSS=420MPEG2\n"
        fw.write(header)
        for frame in frames:
            y = frame[:, :, 0]
            cb = frame[:, :, 1].astype(np.float32)
            cr = frame[:, :, 2].astype(np.float32)
            cb_s = np.zeros([i // 2 for i in cb.shape], dtype=np.float32)
            cr_s = np.zeros([i // 2 for i in cr.shape], dtype=np.float32)

            cb_s += cb[0::2, 0::2]
            cb_s += cb[1::2, 0::2]
            cb_s += cb[0::2, 1::2]
            cb_s += cb[1::2, 1::2]
            cb = (cb_s / 4).round().astype(np.uint8)

            cr_s += cr[0::2, 0::2]
            cr_s += cr[1::2, 0::2]
            cr_s += cr[0::2, 1::2]
            cr_s += cr[1::2, 1::2]
            cr = (cr_s / 4).round().astype(np.uint8)

            fw.write(b'FRAME\n')
            for i in range(y.shape[0]):
                for j in range(y.shape[1]):
                    fw.write(y[i][j])

            for i in range(cb.shape[0]):  # half size
                for j in range(cb.shape[1]):
                    fw.write(cb[i][j])

            for i in range(cr.shape[0]):  # half size
                for j in range(cr.shape[1]):
                    fw.write(cr[i][j])

    return save_path


def read_as_tensor(img_path, dformate='numpy'):

    img = np.load(img_path)  if dformate=='numpy' else misc.imread(img_path)  # ycbcr or rgb
    if args.debug:
        img = img[:64, :48]
    img = img.transpose((2, 0, 1))
    img = img.astype(np.float32) / 255. - 0.5
    img = torch.from_numpy(img).float()
    img = img.unsqueeze(0)

    return img


def gen_y4m(net, vid, device):
    lr_fns = sorted(glob.glob(os.path.join(args.infer_path, '*{:05d}*.npy'.format(vid))))
    lr_fns = lr_fns[::25] if (vid >= mid and len(lr_fns)>4) else lr_fns
    message = 'generating vid {}, total frame {}'.format(vid, len(lr_fns))
    print(message)
    logging(logger, message)

    frames = []
    for lr_fn in lr_fns:
        lr_tensor = read_as_tensor(lr_fn).to(device)

        sr_tensor = net(lr_tensor).squeeze().float().permute(1, 2, 0)
        sr_tensor = torch.clamp((sr_tensor + mean_torch) * 255, 0, 255).round()  # 这个存的位数和numpy似乎有点不一样
        sr_np = sr_tensor.detach().cpu().numpy().astype(np.uint8)  # ycbcr for np.load, rgb for misc.imread

        frames.append(sr_np)

    today = time.strftime('%m%d',time.localtime(time.time()))
    spcified_path = os.path.join(args.output_path, '{}_{}'.format(today, args.model.lower()))

    if not os.path.exists(spcified_path): os.makedirs(spcified_path)
    if vid<mid:
        whole_save_path = os.path.join(spcified_path, 'Youku_{:05d}_h_Res.y4m'.format(vid))
    else:
        whole_save_path = os.path.join(spcified_path, 'Youku_{:05d}_h_Sub25_Res.y4m'.format(vid))
    frames2y4m(frames, whole_save_path)

    return vid


if __name__ == '__main__':

    # logger = 'log/{}_infer.txt'.format(args.model.lower())
    logger = 's4_rcan_ps32_bs16_lossYUV.txt'
    now = time.strftime('%Y.%m.%d %H:%M:%S\n', time.localtime(time.time()))
    logging(logger, args.message + now)
    logging(logger, 'using gpu: {}\n'.format(torch.cuda.is_available()))

    if args.output_path: mkdirs(args.output_path)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    mean_numpy = np.array(args.mean).reshape((1, 1, -1))
    mean_torch = torch.from_numpy(mean_numpy).float().to(device)

    #### vid setting ####
    begin = args.begin_id
    mid = args.mid_id
    end = args.end
    assert mid - begin == 5, 'full video must be 5'
    assert end - begin == 50, 'end id must be 50 more than actual given id'
    vids = [i for i in range(begin, end)]
    vids = [i for i in range(begin, end)]
    #### vid setting ####

    print('begin inferring, total vid: {}, start from {:05d} to {:05d}'.format(len(vids), vids[0], vids[-1]))
    with torch.no_grad():

        net = model.Model()
        net.load(args.ckpt)

        # num_workers = 1  # 显存超了就报错，要是大点的可以开大点
        # thread_pool = ThreadPoolExecutor(max_workers=num_workers)
        # threads = []

        for vid in vids:
            gen_y4m(net, vid, device)
        #     threads.append(thread_pool.submit(gen_y4m, net, vid, device))
        #     if len(threads) >= num_workers:
        #         done, pending = wait(threads, timeout=None, return_when='FIRST_COMPLETED')
        #         print(done)
        #         if done in threads:
        #             threads.remove(done)
        # wait(threads)
        # threads.clear()
