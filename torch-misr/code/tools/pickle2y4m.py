import os
import sys
import cv2
import numpy as np
import random as rd
import copy
import pickle
import subprocess
from threading import Thread
from multiprocessing import Pool
from multiprocessing import Process, Queue


# -*- coding: UTF-8 -*-


def eachFile(filepath):
    pathDir = os.listdir(filepath)
    subfile = []
    for allDir in pathDir:
        child = os.path.join('%s/%s' % (filepath, allDir))
        subfile.append(child)
    return subfile


# 从pickle文件读出yuv文件
def load_yuv(path):
    fr = open(path, 'rb')
    yuv = pickle.load(fr)
    fr.close()
    return yuv


# yuv to y4m
def yuv2y4m(dstPath, frames):
    print(dstPath)
    fw = open(dstPath, 'wb')
    header = b"YUV4MPEG2 W1920 H1080 F24:1 Ip A0:0 C420mpeg2 XYSCSS=420MPEG2\n"
    fw.write(header)

    for f in frames:
        y = f[0][0].astype(np.uint8)
        u = f[1][0].astype(np.uint8)
        v = f[1][1].astype(np.uint8)
        fw.write(b'FRAME\n')
        for m in range(y.shape[0]):
            for n in range(y.shape[1]):
                num = y[m, n]
                fw.write(num)
        for m in range(u.shape[0]):
            for n in range(u.shape[1]):
                num = u[m, n]
                fw.write(num)
        for m in range(v.shape[0]):
            for n in range(v.shape[1]):
                num = v[m, n]
                fw.write(num)
    fw.close()


def gatherVideoFrame(root, video_id):
    frames = []
    framePath = []
    frameID = []
    files = eachFile(root)
    for f in files:
        if '.pickle' not in f:
            continue
        vid = f.split('_sr_')[0]
        vid = int(vid[-5:])
        fid = f.split('_sr_')[1]
        fid = int(fid[:4])
        if vid == video_id:
            framePath.append(f)
            frameID.append(fid)
    for i in range(len(frameID)):
        ind = frameID.index(i)
        yuv = load_yuv(framePath[ind])
        frames.append(yuv)
    return frames


def zipVideo(root, dstPath, vid_queue, tid):
    while not vid_queue.empty():
        vid = vid_queue.get(1, 5)
        frames = gatherVideoFrame(root=root, video_id=vid)
        vid = str(vid)
        vid = '0' * (5 - len(vid)) + vid
        savePath = dstPath + 'Youku_' + vid + '_h_Res.y4m'
        yuv2y4m(savePath, frames)
        print("thread " + str(tid) + ' saved ' + savePath)


def generateTestsetVideo(root, dstPath, vid=[200, 249], numThreads=16):
    vid_queue = Queue()
    pool = []
    for i in range(vid[0], vid[1] + 1):
        vid_queue.put(i)
    for i in range(numThreads):
        print('start process ' + str(i))
        p = Process(target=zipVideo, args=(root, dstPath, vid_queue, i,))
        p.start()
        pool.append(p)
    for i in range(numThreads):
        pool[i].join()
    for i in range(numThreads):
        pool[i].terminate()


def Sub25(src, dst):
    print(src, dst)
    subprocess.check_call(['ffmpeg', '-i', src, '-vf', "select='not(mod(n\,25))'",
                           '-vsync', '0', '-y', dst])
    subprocess.check_call(['rm', src])


def Subvideo(root, vid=[205, 249]):
    files = eachFile(root)
    for f in files:
        if '.y4m' not in f:
            continue
        if 'Sub' in f:
            continue
        videoid = f.split('_h_Res')[0]
        videoid = int(videoid[-5:])
        if vid[0] <= videoid <= vid[1]:
            # print(f)
            # print(f.replace('_Res','_Sub25_Res'))
            Sub25(f, f.replace('_Res', '_Sub25_Res'))


# root = "../../data/result_pickle/"
root = "../../submit/yuvPickle/"
dstPath = "../../submit/result/"
vid = [850, 899]
generateTestsetVideo(root, dstPath, vid)
Subvideo("../../submit/result/", vid=[855, 899])


