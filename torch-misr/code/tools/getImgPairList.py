import os
import sys
import cv2
import numpy as np
import random as rd
import copy


# -*- coding: UTF-8 -*-


def eachFile(filepath):
    pathDir = os.listdir(filepath)
    subfile = []
    for allDir in pathDir:
        child = os.path.join('%s/%s' % (filepath, allDir))
        subfile.append(child)
    return subfile


def writeList(pathlist, path):
    f = open(path, 'w')
    for p in pathlist:
        f.write(p + ' \n')
    f.close()


def getDIV2KImgPairList(path0, path1, dstPath):
    imgset = []
    paths = eachFile(path0)
    for p in paths:
        if not '.png' in p:
            continue

        name = p.split('/')[-1]

        if os.path.exists(path1 + '/' + name.replace('.png', 'x2.png')):
            imgset.append(p + ' ' + path1 + '/' + name.replace('.png', 'x2.png'))

    num = int(len(imgset) * 0.9)
    writeList(imgset[0:num], dstPath + 'trainPair.txt')
    writeList(imgset[num:], dstPath + 'testPair.txt')


def getTCSRImgPairList(path, dstPath):
    type_01 = []
    type_23 = []

    f = open("../type_label.txt", 'r')
    for line in f.readlines():
        if int(line.split(' ')[1]) == 1 or int(line.split(' ')[1]) == 0:
            type_01.append(line.split(' ')[0])
        else:
            type_23.append(line.split(' ')[0])

    trainset = []
    testset = []
    trainID = [250, 799]
    testID = [800, 849]
    paths = eachFile(path)
    # print(len(paths))
    for p in paths:
        if '.pickle' not in p:
            continue
        if 'h_GT' not in p:
            continue
        name = p.split('/')[-1]
        video_ind = int(name.split('_')[1])

        if os.path.exists(p.replace('h_GT', 'l')):
            if trainID[0] <= video_ind <= trainID[1] and type_23.__contains__(str(video_ind)):
                trainset.append(p + ' ' + p.replace('h_GT', 'l'))
            if testID[0] <= video_ind <= testID[1] and type_23.__contains__(str(video_ind)):
                testset.append(p + ' ' + p.replace('h_GT', 'l'))

    writeList(trainset, dstPath + 'train_type23.txt')
    writeList(testset, dstPath + 'test_type23.txt')








