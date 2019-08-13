import os
import sys
import cv2
import numpy as np
import random as rd
import copy
import os.path as osp
import subprocess
from y4m2pickle import *
from getImgPairList import *
# -*- coding: UTF-8 -*-


def eachFile(filepath):
    pathDir = os.listdir(filepath)
    subfile=[]
    for allDir in pathDir:
        child = os.path.join('%s/%s' % (filepath, allDir))
        subfile.append(child)
    return subfile


def unzip_tcsr(root, dstPath):
    files = eachFile(root)
    for f in files:
        if '.zip' not in f:
            continue
        print('unzip '+f)
        subprocess.check_call(['unzip', '-d', dstPath, f])


root = "../../data/round1_train_label"
dstPath = "../../data/round1_train_label"

#unzip_tcsr(root, dstPath)
#unzip_tcsr(root.replace('label', 'input'), dstPath.replace('label', 'input'))
#unzip_tcsr(root.replace('train', 'val'), dstPath.replace('train', 'val'))
#unzip_tcsr(root.replace('label', 'input').replace('train', 'val'),
#           dstPath.replace('label', 'input').replace('train', 'val'))
#unzip_tcsr("../../data/round2_train_input", "../../data/round2_train_input")
#unzip_tcsr("../../data/round2_train_label", "../../data/round2_train_label")
#unzip_tcsr("../../data/round1_test_input", "../../data/round1_test_input")
#unzip_tcsr("../../data/round2_test_input", "../../data/round2_test_input")

if not os.path.exists("../../data/train/yuvPickle2/"):
    os.makedirs("../../data/train/yuvPickle2/")
if not os.path.exists("../../data/test2/yuvPickle/"):
    os.makedirs("../../data/test2/yuvPickle/")


root = "../../data/round1_train_label"
dstPath = "../../data/train/yuvPickle2/"

# decodeDataset(root, dstPath)
# decodeDataset(root.replace('label', 'input'), dstPath)
# decodeDataset(root.replace('train', 'val'), dstPath)
# decodeDataset(root.replace('label', 'input').replace('train', 'val'), dstPath)
# decodeDataset("../../data/round1_test_input", "../../data/test/yuvPickle/")
# decodeDataset("../../data/round2_train_input", dstPath)
# decodeDataset("../../data/round2_train_label", dstPath)
# decodeDataset("../../data/round2_test_input", "../../data/test2/yuvPickle/")

path = "../../data/train/yuvPickle2"
getTCSRImgPairList(path, "../../data/train/")




