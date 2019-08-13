import y4m
import os
import sys
import cv2
import numpy as np
import random as rd
import copy
#from six.moves.queue import Queue
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


def decodey4m(file_queue, dstPath, threadID):
    while not file_queue.empty():
        f = file_queue.get(1, 5)
        print("thread "+str(threadID)+' decode '+f)
        parser = y4m.Reader(sourcePath=f, dstPath=dstPath, verbose=False)
        fp = open(f, 'rb')
        try:
            parser.decode(fp.read())
        except:
            print('error '+f)
        fp.close()


# y4m to pickle
def decodeDataset(root, dstPath, numThreads=8):
    files = eachFile(root)
    file_queue = Queue()
    pool = []
    for f in files:
        if '.y4m' in f:
            file_queue.put(f)
    for i in range(numThreads):
        p = Process(target=decodey4m, args=(file_queue, dstPath, i,))
        p.start()
        pool.append(p)
    for i in range(numThreads):
        pool[i].join()
    for i in range(numThreads):
        pool[i].terminate()

    
    
    
    


