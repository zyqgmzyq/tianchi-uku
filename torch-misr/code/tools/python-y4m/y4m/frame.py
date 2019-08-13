
__all__ = ['Frame']


import numpy as np
import pickle

class Frame():
    def __init__(self,yuv,shape, frame_headers, frame_id):
        self.frame_headers=frame_headers
        self.frame_id=frame_id
        self.shape=shape
        count=0
        dims=[shape[1],shape[0]]
        d00=dims[0]//2  
        d01=dims[1]//2  
        Yt=np.zeros((dims[0],dims[1]),np.uint8,'C')  
        Ut=np.zeros((d00,d01),np.uint8,'C')  
        Vt=np.zeros((d00,d01),np.uint8,'C')   
        for m in range(dims[0]):  
            for n in range(dims[1]):  
                Yt[m,n]=yuv[count]
                count+=1  
        for m in range(d00):  
            for n in range(d01):  
                Ut[m,n]=yuv[count]
                count+=1    
        for m in range(d00):  
            for n in range(d01):  
                Vt[m,n]=yuv[count]
                count+=1  
        self.yuv=[Yt,Ut,Vt]
    
    def save(self,dstPath):
        fw = open(dstPath,'wb')
        pickle.dump(self.yuv, fw,pickle.HIGHEST_PROTOCOL)
        fw.close()
 