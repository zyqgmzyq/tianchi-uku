import mxnet as mx
import numpy as np
import math
class PSNR(mx.metric.EvalMetric):
    def __init__(self):
        super(PSNR, self).__init__('PSNR')

    def update(self,labels, preds):
        shave=1
        sum_psnr=0
        
        l_y=labels[0].asnumpy()
        y_y=preds[0].asnumpy()
        l_uv=labels[1].asnumpy()
        y_uv=preds[1].asnumpy()
        diff_y = (l_y - y_y) / 255.0
        diff_uv = (l_uv - y_uv) / 255.0
        for b in range(l_y.shape[0]):
           valid_y = diff_y[b,:, shave:-shave, shave:-shave]**2
           valid_uv = diff_uv[b,:, shave:-shave, shave:-shave]**2
           mse = valid_y.mean()+valid_uv.mean()*0.5
           mse /= 1.5
           sum_psnr+= -10 * math.log10(mse)

        self.sum_metric += sum_psnr
        self.num_inst += l_y.shape[0]

class PSNR_Y(mx.metric.EvalMetric):
    def __init__(self):
        super(PSNR_Y, self).__init__('PSNR_Y')

    def update(self,labels, preds):
        shave=1
        sum_psnr=0
        l_y=labels[0].asnumpy()
        y_y=preds[0].asnumpy()

        diff_y = (l_y - y_y) / 255.0
        
        for b in range(l_y.shape[0]):
           valid_y = diff_y[b,:, shave:-shave, shave:-shave]**2
           
           mse = valid_y.mean()
           
           sum_psnr+= -10 * math.log10(mse)

        self.sum_metric += sum_psnr
        self.num_inst += l_y.shape[0]

class YUV420MAE(mx.metric.EvalMetric):
    def __init__(self):
        super(YUV420MAE, self).__init__('YUV420MAE')

    def update(self,labels, preds):
        shave=1
        sum_mse=0
        
        l=labels[0].asnumpy()
        y=preds[0].asnumpy()
        diff = np.abs(l - y).mean()
        sum_mse+=diff
        
        l=labels[1].asnumpy()
        y=preds[1].asnumpy()
        diff = np.abs(l - y).mean()
        sum_mse+=(diff*0.5)
        
        sum_mse/=1.5

        self.sum_metric += sum_mse
        self.num_inst += 1