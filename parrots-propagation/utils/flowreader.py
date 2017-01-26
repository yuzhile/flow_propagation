from pyparrots.dnn import reader
from flowlib import read_flow
import random
import numpy as np
from PIL import Image
import cv2
import os

class FlowReader:
    '''
    Load (img1,img2) pairs and corresponding flow gt fro FlyChair
    one-at-a-time while reshaping the net to preserve dimensions
    '''


    support_keys = ['img1_list','img2_list','gt_list','data_dir','mean','randomize','crop_size','shrink_factor','train','gt_scale']
    
    def config(self,cfg):
        '''
        Setup data parameters:
        '''
    
        self.img1_list = cfg['img1_list']
        self.img2_list = cfg['img2_list']
        self.gt_list = cfg['gt_list']
        self.data_dir = cfg['data_dir']
        self.mean = np.array(map(np.float32,cfg['mean']['value']))
        self.mean_input_scale = np.float32(cfg['mean']['input_scale'])
        self.gt_scale= np.float32(cfg['gt_scale'])
        self.random = cfg.get('randomize',False)
        self.seed = cfg.get('seed',None)
        self.img1_indices = open(self.img1_list,'r').read().splitlines()
        self.img2_indices = open(self.img2_list,'r').read().splitlines()
        self.gt_indices = open(self.gt_list,'r').read().splitlines()
        self.num_per_epoch = len(self.img1_indices)
        self.idx = 0
        self.iter = self._read_iter()
    
        if self.random:
            random.seed(self.seed)
    
    def read(self):
        '''
        This read function should read a sample every call.
        We use iterator to implement it.
        '''
        return self.iter.next()
    def _read_iter(self):
        '''
        Implement reading a sample every call.
        returns
        - img1
        - img2
        - gt
        process:
            read img1
            img1 bgr->rgb
            read img2
            img2 bgr->rgb
            read gt flow
            img1 sub mean
            img2 sub mean
            hwc->whc
        '''
        while True:
            order = range(self.num_per_epoch)
            if self.random:
                random.shuffle(order)
            for i in order:
                
                img1_file = os.path.join(self.data_dir,self.img1_indices[i])
                img1 = self._load_image(img1_file)
                img1 = self._mean(img1)
                
                img2_file = os.path.join(self.data_dir,self.img2_indices[i])
                img2 = self._load_image(img2_file)
                img2 = self._mean(img2)
                
                flow_gt_file = os.path.join(self.data_dir,self.gt_indices[i])
                flow_gt = read_flow(flow_gt_file)
                flow_gt = flow_gt * self.gt_scale
                blob6 = self._downsample(flow_gt,(8,6))
                blob5 = self._downsample(flow_gt,(16,12))
                blob4 = self._downsample(flow_gt,(32,24))
                blob3 = self._downsample(flow_gt,(64,48))
                blob2 = self._downsample(flow_gt,(128,96))
                yield [img1.transpose((1,0,2)),img2.transpose((1,0,2)),blob6.transpose((1,0,2)),blob5.transpose((1,0,2)),blob4.transpose((1,0,2)),blob3.transpose((1,0,2)),blob2.transpose((1,0,2))]

    def _downsample(self,flow_gt,dsize):
        '''
        downsample the flow gt and feed it to loss
        inputs:
        - flow_gt, with shape(h,w,2)
        - dsize, destnition shape(w,h)
        '''
        return cv2.resize(flow_gt,dsize)

    def _load_image(self,image_file):
        '''
        Load image and convert BGR->RGB
        '''

        img = cv2.imread(image_file)
        img = img[:,:,::-1]
        return img
        
    def _mean(self,image):
        '''
        Mean transform.
        - scale input
        - sub mean
        '''
        image = image*self.mean_input_scale - self.mean
        return image

reader.register_pyreader(FlowReader, 'flow')
