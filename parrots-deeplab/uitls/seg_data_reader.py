#from pyparrots.dnn import reader

import random
import numpy as np
from PIL import Image
import cv2

class SegReader:
    '''
    Load (input image, label image) pairs from PASCAL VOC
    one-at-a-time while reshaping the net to preserve dimensions

    Use this to feed data to a fully convolutional network.
    '''
    
    support_keys = ['split','voc_dir','mean','randomize','crop_size','ignore_label','shrink_factor','num_class','train']

    def config(self,cfg):
        '''
        Setup data parameters:
        - voc_dir: path to PASCAL VOC year dir
        - split: train / val / test
        - mean: tuple of mean values to subtrack
        - randomize: load in random order (default: True)
        
        '''

        self.split = cfg['split']
        self.voc_dir = cfg['voc_dir']
        self.mean = np.array(map(np.float32,cfg['mean']))
        self.random = cfg.get('randomize',False)
        self.seed = cfg.get('seed',None)
        self.crop_size = int(cfg['crop_size'])
        self.ignore_label = int(cfg['ignore_label'])
        self.shrink_factor = int(cfg['shrink_factor'])
        self.num_class = int(cfg.get('num_class',20))
        self.train = cfg.get('train',False)
        self.indices = open(self.split,'r').read().splitlines()
        self.idx = 0
        self.iter = self._read_iter()
        #if 'train' not in self.split:
        # use train flag to indicate the forward phase
        if not self.train:
            self.random = False

        if self.random:
            random.seed(self.seed)

    def read(self):
        '''
        This read function should read a sample every call.
        We use iterator to implement it.
        '''
        return self.iter.next()

    def _read_iter(self):
        while True:
            size = len(self.indices)
            order = range(size)
            if self.random:
                random.shuffle(order)
            for i in order:
                #idxs = self.indices[i].strip()
                idxs = [x.strip() for x in self.indices[i].split()]
                #print 'image name',idxs[0]
                #debug
                #print 'idxs',idxs
                #image,src_img = self.load_image(idxs[0])
                # load the raw image 
                #print 'begin {}'.format(i)
                image = self._load_image(idxs[0])
                h, w, c = image.shape

                label = self._load_label(idxs[1])
                crop_img, crop_label = self._deeplab_crop(image,label) 
                #print 'image and raw shape',image.shape,label.shape
                #resize_size , top_left = self._crop(h,w)
                #print 'resize and top left',resize_size, top_left
                #resize image
                #resize_image = cv2.resize(image,resize_size)
                #resize_label = cv2.resize(label,resize_size,interpolation=cv2.INTER_NEAREST)
                # crop image
                #print 'image after shape',resize_image.shape,resize_label.shape

                
                #crop_img = resize_image[top_left[0]:top_left[0]+self.crop_size,
                #        top_left[1]:top_left[1]+self.crop_size,:]

                #crop_label = resize_label[top_left[0]:top_left[0]+self.crop_size,
                #        top_left[1]:top_left[1]+self.crop_size]
                # get label mask
                label_shrink,label_mask = self._transform_label(crop_label)
                # substract mean
                crop_img -= self.mean
                #print 'after transfrom shape',crop_img.shape,label_shrink.shape,label_mask.shape
                #if (label > self.num_class).sum() > 0:
                #    rai:se Exception('error label',label[label > self.num_class])
                # The data should be transposed for the data in caffe is chw,but parrots is whc
                # transpose hwc -> whc
                yield [crop_img.transpose((1,0,2)),label_shrink.transpose((1,0,2)), label_mask.transpose((1,0,2))]

                #yield [image,label,src_img]
    def _deeplab_crop(self,seg_image,seg_label):
        '''
        apply the deeplab crop methods: fisrt padding the seg_image using mean values and padding 
        the seg_label using ignore_label if necerary.Then random crop when trainning and middle crop when val
        inputs
        - seg_image: from cv2, with order(B,G,R)
        - seg_label: with the same shape with seg_image.
        returns:
        seg_image: with shape(crop_size,crop_size,c)
        seg_label: with shape(crop_size,crop_size)
        '''
        assert len(seg_image.shape) == 3,'the image should be BGR channels'
        assert len(seg_label.shape) == 2,'the image seg label should be 2 dim'
        data_height,data_width,data_c =  seg_image.shape
        label_height,label_width = seg_label.shape
        assert data_height == label_height,'image and label should have the same height'
        assert data_width == label_width,'image and label should have the same width'
        
        crop_size = self.crop_size
        pad_height = max(crop_size - data_height, 0)
        pad_width = max(crop_size - data_width, 0)

        # when crop is needed
        # pad 
        if pad_height > 0 or pad_width > 0:
            #cv2 copymakevorder need float64 as value parameter
            mean = np.array(self.mean,dtype=np.float64)
            seg_image = cv2.copyMakeBorder(seg_image,0,pad_height,0,pad_width,cv2.BORDER_CONSTANT,value=mean)
            seg_label = cv2.copyMakeBorder(seg_label,0,pad_height,0,pad_width,cv2.BORDER_CONSTANT,value=self.ignore_label)

            # update height/width
            data_height = seg_image.shape[0]
            data_width = seg_image.shape[1]
            label_height = seg_label.shape[0]
            label_width = seg_label.shape[1]
        #crop
        if  self.train:
            # random crop
            h_off = np.random.randint(data_height - crop_size + 1)
            w_off = np.random.randint(data_width - crop_size + 1)
        else:
            h_off = (data_height - crop_size) / 2
            w_off = (data_width - crop_size) / 2

        # roi image
        return seg_image[h_off:h_off+crop_size, w_off:w_off+crop_size,:],seg_label[h_off:h_off+crop_size,w_off:w_off+crop_size]


    def _transform_label(self,raw_label):
        '''
        shrink the label and get the label mask
        inputs 
        - raw_label
        returns:
        - label_shrink
        - label_mask
        '''
        out_size = (self.crop_size - 1) / self.shrink_factor + 1
        #print 'transform label, raw label shape',raw_label.shape
        label_shrink = cv2.resize(raw_label,(out_size,out_size),
                interpolation=cv2.INTER_NEAREST)

        label_shrink = label_shrink[...,np.newaxis]
        # generate the label mask
        label_mask = np.ones_like(label_shrink)
        label_mask[label_shrink == self.ignore_label] = 0
        # we need to set the ignore_label as 0 to remove the error label 
        label_shrink[label_shrink == self.ignore_label] = 0
        return label_shrink, label_mask

    def _crop(self,h,w):
        '''
        random crop the image by resize the shortest edge to self.crop_size
        inputs:
        - h: image height
        - w: image width
        returns:
        - resize_size:resize_width,resize_height, this is the paramenter of cv2
        - top_left: height start,width start
        '''
        #r = float(h) / w if h > w else float(w) / h
        #we crop the shortest edge to self.crop_size, and mantian the same
        # ratio of w,h
        shortest = w if h > w else h
        # If shortest edge less than crop size, then resize the shortest edge to crop size to crop size and maitains the w/h
        if shortest <= self.crop_size:
            crop_len = self.crop_size* h /w if h > w else self.crop_size*w / h
        # get the offet
            offset = np.random.randint(crop_len - self.crop_size + 1)
            resize_size,top_left =   [(self.crop_size,crop_len),(offset,0)] if h > w else [(crop_len, self.crop_size),(0,offset)]
        # if short edge more than crop size, then crop them directly
        else:
            resize_size = (w,h)
            top_left = (np.random.randint(h-self.crop_size+1),np.random.randint(w-self.crop_size+1))
        #if h > w:
        #    crop_size = (h*self.crop_size/w,self.crop_size)
        #    top_left = (
        #crop_size = (h*self.crop_size / w,self.crop_size) if h > w or (self.crop_size,w*self.crop_size / h)
        return resize_size,top_left
    def _load_image(self,idx):
        '''
        Load input image and preprocess for parrots
        - cast to float
        '''
        
        #im = Image.open('{}{}'.format(self.voc_dir, idx))
        im = cv2.imread('{}{}'.format(self.voc_dir,idx))
        in_ = np.array(im,dtype=np.float32)
        # RGB->BGR
        #in_ = im[:,:,::-1]
        # sub mean
        #in_ -= self.mean

        
        return in_
           
    def load_image(self,idx):
        '''
        Load input image and preprocess for parrots
        - cast to float
        - witch channels RGB->BGR
        - substract mean
        '''
        
        im = Image.open('{}{}'.format(self.voc_dir, idx))
        im = im.resize((self.crop_size,self.crop_size),Image.ANTIALIAS)
        #im = cv2.imread('{}{}'.format(self.voc_dir,idx))
        #im = cv2.resize(im,(self.crop_size,self.crop_size))
        #import matplotlib.pyplot as plt
        #plt.imshow(im)
        #plt.show()
        in_ = np.array(im,dtype=np.float32)
        in_ = in_[:,:,::-1]
        #debug 
        #print 'mean',self.mean
        # mean is b g r
        in_ -= self.mean

        #return in_,np.array(im,dtype=np.float32)
        #in_ = in_.transpose((2,1,0))
        #in_ = in_[np.newaxis,...]
        #return in_,np.array(im,dtype=np.float32)
        return in_
    def _decode_label(self,label_file):
        '''
        Read label image as raw data.

        '''
        with open(label_file,'rb') as infile:
            buf = infile.read()
        raw = np.fromstring(buf,dtype='uint8')
        img = cv2.imdecode(raw,cv2.IMREAD_UNCHANGED)
        return img

    def _load_label(self,idx):
        '''
        Load label image as height x width interger array of label indices.
        return:
        - label,with shape(h,w)
        '''
        img_file = '{}{}'.format(self.voc_dir,idx)
        label = self._decode_label(img_file)
        assert len(label.shape) == 2,'the image seg labe should be 2 dim'
        return label


    def load_label(self,idx):
        '''
        Load label image as height x width interger array of label indices.
        return:
        - label
        - label_mask: ignore some label, with the same shape as label
        '''
        #im = Image.open('{}{}'.format(self.voc_dir,idx))
        # read label
        img_file = '{}{}'.format(self.voc_dir,idx)
        img = self._decode_label(img_file)
        # delete the ignore_label
        #img[img == self.ignore_label] = 0
        # resize label
        out_size = (self.crop_size - 1) / self.shrink_factor + 1
        #im = cv2.imread('{}{}'.format(self.voc_dir,idx)) 
        #instead of linear interpolation, we use nearest interpolation
        label = cv2.resize(img,(out_size,out_size),
                interpolation=cv2.INTER_NEAREST)
        #im = im.resize((self.crop_size,self.crop_size),Image.ANTIALIAS)
        #im = im.resize((out_size,out_size),Image.ANTIALIAS)

        #label = np.array(im,dtype=np.uint8)
        #print 'label max:',label.max()
        #print 'ignore:',self.ignore_label
        #this we do not need by set ignore_label to 0, we can use mask
        
        # extend the label axis
        label = label[...,np.newaxis]
        # generate the label mask
        label_mask = np.ones_like(label)
        label_mask[label == self.ignore_label] = 0
        # we need to set the ignore_label as 0 to remove the error label 
        label[label == self.ignore_label] = 0
        #print 'label_mask:',label_mask.shape
        #print label.shape
        #debug ncwh
        #label = label.transpose((2,1,0))
        #label = label[np.newaxis,...]
        #print 'label', label.shape
        return label, label_mask

#reader.register_pyreader(SegReader,'seg')
