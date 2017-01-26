
import random
import numpy as np
from PIL import Image
import cv2
import sys
from multiprocessing import Process, Queue
import lmdb

class CityscapesLmdbReader:
    '''
    Load (input image, label image) pairs from PASCAL VOC
    one-at-a-time while reshaping the net to preserve dimensions

    Use this to feed data to a fully convolutional network.
    '''
    
    support_keys = ['split','data_dir','randomize','crop_size','ignore_label','shrink_factor','num_class','train','mirror']

    def config(self,cfg):
        '''
        Setup data parameters:
        - voc_dir: path to PASCAL VOC year dir
        - split: train / val / test
        - mean: tuple of mean values to subtrack
        - randomize: load in random order (default: True)
        
        '''

        self.split = cfg['split']
        self.data_dir = cfg['data_dir']
        self.mean = np.array((72.78044, 83.21195, 73.45286))
        self.random = cfg.get('randomize',False)
        self.seed = cfg.get('seed',None)
        self.mirror = cfg.get('mirror',False)
        self.crop_size = int(cfg['crop_size'])
        self.ignore_label = int(cfg['ignore_label'])
        self.shrink_factor = int(cfg['shrink_factor'])
        self.num_class = int(cfg.get('num_class',20))
        self.train = cfg.get('train',False)
        self.indices = self._prepare_input(open('{}/{}.txt'.format(self.data_dir,self.split),'r').read().splitlines(),self.split)
        self.idx = 0
        self.w = 2048
        self.h = 1024
        self.c = 4
        env = lmdb.open('{}/trainval_lmdb'.format(self.data_dir),readonly=True)
        self.txn = env.begin()
        self.iter = self._read_iter()
        #self.iter = self._queue_read_iter()
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

    def _read_image(self,q,order):
        for i in order:
            idxs = self.indices[i]
            image = self._load_image(idxs[0],idxs[1],idxs[2])
            label = self._load_label(idxs[0],idxs[1],idxs[2])
            q.put([image,label])
    def _process_image(self,q):
        while not q.empty():
            image,label = q.get()
            if self.mirror and np.random.randint(2):
                image = cv2.flip(image,1)
                label = cv2.flip(label,1)

            h, w, c = image.shape

            crop_img, crop_label = self._deeplab_crop(image,label) 
            # get label mask
            label_shrink,label_mask = self._transform_label(crop_label)
            # substract mean
            crop_img -= self.mean

            # The data should be transposed for the data in caffe is chw,but parrots is whc
            # transpose hwc -> whc
            yield [crop_img.transpose((1,0,2)),label_shrink.transpose((1,0,2)), label_mask.transpose((1,0,2))]


    def _queue_read_iter(self):
        '''
        Read image using multiprocessing
        '''
        while True:
            size = len(self.indices)
            order = range(size)
            #debug code
            #idxs = self.indices[0]
            #image = self._load_image(idxs[0],idxs[1],idxs[2])
            #label = self._load_label(idxs[0],idxs[1],idxs[2])
            if self.random:
                random.shuffle(order)
            q = Queue()
            p = Process(target=self._read_image, args=(q,order))
            p.start()
            while p.is_alive():
                while not q.empty():
                    image, label = q.get()

                    #apply mirror transformer
                    if self.mirror and np.random.randint(2):
                        image = cv2.flip(image,1)
                        label = cv2.flip(label,1)


                    crop_img, crop_label = self._deeplab_crop(image,label) 
                    # get label mask
                    label_shrink,label_mask = self._transform_label(crop_label)
                    # substract mean
                    crop_img -= self.mean

                    # The data should be transposed for the data in caffe is chw,but parrots is whc
                    # transpose hwc -> whc
                    yield [crop_img.transpose((1,0,2)),label_shrink.transpose((1,0,2)), label_mask.transpose((1,0,2))]
            p.join()

 
    def _read_iter(self):
        while True:
            size = len(self.indices)
            order = range(size)
            #debug code
            #idxs = self.indices[0]
            #image = self._load_image(idxs[0],idxs[1],idxs[2])
            #label = self._load_label(idxs[0],idxs[1],idxs[2])
            if self.random:
                random.shuffle(order)
            for i in order:
                idxs = self.indices[i]
#                print 'handling {}:{}'.format(i,idxs)

                raw_data = np.fromstring(self.txn.get(idxs[0]),dtype='uint8')
                ndata = raw_data.reshape(self.h,self.w,self.c)
                image = ndata[:,:,:3]
                image = np.array(image,dtype=np.float32)
                label = ndata[:,:,-1]
                label = np.squeeze(label)
                #apply mirror transformer
                if self.mirror and np.random.randint(2):
                    image = cv2.flip(image,1)
                    label = cv2.flip(label,1)

                h, w, c = image.shape

                crop_img, crop_label = self._deeplab_crop(image,label) 
                # get label mask
                label_shrink,label_mask = self._transform_label(crop_label)
                # substract mean
                crop_img -= self.mean

                # The data should be transposed for the data in caffe is chw,but parrots is whc
                # transpose hwc -> whc
                yield [crop_img.transpose((1,0,2)),label_shrink.transpose((1,0,2)), label_mask.transpose((1,0,2))]

                #yield [image,label,src_img]
    def _prepare_input(self, indices, split):
        '''
        Augment each index with left/right pair and its split for loading
        half-image crops to cope with memory limits.
        '''
        full_indices = [(idx, split, 'right') for idx in indices]
        full_indices.extend([(idx, split, 'left') for idx in indices])
        return full_indices
    def half_crop_image(self, im, position, label=False):
        '''
        Generate a crop of full height and width = width/2 + overlap.
        Align the crop along the left or right border as specified by position.
        If the image is a label, ignore the pixels in the overlap.
        '''
        overlap = 210
        w = im.shape[1]
        if position == 'left':
            crop = im[:,:(w/2 + overlap)]
            if label:
                crop[:, (w / 2):(w / 2 + overlap)] = 255
        elif position == 'right':
            crop = im[:,(w / 2 - overlap):]
            if label:
                crop[:,:overlap] = 255
        else:
            raise Exception("Unsupported crop")
        return crop
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
        #check the id >=0
        raw_label[raw_label < 0] = self.ignore_label
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

    def _load_image(self,idx,split,position):
        '''
        Load input image and preprocess for parrots
        - cast to float
        '''
        
        #im = Image.open('{}{}'.format(self.voc_dir, idx))
        full_name = '{}/leftImg8bit/{}/{}_leftImg8bit.png'.format(self.data_dir,split,idx)
        im = cv2.imread(full_name)
        im = self.half_crop_image(im,position,label=False)
        in_ = np.array(im,dtype=np.float32)
        # RGB->BGR
        #in_ = im[:,:,::-1]
        # sub mean
        #in_ -= self.mean

        
        return in_
           
    def assign_trainIds(self,label):
        '''
        Map the given label IDs to the train IDs appropriate fro training.
        This will map all classes we don't care to ignre label 255.
        Use the label map provided in label.py
        '''
        for k,v in self.id2trainId.iteritems():
            label[label == k] = v
        return label
    def _decode_label(self,label_file):
        '''
        Read label image as raw data.

        '''
        with open(label_file,'rb') as infile:
            buf = infile.read()
        raw = np.fromstring(buf,dtype='uint8')
        img = cv2.imdecode(raw,cv2.IMREAD_UNCHANGED)
        return img



    def _load_label(self,idx,split,position):
        '''
        Load label image as height x width interger array of label indices.
        return:
        - label,with shape(h,w)
        '''
        full_name = '{}/gtFine/{}/{}_gtFine_labelIds.png'.format(self.data_dir,split,idx)
        label = self._decode_label(full_name)
        label = self.half_crop_image(label,position,label=True)
        label = self.assign_trainIds(label)
        assert len(label.shape) == 2,'the image seg labe should be 2 dim'
        return label

if 'TEST_CITYSCAPES' in globals():
    pass
else:
    from pyparrots.dnn import reader
    reader.register_pyreader(CityscapesLmdbReader,'CityscapesLmdb')
