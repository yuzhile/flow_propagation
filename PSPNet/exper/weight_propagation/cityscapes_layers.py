import sys
import numpy as np
import lmdb
import random
from PIL import Image
sys.path.insert(0,'/home/yuzhile/toolboxes/utils')
from dataset import Cityscapes
from transformer import random_crop
sys.path.insert(0,'/home/yuzhile/work2017/flow_propagation/PSPNet/python')
import caffe

import matplotlib.pyplot as plt
class CityscapesSegDataLayer(caffe.Layer):
    """
    Load (input image, label image) fine annotation pairs from Cityscapes
    one-at-a-time while reshaping the net to preserve dimensions.

    Use this to feed data to a fully convolutional network.

    N.B. Only half and image is loaded at a time due to memory constraints, but
    care is taken to guarantee equivalence to whole image processing. Every
    crop must be processed for this equivalence to hold, effectively making the
    training + val sets twice as large for indexing and the like.
    """

    def setup(self, bottom, top):
        """
        Setup data layer according to parameters:

        - dir: path to Cityscapes dir
        - split: train/val/trainval/test
        - randomize: load in random order (default: True)
        - seed: seed for randomization (default: None / current time)

        for Cityscapes semantic segmentation.

        example

        params = dict(dir='/path/to/Cityscapes', split='val')
        """
        # config
        params = eval(self.param_str)
        self.dir = params['cscapes_dir']
        print self.dir
        self.split = params['split']
        #self.mean = np.array((72.78044, 83.21195, 73.45286), dtype=np.float32)
        self.mean = np.array((103.999, 116.779, 123.68), dtype=np.float32)
        self.random = params.get('randomize', True)
        self.seed = params.get('seed', None)
        #self.crop = params.get('crop',False)
        self.crop_witdh = params.get('crop_witdh',0)
        self.crop_height = params.get('crop_height',0)
        self.net_dir = '/home/yuzhile/work2017/flow_propagation/PSPNet/exper'

        self.feat = params.get('feat',False)
        if self.feat:
            #caffe.set_mode_gpu()
            self.resnet = caffe.Net('{}/weight_propagation/pspnet101_cityscapes_713.prototxt'.format(self.net_dir),'{}/model/pspnet101_cityscapes.caffemodel'.format(self.net_dir),caffe.TEST)
            #self.resnet = caffe.Net('{}/weight_propagation/pspnet101_cityscapes_713.prototxt'.format(self.net_dir),'{}/model/pspnet101_cityscapes.caffemodel'.format(self.net_dir),caffe.TEST)
            #caffe.set_device(7)
            self.layer_name = params.get('layer_name','Interp')

        self.image_env = lmdb.open('{}/trainval_lmdb'.format(self.dir),readonly=True)
        self.image_txn = self.image_env.begin()
        self.sequence_env = lmdb.open('/DATA/segmentation/sequence_lmdb',readonly=True)
        self.sequence_txn = self.sequence_env.begin()
        self.seq_len = params.get('seq_len',1)

        self.error_list = open('{}/error_list.txt'.format(self.dir),'r').read().splitlines()



        # import cityscapes label helper and set up label mappings
        sys.path.insert(0, '{}/scripts/helpers/'.format(self.dir))
        labels = __import__('labels')
        self.id2trainId = {label.id: label.trainId for label in labels.labels}  # dictionary mapping from raw IDs to train IDs

        # two tops: data and label
        if not self.feat and len(top) != 3:
            raise Exception("Need to define three tops: current_im, key_im and label.")
        if self.feat and len(top) != 4:
            raise Exception("Need to define four tops: current_im, key_im label and feat.")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")

        # load indices for images and labels
        self.indices = []

        #for s in self.split:
        #    split_f = '{}/{}.txt'.format(self.dir, s)
        #    self.indices.extend(self.prepare_input(open(split_f, 'r').read().splitlines(), s))
        split_f = '{}/{}.txt'.format(self.dir, self.split)
        #self.indices.extend(self.prepare_input(open(split_f,
        #    'r').read().splitlines(), self.split))
        self.indices = open(split_f,'r').read().splitlines()
        self.idx = 0

        self.order =  range(len(self.indices)) 

        # make eval deterministic
        if 'train' not in self.split:
            self.random = False

        # randomization: seed and pick
        if self.random:
            random.seed(self.seed)
            #random.shuffle(self.indices)
            random.shuffle(self.order)

    def reshape(self, bottom, top):

        # load image + label image pair
        isTrain = False
        if self.random:
            print self.indices[self.order[self.idx]]
            isTrain = True
            self.current_im,self.key_im,self.label = Cityscapes.lmdb_load_image_label(self.image_txn,self.sequence_txn,self.indices[self.order[self.idx]],self.error_list,'separate',self.seq_len)
               #self.current_im,self.key_im,self.label = random_crop(self.current_im,self.key_im,self.label,self.crop_height,self.crop_witdh,self.mean)
        else:
            print 'data from sequence image'
            self.current_im,self.key_im = Cityscapes.load_sequence_image(self.indices[self.order[self.idx]],self.split,self.seq_len)

            self.label = Cityscapes.load_label(self.indices[self.idx],self.split)
         
        if self.crop_height != 0:
            assert self.crop_height > 0, 'crop height should positive'
            if self.crop_witdh == 0:
                self.crop_witdh = self.crop_height
            self.current_im,self.key_im,self.label = random_crop(self.current_im,self.key_im,self.label,self.crop_height,self.crop_witdh,self.mean,isTrain) # set False to debug
             
        #substract mean
        self.current_im -= self.mean
        self.key_im -= self.mean
        self.current_im = self.current_im.transpose((2,0,1))
        self.key_im = self.key_im.transpose((2,0,1))


        # reshape tops to fit (leading 1 is for batch dimension)
    
        if self.feat:
            caffe.set_device(4)
            self.resnet.blobs['data'].reshape(1,*self.current_im.shape)
            self.resnet.blobs['data'].data[...] = self.current_im
            #out = self.resnet.forward(start='conv1_1_3x3_s2',end=self.layer_name)

            self.resnet.forward(start='conv1_1_3x3_s2')
            out = self.resnet.blobs[self.layer_name]
            #self.resnet
            self.feat_map = out.data
        #    caffe.set_mode_gpu()
        #    del self.resnet
            caffe.set_device(5)
            top[3].reshape(*self.feat_map.shape)

        top[0].reshape(1, *self.current_im.shape)
        top[1].reshape(1,*self.key_im.shape)
        top[2].reshape(1,1, *self.label.shape)


    def forward(self, bottom, top):
        # assign output
        top[0].data[...] = self.current_im
        top[1].data[...] = self.key_im
        top[2].data[...] = self.label
        if self.feat:
            top[3].data[...] = self.feat_map

        # pick next input
        self.idx += 1
        #self.idx = 0
        if self.idx == len(self.indices):
            if self.random:
                #random.shuffle(self.indices)
                random.shuffle(self.order)
            self.idx = 0
    def backward(self, top, propagate_down, bottom):
                
        pass

    def prepare_input(self, indices, split):
        """
        Augment each index with left/right pair and its split for loading
        half-image crops to cope with memory limits.
        """
        full_indices = [(idx, split, 'right') for idx in indices]
        full_indices.extend([(idx, split, 'left') for idx in indices])
        return full_indices

    def half_crop_image(self, im, position, label=False):
        """
        Generate a crop of full height and width = width/2 + overlap.
        Align the crop along the left or right border as specified by position.
        If the image is a label, ignore the pixels in the overlap.
        """
        overlap = 210
        w = im.shape[1]
        if position == 'left':
            crop = im[:, :(w / 2 + overlap)]
            if label:
                crop[:, (w / 2):(w / 2 + overlap)] = 255
        elif position == 'right':
            crop = im[:, (w/2 - overlap):]
            if label:
                crop[:, :overlap] = 255
        else:
            raise Exception("Unsupported crop")
        return crop

    def load_image(self, idx, split, position):
        """
        Load input image and preprocess for Caffe:
        - cast to float
        - switch channels RGB -> BGR
        - subtract mean
        - transpose to channel x height x width order
        """
        full_im = np.array(Image.open('{}/leftImg8bit/{}/{}_leftImg8bit.png'.format(self.dir, split, idx)), dtype=np.uint8)
        im = self.half_crop_image(full_im, position, label=False)
        in_ = im.astype(np.float32)
        in_ = in_[:, :, ::-1]
        in_ -= self.mean
        in_ = in_.transpose((2, 0, 1))
        return in_

    def assign_trainIds(self, label):
        """
        Map the given label IDs to the train IDs appropriate for training
        This will map all the classes we don't care about to label 255
        Use the label mapping provided in labels.py
        """
        for k, v in self.id2trainId.iteritems():
            label[label == k] = v
        return label

    def load_label(self, idx, split, position):
        """
        Load label image as 1 x height x width integer array of label indices.
        The leading singleton dimension is required by the loss.
        """
        full_label = np.array(Image.open('{}/gtFine/{}/{}_gtFine_labelIds.png'.format(self.dir, split, idx)), dtype=np.uint8)
        label = self.half_crop_image(full_label, position, label=True)
        label = self.assign_trainIds(label)
        label = np.array(label, dtype=np.uint8)
        label = label[np.newaxis, ...]
        return label
