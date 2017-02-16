import numpy as np
from PIL import Image
import os
import sys
sys.path.insert(0,'../python')
import caffe
sys.path.append('/home/yuzhile/toolboxes/visual')
sys.path.append('/home/yuzhile/toolboxes/utils')
sys.path.append('/home/yuzhile/work2017/flow_propagation/parrots-deeplab/uitls')
sys.path.append('/home/yuzhile/work2017/flow_propagation/parrots-propagation/utils')
from cityscapes_lmdb_reader import CityscapesLmdbReader
from cityscapes_sequence_reader import CityscapesSequenceReader
#from pyparrots.env import Environ
import yaml
from visual import Visual
from transformer import zoom
from h5_tools import read_h5_data
import cv2
from transformer import split_forward
class CaffeNet:
    '''
    It's just implement the process interface which is feeded into a image and return feature
    '''


    def config(self,gpu_ids,caffe_prototxts=None,caffe_weights=None):
        '''
        gpu is defaultly used
        inputs:
        - gpu_ids: the gpu ids appointed to net, may one or two gpus
        - caffe_prototxt
        - caffe_weights
        '''
        caffe.set_mode_gpu()
        #caffe.set_device(gpu_id)
        self.gpu_ids = gpu_ids
        if caffe_prototxts is not None:
            self.res_net = caffe.Net(caffe_prototxts[0],caffe_weights[0],caffe.TEST)
        else:
            self.res_net = caffe.Net('./model/pspnet101_cityscapes_713.prototxt','./model/pspnet101_cityscapes.caffemodel',caffe.TEST)
        if len(gpu_ids) > 1:
            if caffe_prototxts  is not None:
                self.vgg_net = caffe.Net(caffe_prototxts[1],caffe_weights[1],caffe.TEST)
            else:
                self.vgg_net = caffe.Net('./weight_propagation/prop_split_val_vgg.prototxt','./weight_propagation/prop_vgg/train_9x9_mean_new_iter_8000.caffemodel',caffe.TEST)
                #self.vgg_net = caffe.Net('./weight_propagation/prop_split_val_vgg.prototxt','./weight_propagation/prop_vgg/train_9x9_mean_l1_iter_1000.caffemodel',caffe.TEST)

    def process(self,img_list,start='conv1_1_3x3_s2'):
        '''
        each img in img_list should preprocess:
        1) substracted mean
        2) brg channels
        3) suitable size for net.forward
        returns:
        - out with shape (feat_cha,h,w)
        '''
        caffe.set_device(self.gpu_ids[0])
        self.res_net.blobs['data'].reshape(1,*img_list[0].shape)
        self.res_net.blobs['data'].data[...] = img_list[0]
        out =  self.res_net.forward(start=start)
        assert len(self.gpu_ids) == len(img_list), "len of img_list equal to number of gpu_ids"
        if len(self.gpu_ids) > 1:
            feat = self.res_net.blobs['conv5_4'].data
            caffe.set_device(self.gpu_ids[1])
            self.vgg_net.blobs['current_data'].reshape(1,*img_list[0].shape)
            self.vgg_net.blobs['current_data'].data[...] = img_list[0] 
            self.vgg_net.blobs['key_data'].reshape(1,*img_list[1].shape)
            self.vgg_net.blobs['key_data'].data[...] = img_list[1] 
            self.vgg_net.blobs['feat'].reshape(*feat.shape)
            self.vgg_net.blobs['feat'].data[...] = feat
            out = self.vgg_net.forward(start='data_concat')
            #label = net_vgg.blobs['conv6_interp'].data[0]

        return out['conv6_interp'][0]
def double_split_forward(reader,process,feat_cha,crop_size):
    '''
    split forward for two sequence frames.
    inputs:
    - reader: supply feed data
    - feat_cha: a number represent the output channels dim
    - crop_size: image crop size suitable for net forward
    returns:
    - feat_map: with shape(feat_cha,h,w) and can generated label by argmax
    '''
    cur_im, key_im, label,label_mask,_ =reader.read()

    feat_map = split_forward(process,[cur_im.T,key_im.T],feat_cha,crop_size)
    return feat_map,label

def single_split_forward(reader,process,feat_cha,crop_size):
    '''
    split forward for single frame.
    inputs:
    - reader: supply feed data
    - feat_cha: a number represent the output channels dim
    - crop_size: image crop size suitable for net forward
    returns:
    - feat_map: with shape(feat_cha,h,w) and can generated label by argmax
    '''
    data, key_im, label,label_mask,_ =reader.read()

    feat_map = split_forward(process,[data.T],feat_cha,crop_size)
    return feat_map,label

def evaluation(reader,process,split_forward,save_root,val_file,debug = False):
    '''
    save gray image for evaluation
    '''
    
    #cfg_text = cfg_text.substitute(Environ().map)
    org_h = 1024
    org_w = 2048
    if not os.path.exists(save_root):
        os.mkdir(save_root)
    val_list = open(val_file,'r').read().splitlines()
    suffix = '_leftImg8bit'

    crop_size = 713
    feat_cha = 19
    
    i = 1
    num_val = len(val_list)
    if debug:
        my_visual = Visual()
    for line in val_list:
        print 'handling {}/{} {}.'.format(i,num_val,line)
        i += 1
        img_name = line.split('/')[1]
        #data, label, label_mask =cityscape_lmdb.read()
        #caffe_processor.config()
        print 'split forward begin'
        feat_map,true_label = split_forward(reader,process,feat_cha,crop_size)
        #need to resize to orig size
        org_feat_map = cv2.resize(feat_map.transpose((1,2,0)),(org_w,org_h))
        pred_label = org_feat_map.argmax(axis=2)    
        save_name = os.path.join(save_root,img_name+suffix+'.png')
        cv2.imwrite(save_name,pred_label)
        true_label = true_label.T
        true_label = true_label[0]
        if debug:
            if pred_label.shape != true_label.shape:
                true_label = cv2.resize(true_label,(org_w,org_h),interpolation=cv2.INTER_NEAREST)
            label = np.concatenate((pred_label,true_label),axis=0)
            my_visual.draw_label(label)
if __name__ == '__main__':
    #get config file to construct reader
    session_file = './extract_val.yaml'
    with open(session_file,'r') as fin:
        cfg_text = fin.read()
    session_yaml = yaml.load(cfg_text)
    flow_yaml = session_yaml['flows'][0]['val']
    feeder_yaml = flow_yaml['feeder']
    reader_cfg = feeder_yaml['pipeline'][0]['attr']

    #construct reader
    #cityscape_lmdb = CityscapesLmdbReader()
    cityscape_lmdb = CityscapesSequenceReader()
    cityscape_lmdb.config(reader_cfg)

    #save_root = './result/cityscapes/val/debug'
    #save_root = './result/cityscapes/val/label_prop_l1_softmax'
    save_root = './result/cityscapes/val/weight_prop_9x9_relu_l4_8000'
    val_file = '/home/yuzhile/data/cityscapes/val.txt'
    caffe.set_mode_gpu()
    caffe_processor = CaffeNet()
    gpu_ids = [6,7]
    caffe_processor.config(gpu_ids)
    if len(gpu_ids) > 1:
        forward = double_split_forward
    else:
        forward = single_split_forward
    evaluation(cityscape_lmdb,caffe_processor.process,forward,save_root,val_file,False)
