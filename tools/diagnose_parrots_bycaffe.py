import sys
sys.path.insert(0,'../python')

import caffe
#import surgery, score

import numpy as np
import os
import sys
sys.path.append('/home/yuzhile/toolboxes/utils')
sys.path.append('/home/yuzhile/toolboxes/visual')
from visual import Visual
from h5_tools import read_h5_data
import cv2
from transformer import zoom

#the extracted data from parrots
#parrots_data_file = '../../parrots-deeplab/model/extract_all.h5'
# the fc6 split data
parrots_data_file = '../../parrots-deeplab/model/parrots_debug_split.h5'
#the parrots weight
parrots_weights_file = '../../parrots-deeplab/model/deeplab_arg.parrots'
# read all extracted data from hf file
parrots_data = read_h5_data(parrots_data_file)
#read parrots weight
parrtos_weights = read_h5_data(parrots_weights_file)




# caffe weights file
weights = './voc12/model/deeplab_largeFOV_arg/train_iter_6000.caffemodel'


solver = caffe.SGDSolver('./voc12/config/deeplab_largeFOV_arg/solver_train_src.prototxt')

solver.net.copy_from(weights)

solver.test_nets[0].set_device(0)
solver.test_nets[0].set_mode_gpu()
solver.test_nets[0].share_with(solver.net)
solver.test_nets[0].set_phase_test()
parrots_data_keys = parrots_data.keys()
parrtos_weights_keys = parrtos_weights.keys()
#print 'data keys from parrots',parrots_data.keys()
#print 'params keys from parrots',parrtos_weights.keys()
# scoring
test = np.loadtxt('./voc12/list/val.txt', dtype=str)
# get the batch size
num_samples = parrots_data['data'].shape[0]
my_visual = Visual()
def feature_to_label(feat_map,zoom_factor=1):
    '''
    from feature blob to label
    inputs
    - feat_map: with shape(n,c,h,w)
    returns
    - label: with shape(h,w)
    '''
#    print feat_map.shape
#    fc8_voc12 = feat_map[0,:,:,:]
    #print feat_map[0,:,:]
    zoom_feat_map = zoom(feat_map,zoom_factor)
    label = zoom_feat_map.argmax(axis=0)
    return label
def diff_values(value1,value2):
    '''
    compute the differences between value1 and value2 by norming the difference of value1 and value2
    inputs
    - value1: np.ndarray
    - value2: np.ndarray with the same dim as value1
    returns:
    - diff_norm
    '''
    return np.linalg.norm(value1.flatten()-value2.flatten())
for i  in range(num_samples):
    #get predict label by parrots
    #parrots_label = feature_to_label(values[0][i],1) 
    #print 'parrots label', parrots_label.shape
    #get predict label by caffe
    image_blob = parrots_data['data'][i,:,:,:]
    solver.test_nets[0].blobs['data'].reshape(1,*image_blob.shape)
    solver.test_nets[0].blobs['data'].data[...] = image_blob
    solver.test_nets[0].forward(start='conv1_1')
    print 'phase'
    caffe_blobs_keys = solver.test_nets[0].blobs.keys()
    for caffe_blobs_key in caffe_blobs_keys:

        #compare data
        print 'handling layer {}.'.format(caffe_blobs_key)
        if parrots_data.has_key(caffe_blobs_key):
            parrots_layer_data = parrots_data[caffe_blobs_key][i,:,:,:]
            caffe_layer_data = solver.test_nets[0].blobs[caffe_blobs_key].data
            print 'parrot caffe data',parrots_layer_data.shape,caffe_layer_data.shape
            data_diff = diff_values(caffe_layer_data,parrots_layer_data)
            print '{} layer data difference between caffe and parrots is {}'.format(caffe_blobs_key,data_diff)
        # when the layer coresponding to the blob has params
        if 'conv' in caffe_blobs_key:
            caffe_params = solver.test_nets[0].params[caffe_blobs_key][0].data
            
            parrots_params = parrtos_weights[caffe_blobs_key+'_0@value']
            conv_param_diff = diff_values(caffe_params,parrots_params)
            print '{} layer params differene between caffe and parrots is {}'.format(caffe_blobs_key,conv_param_diff)
            caffe_baises = solver.test_nets[0].params[caffe_blobs_key][1].data
            parrots_baises = parrtos_weights[caffe_blobs_key+'_1@value']
            conv_bais_diff = diff_values(caffe_baises,parrots_baises)
            print '{} layer biases differene between caffe and parrots is {}'.format(caffe_blobs_key,conv_bais_diff)

    #print solver.test_nets[0].blobs.keys()
    #caffe_map = solver.test_nets[0].blobs['fc8_voc12'].data[0,:,:,:]
    #caffe_label = feature_to_label(caffe_map,1)

    
    #concate_label = np.concatenate((parrots_label,caffe_label),axis=0)
    #score.seg_tests(solver,False, test, layer='fc8_voc12', gt='label',zoom_factor=8)
    #my_visual.draw_label(concate_label,'./classes.txt')
