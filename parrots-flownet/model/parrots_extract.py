from pyparrots.dnn import Model, Session
import yaml
import  argparse
from pyparrots.dnn.pyreaders.pyh5reader import PyH5Reader
from pyparrots.dnn.config import SessionConfig
from pyparrots.dnn import Runner
import string
from pyparrots.env import Environ
import sys

sys.path.append('/home/yuzhile/toolboxes/visual')
sys.path.append('/home/yuzhile/toolboxes/utils')
sys.path.append('/home/yuzhile/work2017/flow_propagation/FlowNet/models/haze/scripts')
from flowlib import flow_to_image
from h5_tools import write_h5_data
from h5_tools import read_h5_data
import numpy as np
import matplotlib.pyplot as plt
#from caffe_data import caffe_read
def add_parser_extract(subparsers):

    parser_extract = subparsers.add_parser('extract',help='extract features')
    parser_extract.add_argument('--session',metavar='session.yaml',help='the session definition file')
    parser_extract.add_argument('in_filename',metavar='in.h5',help='filename for the input .h5')
    parser_extract.add_argument('out_filename',metavar='out.h5',help='filename for the output .h5')
     
def parse_args():
    parser = argparse.ArgumentParser(description="Debug the parrots extract in python")
    parser.add_argument('--env_path',metavar='path',default=None,
                    help='parrots env file path')
    parser.add_argument('--seed', metavar='random number', type=int,default=None,help='random seed for parrots')

    subparsers = parser.add_subparsers(dest='cmd', help='parrots CLI commands')
    add_parser_extract(subparsers)
    return parser.parse_args()

def extract(args):
    '''
    Extract data by session_extract and Runner, we first configue the extract session and then
    configure the runner to get forward flow. We feed input to forward flow and get the output of    flow.
    '''
    session_file = args.session
    runner = Runner(session_file,args)
    runner.setup(use_logger=False)
    print runner.primary_flow_cfg
    print 'cmd is',args.cmd
    #we set show_key = '' to disable show
    show_key = 'blob44'
    with runner.session.flow(runner.primary_flow_id) as flow:
        vars = runner.primary_flow_cfg['spec']['outputs']
        #seg_reader = SegReader()
        reader_cfg = runner.primary_flow_cfg['feeder']['pipeline'][0]['attr']
        #seg_reader.config(reader_cfg)
        num_samples = runner.config.config['sample_num']
        extract_dict = {}
        out_file = args.out_filename
        in_file = args.in_filename
        inputs = read_h5_data(in_file,['input'])
        for i in xrange(num_samples):
            print 'iter {}.'.format(i)
            #out = caffe_read().next()
            flow.set_input('input',inputs[0])
            flow.forward()
            for var in vars:
                data = flow.data(var).value()
                print 'layer {} output shape:'.format(var),data.shape
                if i == 0:
                    feat_shapes = list(reversed(data.shape))
                    feat_shapes[0] = num_samples
                    extract_dict[var] = np.zeros(feat_shapes,dtype=data.dtype)
                extract_dict[var][i] = data.T
                if var == show_key:
                    flow_image = flow_to_image(data[:,:,:,0].transpose((1,0,2)))
                    plt.imshow(flow_image)
                    plt.show()
                
        write_h5_data(out_file,extract_dict)
if __name__ == '__main__':
    args = parse_args()
    extract(args)
