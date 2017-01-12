from pyparrots.dnn import Model, Session
import yaml
import  argparse
from pyparrots.dnn import Model, Session
from flowreader import FlowReader
import string
from pyparrots.env import Environ
import sys
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="test the parrots reader in python")

parser.add_argument('--session',help='the session definition file')
args = parser.parse_args()
def load_model(model_cfg):
    if 'yaml' in model_cfg:
        with open(model_cfg['yaml'],'r') as fin:
            model_text = fin.read()
        return Model.from_yaml_text(model_text)
def test_reader(flow_name='train'):
    session_file = args.session
    with open(session_file,'r') as fin:
        cfg_text = fin.read()
    # to replace the env variable such as $HOME etc.
    cfg_text = string.Template(cfg_text)
    #print cfg_text
    cfg_text = cfg_text.substitute(Environ().map)
    #print cfg_text
    session_yaml = yaml.load(cfg_text)
    print session_yaml.keys()
    print session_yaml['model']
    train_flow = session_yaml['flows'][0][flow_name]
    print train_flow.keys()
    train_feeder = train_flow['feeder']
    print 'pipeline',train_feeder['pipeline']
    print train_feeder['pipeline'][0]
    reader_cfg = train_feeder['pipeline'][0]['attr']
    print reader_cfg
    print 'raondom type',type(reader_cfg['randomize'])
    if reader_cfg['randomize']:
        print 'start random'
    flow_reader = FlowReader()
    flow_reader.config(reader_cfg)
    #seg_reader.read()
    for _ in range(1):
        img1, img2,gt= flow_reader.read()
        plt.imshow(img1[:,:,::-1])
        plt.show()
        
if __name__ == '__main__':
    test_reader('train')
