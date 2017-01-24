from pyparrots.dnn import Model, Session
import yaml
import  argparse
from pyparrots.dnn import Model, Session
import string
from pyparrots.env import Environ
import sys
sys.path.append('/home/yuzhile/toolboxes/visual')
sys.path.insert(0,'../uitls')
from cityscapes_reader import CityscapesReader
import visual
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Debug the parrots reader in python")

parser.add_argument('--session',help='the session definition file')
args = parser.parse_args()
def load_model(model_cfg):
    if 'yaml' in model_cfg:
        with open(model_cfg['yaml'],'r') as fin:
            model_text = fin.read()
        return Model.from_yaml_text(model_text)
def debug_h5_reader():
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
    train_flow = session_yaml['flows'][0]['train']
    print train_flow.keys()
    train_feeder = train_flow['feeder']
    print train_feeder['pipeline']
    reader_cfg = train_feeder['pipeline'][0]['attr']
    print reader_cfg

    h5reader = PyH5Reader()
    h5reader.config(reader_cfg)
    x, y = h5reader.read()
    print 'x',x.shape
    print 'y',y.shape
def test_cityscapes_reader(flow_name='train'):
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
    cityscapes_reader = CityscapesReader()
    cityscapes_reader.config(reader_cfg)
    #seg_reader.read()
    for _ in range(100):
        image, label,mask = cityscapes_reader.read()
        plt.imshow(image[:,:,::-1])
        plt.show()
        
#    import matplotlib.pyplot as plt #    plt.figure()
#    plt.imshow(x)
#    plt.figure()
#    plt.imshow(y[:,:,0])
#    plt.show()
    #my_visual = visual.Visual()
    #my_visual.rescore(x,y[:,:,0],'./classes.txt')
if __name__ == '__main__':
    test_cityscapes_reader('train')
