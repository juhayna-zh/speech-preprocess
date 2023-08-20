from pipeline.tester import Tester
from model import create_BSRNN
import argparse
import yaml
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '7'
from shutil import copyfile

def main(args):
    print('Hello')
    config_path = args.config
    with open(config_path, mode='r') as f:
        opt = yaml.load(f, Loader=yaml.FullLoader)

    train_opt = opt['model']['config']
    with open(train_opt, mode='r') as f:
        train_opt = yaml.load(f, Loader=yaml.FullLoader)
    
    tester = Tester(make_model= create_BSRNN, opt=opt, train_opt=train_opt)
    tester.run()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="path to yaml config file for test.", 
                        default="speech-preprocess/config/test.yml")
    args = parser.parse_args()

    main(args)