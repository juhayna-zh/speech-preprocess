from pipeline.tester import Test
from model import create_BSRNN
import argparse
import yaml
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
from shutil import copyfile

def main(args):
    config_path = args.config
    with open(config_path, mode='r') as f:
        opt = yaml.load(f, Loader=yaml.FullLoader)
    
    model = create_BSRNN(opt)
    tester = Tester(model, opt=opt)
    trainer.run()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="the MODEL config yaml file.", default="speech-preprocess/config/train.yml")
    parser.add_argument("--checkpoint", type=str, help="path to PyTorch checkpoint(.pt) file.")
    parser.add_argument("--data_path", type=str, help="directory path to load audio data.", default="speech-preprocess/config/train.yml")
    parser.add_argument("--save_path", type=str, help="directory path to save result audio.", default="speech-preprocess/config/train.yml")
    parser.add_argument("--no_criterion", type=bool, help="will not do criterion, use this option if no GT offered.", default=False)
    args = parser.parse_args()

    main(args)