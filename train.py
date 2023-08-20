from pipeline.trainer import Trainer
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
    if not opt['resume']['state']:
        os.makedirs(opt['train']['save_stat_dir'], exist_ok=True)
        copyfile(config_path, opt['train']['save_stat_dir']+os.sep+'config.yml')
    
    model = create_BSRNN(opt)
    trainer = Trainer(model, opt=opt)
    trainer.run()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="where yaml config file path.", default="speech-preprocess/config/train.yml")
    
    args = parser.parse_args()

    main(args)