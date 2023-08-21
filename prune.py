import argparse
import torch

def main(args):
    path = args.path
    pruned_path = path.replace('.pt','_model.pt')
    ckpt = torch.load(path)
    model_ckpt = ckpt['model']
    torch.save(model_ckpt, pruned_path)
    print('done.')

#修剪.pt文件，去除与训练相关的optimizer参数，仅保留模型参数
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', help='unpruned model checkpoint path(.pt)', default='speech-preprocess/save/x64s_8w/checkpoint/best.pt')
    args = parser.parse_args()
    main(args)