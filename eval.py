from model import create_BSRNN
import yaml
import torch
from pipeline.data import MixnAudioDataset
import os
from torch.utils.data import DataLoader
import torchaudio
from tqdm import tqdm
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio

config_path = 'speech-preprocess/save/lay10_8w/trainstat/config.yml'

with open(config_path, mode='r') as f:
    opt = yaml.load(f, Loader=yaml.FullLoader)
    
device = "cpu"
model = create_BSRNN(opt)

checkpoint_path = 'speech-preprocess/save/lay10_8w/checkpoint/best.pt'

model_state_dict = torch.load(checkpoint_path, map_location=device)['model']
model.load_state_dict(model_state_dict)
model.to(device)
model.eval()

test_data_dir = 'speech-preprocess/data/Data_100'

test_result_dir = test_data_dir+'_Result'
os.makedirs(test_result_dir, exist_ok=True)
os.makedirs(test_result_dir+os.sep+'spk1_est', exist_ok=True)
os.makedirs(test_result_dir+os.sep+'spk2_est', exist_ok=True)

dataloader = DataLoader(MixnAudioDataset(test_data_dir, preload=False), batch_size=1)

sisdr = ScaleInvariantSignalDistortionRatio()

sisnr1, sisnr2 = 0, 0
i = 0

for (mixn, spk1, spk2, sr) in tqdm(dataloader):
    spk1_est, spk2_est = model(mixn.to(device))
    torchaudio.save(test_result_dir+os.sep+'spk1_est'+os.sep+str(i)+'.wav', spk1_est.squeeze(0).cpu(), sr)
    torchaudio.save(test_result_dir+os.sep+'spk2_est'+os.sep+str(i)+'.wav', spk2_est.squeeze(0).cpu(), sr)
    sisnr1 += sisdr(spk1_est, spk1)
    sisnr2 += sisdr(spk2_est, spk2)
    i += 1

print("sisnr1:", sisnr1/i)
print("sisnr2:", sisnr2/i)
print("Done.")