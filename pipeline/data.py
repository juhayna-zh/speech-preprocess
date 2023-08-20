from torch.utils.data import Dataset, DataLoader
import os
import jsonlines
import torchaudio
from torchaudio.transforms import Resample
from tqdm import tqdm

def read_audio(path, return_sr=False, resample_sr=None):
    audio, sr = torchaudio.load(path)
    if resample_sr is not None and resample_sr != sr:
        audio = Resample(sr, resample_sr)(audio)
        sr = resample_sr
    if return_sr:
        return audio,sr
    else:
        return audio


def read_metadata(data_dir):
    metadata_path = data_dir + os.sep + 'metadata.jsonl'
    metadata = []
    with jsonlines.open(metadata_path, mode='r') as reader:
        for line in reader:
            metadata.append(line)
    return metadata

def read_3_audios(data_dir, i, sr):
    mixn_audio = read_audio(data_dir+os.sep+'mixn'+os.sep+str(i)+'.wav', resample_sr=sr)
    spk1_audio = read_audio(data_dir+os.sep+'spk1'+os.sep+str(i)+'.wav', resample_sr=sr)
    spk2_audio = read_audio(data_dir+os.sep+'spk2'+os.sep+str(i)+'.wav', resample_sr=sr)
    return mixn_audio, spk1_audio, spk2_audio

class MixnAudioDataset(Dataset):
    def __init__(self, data_dir:str, sr=44100, preload=False, sub_batch=None):
        super().__init__()
        self.data_dir = data_dir
        self.metadata = read_metadata(data_dir)
        self.preload = preload
        self.sr = sr
        self.sub_batch = sub_batch
        self.current_sub_batch = 0
        if preload:
            self.mixn = []
            self.spk1 = []
            self.spk2 = []
            for i,md in enumerate(tqdm(self.metadata, desc="LoadData")):
                mixn_audio, spk1_audio, spk2_audio = read_3_audios(data_dir, i, sr=self.sr)
                self.mixn.append(mixn_audio)
                self.spk1.append(spk1_audio)
                self.spk2.append(spk2_audio)



    def __getitem__(self, i):
        if self.sub_batch:
            i = self.current_sub_batch * self.sub_batch + i
        if self.preload:
            return self.mixn[i], self.spk1[i], self.spk2[i], self.sr
        else: 
            return *read_3_audios(self.data_dir, i, sr=self.sr), self.sr
    
    def __len__(self):
        if self.sub_batch:
            return self.sub_batch
        else:
            return len(self.metadata)

    def next_epoch(self):
        if self.sub_batch:
            self.current_sub_batch += 1
            if self.sub_batch * self.current_sub_batch >= len(self.metadata):
                self.current_sub_batch = 0

if __name__ == '__main__':
    dataset = MixnAudioDataset(
        '/mnt/lustre/sjtu/home/hnz01/code/AILabInTask1/speech-preprocess/data/DatasetTest12h',
        preload=False)
    # dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # for data in dataloader:
    #     print(data.shape)
    #     break
    