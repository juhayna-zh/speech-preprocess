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
    """读取数据集的metadata文件"""
    metadata_path = data_dir + os.sep + 'metadata.jsonl'
    metadata = []
    with jsonlines.open(metadata_path, mode='r') as reader:
        for line in reader:
            metadata.append(line)
    return metadata

def read_3_audios(data_dir, i, sr):
    """读取mixn,spk1,spk2目录下对应的音频"""
    mixn_audio = read_audio(data_dir+os.sep+'mixn'+os.sep+str(i)+'.wav', resample_sr=sr)
    spk1_audio = read_audio(data_dir+os.sep+'spk1'+os.sep+str(i)+'.wav', resample_sr=sr)
    spk2_audio = read_audio(data_dir+os.sep+'spk2'+os.sep+str(i)+'.wav', resample_sr=sr)
    return mixn_audio, spk1_audio, spk2_audio

class MixnAudioDataset(Dataset):
    """用于加载混合加噪语音数据的Dataset.
    'Mixn'的含义是'Mix'+'Noise'
    """
    def __init__(self, data_dir:str, sr=44100, preload=False, sub_batch=None):
        super().__init__()
        self.data_dir = data_dir
        self.metadata = read_metadata(data_dir)
        self.preload = preload
        self.sr = sr
        self.sub_batch = sub_batch
        self.current_sub_batch = 0
        if preload:
            #使用此选项会预加载所有数据.占用内存巨大,谨慎使用.
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
            #如果使用sub_batch,则每次只返回sub_batch个数据
            return self.sub_batch
        else:
            return len(self.metadata)

    def next_epoch(self):
        #移动到下一个sub_batch
        if self.sub_batch:
            self.current_sub_batch += 1
            if self.sub_batch * self.current_sub_batch >= len(self.metadata):
                self.current_sub_batch = 0


def get_filenames(data_dir):
    """获取数据集目录下的所有音频文件名"""
    return [f for f in os.listdir(data_dir+os.sep+'mixn') if f[-4:] in ('.wav','.mp3','flac','.ogg')]

class TestMixnAudioDataset(Dataset):
    """测试时使用的Dataset,与MixnAudioDataset的区别如下:
    1.不支持sub_batch,preload等训练时属性
    2.输入sub_id,num_threads以支持读取多线程中读取特定线程的数据
    3.有no_GT参数,如果为True,则不会读取和返回标签,以适用于没有提供GT的数据
    4.允许数据集不提供metadata.jsonl文件,以方便各种数据的测试
    """
    def __init__(self, data_dir:str, filenames, sub_id, num_threads, sr=44100, no_GT=False):
        super().__init__()
        self.data_dir = data_dir
        self.sr = sr
        self.no_GT = no_GT
        
        total = len(filenames)
        self.filenames = [filenames[i] for i in range(sub_id, total, num_threads)] #通过sub_id筛选出本线程的数据的文件名

    def __getitem__(self, i):
        filename = self.filenames[i]
        mixn_audio = read_audio(self.mixn_dir_+filename, resample_sr=self.sr)
        spk1_audio = None if self.no_GT else read_audio(self.spk1_dir_+filename, resample_sr=self.sr)
        spk2_audio = None if self.no_GT else read_audio(self.spk2_dir_+filename, resample_sr=self.sr)
        return mixn_audio, spk1_audio, spk2_audio, self.sr, filename
    
    def __len__(self):
        return len(self.filenames)
    
    @property 
    def mixn_dir_(self):
        return self.data_dir+os.sep+'mixn'+os.sep

    @property 
    def spk1_dir_(self):
        return self.data_dir+os.sep+'spk1'+os.sep
    
    @property 
    def spk2_dir_(self):
        return self.data_dir+os.sep+'spk2'+os.sep
    

if __name__ == '__main__':
    dataset = MixnAudioDataset(
        '/mnt/lustre/sjtu/home/hnz01/code/AILabInTask1/speech-preprocess/data/DatasetTest12h',
        preload=False)
    # dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # for data in dataloader:
    #     print(data.shape)
    #     break
    