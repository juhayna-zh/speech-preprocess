import torch 
import torchaudio
from torchaudio.transforms import Resample
import random
import numpy as np
import os
import jsonlines
from tqdm import tqdm
import gc
import random
from functools import lru_cache


def random_lh(low, high):
    return random.random() * (high - low) + low

def randn_lh(low, high):
    mid = (low+high)/2
    sig = (high - low) / 6
    return np.clip(np.random.normal(mid, sig), low, high)

def pad_or_trim(audio, sr, secs):
    length = int(sr * secs)
    if audio.shape[-1] < length:
        padding = torch.zeros(length - audio.shape[-1])
        offset = random.randint(0, padding.shape[0])
        padded = torch.cat([padding[:offset].unsqueeze(0), audio, padding[offset:].unsqueeze(0)], dim=-1)
        return padded
    else:
        offset = random.randint(0, audio.shape[-1] - length)
        trimmed = audio[..., offset:offset+length]
        return trimmed 

@lru_cache()
def load_noise(noise_path, to_sr):
    noise, noise_sample_rate = torchaudio.load(noise_path)
    noise = Resample(noise_sample_rate, to_sr)(noise)
    return noise

def load_and_mix_audio(audio1_path, audio2_path, noise_path, secs = 3,  \
    mix_ratio_lh = (0.3, 0.7), noise_ratio_lh = (0.01, 0.1)):
    # 加载两段单人语音
    waveform1, sample_rate1 = torchaudio.load(audio1_path)
    waveform2, sample_rate2 = torchaudio.load(audio2_path)

    # 调整采样率
    waveform1 = Resample(sample_rate1, sample_rate2)(waveform1)
    # 调整音频长度
    waveform1 = pad_or_trim(waveform1, sample_rate2, secs)
    waveform2 = pad_or_trim(waveform2, sample_rate2, secs)

    # 生成混合比例  
    mix_ratio = randn_lh(*mix_ratio_lh)

    # 混合语音
    waveform_mix = mix_ratio * waveform1 + (1 - mix_ratio) * waveform2 

    # 加载噪声
    noise = load_noise(noise_path, sample_rate2)
    start = random.randint(0, noise.shape[1] - int(sample_rate2*secs))
    noise = noise[:, start:start+int(sample_rate2*secs)]

    # 添加噪声
    noise_ratio = random_lh(*noise_ratio_lh)
    waveform_noise = waveform_mix + noise_ratio * noise

    return waveform_noise, waveform1, waveform2, {
        'sr': sample_rate2,
        'nframe': waveform_noise.shape[-1],
        'mix_r':mix_ratio,
        'noise_r':noise_ratio
    }

def make_audio_index(parent_dir:str, save_jsonl_path:str):
    """
    为音频编制索引,生成JSONL文件
    parent_dir:所有音频文件的父文件夹
    save_jsonl_path:保存JSONL文件的路径
    """
    real_path = save_jsonl_path + ''
    save_jsonl_path = save_jsonl_path.replace(".json", "_unsort.json")
    data = []
    print("Walking...")
    for root, dirs, files in os.walk(parent_dir):
        if root == parent_dir:
            continue
        speaker = os.path.basename(root)
        for f in tqdm(files, desc=speaker):
            info = torchaudio.info(root + os.sep + f)
            nframe = info.num_frames
            sr = info.sample_rate
            data.append({"speaker":speaker,
                        "file":f,
                        "nframe":nframe,
                        "sr":sr })
        if len(data) > 1000:
            print("Writing...")
            with jsonlines.open(save_jsonl_path, mode='a') as writer:
                for d in tqdm(data, desc="Write"):
                    writer.write(d)
            del data
            gc.collect()
            data = []
    print("Writing...")
    with jsonlines.open(save_jsonl_path, mode='a') as writer:
        for d in tqdm(data, desc="Write"):
            writer.write(d)
    data = []
    with jsonlines.open(save_jsonl_path, mode='r') as reader:
        for line in reader:
            data.append(line)

    print("Sorting...")
    data.sort(key=lambda d:d['nframe']/d['sr'])

    print("Writing...")
    with jsonlines.open(real_path, mode='w') as writer:
        for d in tqdm(data, desc="Write"):
            writer.write(d)
    print("Index Done!")

def save_data(i, save_dir, audios, a, noise_name, desc):
    (mixn_audio, audio1, audio2) = audios
    (a1,a2) = a
    torchaudio.save(save_dir+os.sep+'mixn'+os.sep+str(i)+'.wav', mixn_audio.cpu(), desc['sr'])
    torchaudio.save(save_dir+os.sep+'spk1'+os.sep+str(i)+'.wav', audio1.cpu(), desc['sr'])
    torchaudio.save(save_dir+os.sep+'spk2'+os.sep+str(i)+'.wav', audio2.cpu(), desc['sr'])
    with jsonlines.open(save_dir+os.sep+'metadata.jsonl', mode='a') as writer:
        writer.write({
            'audio1':a1['speaker']+'/'+a1['file'],
            'audio2':a2['speaker']+'/'+a2['file'],
            'noise':noise_name,
            'desc':desc
        })


def make_dataset(size, save_dir, index_jsonl_path, parent_dir, noise_dir):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    for subdir in ['spk1','spk2','mixn']:
        if not os.path.exists(save_dir+os.sep+subdir):
            os.mkdir(save_dir+os.sep+subdir)

    data = []
    with jsonlines.open(index_jsonl_path, mode='r') as reader:
        for line in reader:
            data.append(line)
    data_len = len(data)

    noise_paths = []
    for root, dirs, files in os.walk(noise_dir):
        for f in files:
            if f[-4:] in ['.wav','.mp3','flac','.ogg']:
                noise_paths.append(root + os.sep + f)
    data_len_range = range(data_len)
    audio1_idxs = [random.choice(data_len_range) for _ in range(size)]
    delta = data_len // 5

    record = set()

    for i, audio1_idx in enumerate(tqdm(audio1_idxs)):
        low = max(0, audio1_idx-delta)
        high = min(data_len, audio1_idx+delta)
        audio2_idx = random.choice(range(low, high))
        #注意取样的两个说话人不能相同
        cnt = 0
        while data[audio1_idx]['speaker'] == data[audio2_idx]['speaker'] \
            or (data[audio1_idx]['file'], data[audio2_idx]['file']) in record:
            low = int(low * 0.9)
            high = min(data_len, int(high * 1.1))
            audio2_idx = random.choice(range(low, high))
            cnt += 1
            if cnt > 100:
                break
        if cnt > 100:
            continue
        record.add((data[audio1_idx]['file'], data[audio2_idx]['file']))
        a1 = data[audio1_idx]
        a2 = data[audio2_idx]
        secs = max(a1['nframe']/a1['sr'], a2['nframe']/a2['sr'])
        noise_path = random.choice(noise_paths)
        mixn_audio, audio1, audio2, desc= load_and_mix_audio(
            audio1_path= parent_dir + os.sep + a1['speaker'] + os.sep + a1['file'],
            audio2_path= parent_dir + os.sep + a2['speaker'] + os.sep + a2['file'],
            noise_path= noise_path,
            secs= secs,
            mix_ratio_lh=(0.3,0.7),
            noise_ratio_lh=(0.01,0.10)
        )

        save_data(i, save_dir, (mixn_audio, audio1, audio2), (a1,a2), os.path.basename(noise_path), desc)
        del mixn_audio, audio1, audio2
        gc.collect()



# if __name__ == '__main__':
#     mixn_audio, mixn_sr = load_and_mix_audio(
#         "test/data/SSB00050353.wav", 
#         "test/data/SSB04340059.wav", 
#         "test/noise/babble-5s.wav"
#         )
#     torchaudio.save("test/mixn/mix2n_1.wav", mixn_audio.cpu(), mixn_sr)
