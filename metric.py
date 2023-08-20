import torch
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
import torchaudio

si_snr1, si_snr2 = 0,0
si_snr = ScaleInvariantSignalDistortionRatio()
siz = 1
for i in range(siz):
    audio_pred, sr1 = torchaudio.load(f'speech-preprocess/data/Data_100_Result/spk1_est/{i}.wav')
    audio_pure, sr2 = torchaudio.load(f'speech-preprocess/data/Data_100/spk1/{i}.wav')
    si_snr1 += si_snr(audio_pred, audio_pure)

    audio_pred, sr1 = torchaudio.load(f'speech-preprocess/data/Data_100_Result/spk2_est/{i}.wav')
    audio_pure, sr2 = torchaudio.load(f'speech-pbreprocess/data/Data_100/spk2/{i}.wav')
    si_snr2 += si_snr(audio_pred, audio_pure)

si_snr1 /= siz
si_snr2 /= siz

print(si_snr1, si_snr2)

# pesq = PerceptualEvaluationSpeechQuality(sr1, mode='wb')
# print(pesq(audio_pred, audio_pure))