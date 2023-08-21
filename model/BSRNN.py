import torch
import torch.nn as nn
import torchaudio
from math import floor

class RNNBlock(nn.Module):
    def __init__(self, in_channel:int, hid_channel:int):
        super().__init__()
        self.norm = nn.GroupNorm(1, in_channel)
        self.rnn = nn.LSTM(in_channel, hid_channel, 1, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hid_channel*2, in_channel)

    def forward(self, x):
        # N, Embd, Seq = x.shape
        y, h = self.rnn(self.norm(x).transpose(1,2).contiguous()) #对S(seq)维度用LSTM
        y = self.fc(y) #将E(Embd)投影回原有的维度
        y = y.transpose(1,2).contiguous() #重新组织维度顺序
        return y + x


class BandSplit(nn.Module):
    def __init__(self, feature_dim=80, *, bands ):
        super().__init__()
        self.bands = bands
        self.norm = nn.ModuleList()
        self.fc = nn.ModuleList()
        for band in self.bands:
            self.norm.append(nn.GroupNorm(1, band*2))
            self.fc.append(nn.Linear(band*2, feature_dim))

    def forward(self, x):
        accum_band = 0
        output = []
        for i,band in enumerate(self.bands):
            x_band = x[:,:,accum_band:accum_band+band,:]
            Batch, _, FreqSub, Seq = x_band.shape
            x_band = torch.reshape(x_band,[Batch, 2*FreqSub, Seq])
            out = self.norm[i](x_band)
            out = self.fc[i](out.transpose(1,2))
            output.append(out.transpose(1,2))
            accum_band = accum_band+band
        return torch.stack(output, dim=1).contiguous()


class MaskDecoder(nn.Module):
    def __init__(self, feature_dim=80, *, bands):
        super(MaskDecoder, self).__init__()
        self.bands = bands
        self.mask = [self._build_layer(band, feature_dim) for band in self.bands]
        self.mask = nn.ModuleList(self.mask)
        
    
    def _build_layer(self, band, feature_dim):
        '''搭建band的mask预测网络'''
        return nn.Sequential(nn.GroupNorm(1, feature_dim),
            nn.Conv1d(feature_dim, feature_dim*4, 1),
            nn.Tanh(),
            nn.Conv1d(feature_dim*4, feature_dim*4, 1),
            nn.Tanh(),
            nn.Conv1d(feature_dim*4, band*4, 1)
            )

    def get_subband_mix_spec(self, spec):
        band_idx = 0
        subband_mix_spec = []
        for band in self.bands: 
            subband_mix_spec.append(spec[:,band_idx:band_idx+band])  
            band_idx += band
        return subband_mix_spec

    def forward(self, sep, spec):
        BatchCh, Band, Feature, Seq = sep.shape
        sep_subband_spec = []
        
        spec = self.get_subband_mix_spec(spec)
        for i,band in enumerate(self.bands):
            this_output = self.mask[i](sep[:,i]).view(BatchCh, 2, 2, band, -1)
            this_mask = this_output[:,0] * torch.sigmoid(this_output[:,1])  # B*nch, 2, K, BW, T
            this_mask_real = this_mask[:,0]  # B*nch, K, BW, T
            this_mask_imag = this_mask[:,1]  # B*nch, K, BW, T
            est_spec_real = spec[i].real * this_mask_real - spec[i].imag * this_mask_imag  # B*nch, BW, T
            est_spec_imag = spec[i].real * this_mask_imag + spec[i].imag * this_mask_real  # B*nch, BW, T
            sep_subband_spec.append(torch.complex(est_spec_real, est_spec_imag))
        est_spec = torch.cat(sep_subband_spec, 1)  # B*nch, F, T
        return est_spec


class BSModel(nn.Module):
    def __init__(self, feature_dim=128):
        super().__init__()
        self.feature_dim = feature_dim
        self.band_rnn = RNNBlock(self.feature_dim, self.feature_dim*2)
        self.seq_rnn = RNNBlock(self.feature_dim, self.feature_dim*2)

    def forward(self, x):
        Batch, nBand, Feat, Seq = x.shape  
        #band rnn
        band_output = self.band_rnn(x.view(Batch*nBand, Feat, -1)).view(Batch, nBand, Feat, Seq)  
        #seq rnn
        band_output = band_output.permute(0,3,2,1).contiguous().view(Batch*Seq, -1, nBand) 
        output = self.seq_rnn(band_output).view(Batch, Seq, -1, nBand).permute(0,3,2,1).contiguous()
        return output.view(Batch, nBand, Feat, Seq)
    

class BSRNN(nn.Module):
    def __init__(self, sr=44100, win=2048, stride=512, feature_dim=80, num_layer=10, num_spk_layer=None, same_mask=False):
        super().__init__()
        self.win = win
        self.stride = stride
        self.group = self.win // 2
        self.enc_dim = self.win // 2 + 1
        self.feature_dim = feature_dim

        self.bands = self._build_bands(sr)
        
        self.band_split = BandSplit(feature_dim, bands=self.bands)
        self.bs_model = [BSModel(feature_dim) for _ in range(num_layer)]
        self.bs_model = nn.Sequential(*self.bs_model)
        self.mask_decoder1 = MaskDecoder(feature_dim, bands=self.bands)

        self.num_spk_layer = num_spk_layer
        if num_spk_layer:
            self.spk1_bs_model = [BSModel(feature_dim) for _ in range(num_spk_layer)]
            self.spk1_bs_model = nn.Sequential(*self.spk1_bs_model)
            self.spk2_bs_model = [BSModel(feature_dim) for _ in range(num_spk_layer)]
            self.spk2_bs_model = nn.Sequential(*self.spk2_bs_model)
        else: 
            self.spk1_bs_model = None
            self.spk2_bs_model = None

        if same_mask:
            self.mask_decoder2 = self.mask_decoder1
        else:
            self.mask_decoder2 = MaskDecoder(feature_dim, bands=self.bands)

    def _build_bands(self, sr):
        bandwidth_50 = int(floor(50 / (sr / 2.) * self.enc_dim))
        bandwidth_100 = int(floor(100 / (sr / 2.) * self.enc_dim))
        bandwidth_250 = int(floor(250 / (sr / 2.) * self.enc_dim))
        bandwidth_500 = int(floor(500 / (sr / 2.) * self.enc_dim))
        bandwidth_1k = int(floor(1000 / (sr / 2.) * self.enc_dim))
        bandwidth_2k = int(floor(2000 / (sr / 2.) * self.enc_dim))

        bands = [bandwidth_100]*10
        bands += [bandwidth_250]*12
        bands += [bandwidth_500]*8
        bands += [bandwidth_1k]*8
        bands += [bandwidth_2k]*2
        bands.append(self.enc_dim - sum(bands))

        return bands

    def forward(self, x):
        Batch, Channel, Times = x.shape 
        x = x.view(Batch*Channel, Times)
        window = torch.hann_window(self.win).to(x.device).type(x.type()) 

        spec = torch.stft(x, n_fft=self.win, hop_length=self.stride, 
                          window=window,return_complex=True) #Complex[Batch, Freq, Seq] 
        spec_ri = torch.stack([spec.real, spec.imag], dim=1) #[Batch, RI, Freq, Seq]
        
        subband_spec = self.band_split(spec_ri) #[BatchCh, Band, Feature, Seq]
        Batch, nBand, Feat, Seq = subband_spec.shape

        sep_output = self.bs_model(subband_spec) 

        if self.num_spk_layer:
            sep_output1 = self.spk1_bs_model(sep_output)
            sep_output2 = self.spk2_bs_model(sep_output)
        else: 
            sep_output1 = sep_output
            sep_output2 = sep_output

        est_spec1 = self.mask_decoder1(sep_output1, spec) #[BatchCh, Freq, Seq]
        output1 = torch.istft(est_spec1, n_fft=self.win, hop_length=self.stride, 
                             window=window, length=Times)
        output1 = output1.view(Batch, Channel, -1)

        est_spec2 = self.mask_decoder2(sep_output2, spec) #[BatchCh, Freq, Seq]
        output2 = torch.istft(est_spec2, n_fft=self.win, hop_length=self.stride, 
                             window=window, length=Times)
        output2 = output2.view(Batch, Channel, -1)

        return output1, output2



    
if __name__ == '__main__':
    from thop import profile, clever_format
    import numpy as np
    model = BSRNN(sr=44100, win=2048, stride=512, feature_dim=80, num_layer=6, num_spk_layer=4, same_mask=True)

    #参数量
    s = 0
    for param in model.parameters():
        s += np.product(param.size())
    print('# of parameters: '+str(s/1024.0/1024.0))
    
    #输入输出
    x = torch.randn((4, 1, 441*3))
    o1, o2 = model(x)
    print(o1.shape, o2.shape)

    #FLOPS计算量
    macs, params = profile(model, inputs=(x,))
    macs, params = clever_format([macs, params], "%.3f")
    print(macs, params)
