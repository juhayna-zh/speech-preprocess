import sys
import os
sys.path.append(os.path.dirname(__file__))
import torch
from loguru import logger
from torch.utils.data import DataLoader
from data import TestMixnAudioDataset, get_filenames
import threading
import torchaudio
from torchmetrics.audio.snr import SignalNoiseRatio, ScaleInvariantSignalNoiseRatio
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility
import torchaudio
from torchaudio.functional import resample
from collections import defaultdict

class Tester:
    def __init__(self, make_model, opt, train_opt) -> None:
        self.make_model = make_model
        self.data_path = opt['test']['data_path'] #数据加载目录
        self.save_path = opt['test']['save_path'] #音频结果保存目录
        self.num_threads = opt['test']['num_threads'] #线程数
        self.no_criterion = opt['test']['no_criterion'] #是否仅预测,不评估指标
        self.measure_dataset_only = opt['test']['measure_dataset_only'] #是否仅评估数据集,不预测

        if self.measure_dataset_only:
            #如果仅评估数据集,则结果(csv文件)保存目录改为数据所在目录
            self.save_path = self.data_path

        self.make_dirs()
        self.ckpt_path =opt['model']['checkpoint']
        self.train_opt = train_opt  
        self.opt = opt
        self.sr = train_opt['model']['sr']

        self.filenames = get_filenames(self.data_path)

        if opt['gpuid'] is None or len(opt['gpuid']) == 0:
            self.devices = [opt['device']]
        else: 
            self.devices = ['cuda:'+str(i) for i in opt['gpuid']]

        self.gpuid = opt['gpuid'] #可供使用的GPU序号
      
        self.result = [[] for _ in range(self.num_threads)] #每个线程结束前将指标结果保存在此
            
    def metric(self, audio_ests, audio_refs, sr, *, opr):
        snr, sisnr, pesq_wb, pesq_nb, stoi = opr #本线程的评估指标

        spk1_est, spk2_est = audio_ests
        spk1, spk2 = audio_refs
        sisnr_swap = (sisnr(spk1_est, spk2).item() + sisnr(spk2_est, spk1).item()) / 2
        sisnr = (sisnr(spk1_est, spk1).item() + sisnr(spk2_est, spk2).item()) / 2
        #根据sisnr确认speaker的顺序
        if sisnr_swap > sisnr:
            sisnr = sisnr_swap
            spk1, spk2 = spk2, spk1
            
        snr = (snr(spk1_est, spk1).item() + snr(spk2_est, spk2).item()) / 2
        stoi = (stoi(spk1_est, spk1).item() + stoi(spk2_est, spk2).item()) / 2
        pesq_wb1, pesq_nb1 = self.calc_pesq(spk1_est, spk1, sr, opr=(pesq_wb,pesq_nb))
        pesq_wb2, pesq_nb2 = self.calc_pesq(spk2_est, spk2, sr, opr=(pesq_wb,pesq_nb))
        pesq_wb = (pesq_wb1 + pesq_wb2) / 2
        pesq_nb = (pesq_nb1 + pesq_nb2) / 2

        return {'snr':snr, 'sisnr':sisnr, 'pesq_wb':pesq_wb, 'pesq_nb':pesq_nb, 'stoi':stoi}
        
            
    def calc_pesq(self, audio_est, audio_ref, sr, opr):
        pesq_wb, pesq_nb = opr
        audio_ref_16k = resample(audio_ref, sr, 16e3)
        audio_est_16k = resample(audio_est, sr, 16e3)
        return pesq_wb(audio_est_16k, audio_ref_16k).item(), pesq_nb(audio_est_16k, audio_ref_16k).item()
    
    def load_checkpoint(self, model, device):
        ckp = torch.load(self.ckpt_path, map_location=device)
        try:
            model.load_state_dict(ckp['model'])
        except:
            model.load_state_dict(ckp)
    
    def save_spk1(self, *args):
        return os.path.join(self.save_path, 'spk1_est', *args)
    
    def save_spk2(self, *args):
        return os.path.join(self.save_path, 'spk2_est', *args)

    def make_dirs(self):
        if (not self.measure_dataset_only) and self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)
            os.makedirs(self.save_spk1(), exist_ok=True)
            os.makedirs(self.save_spk2(), exist_ok=True)
    
    def do_test_job(self, device, sub_id, gpuid=None):
        if not self.measure_dataset_only:
            model = self.make_model(self.train_opt).to(device)
            self.load_checkpoint(model, device)
            model.eval()
        logger.info(f'Thread[{sub_id}]: load model done, run on {device}, gpuid={gpuid}')
        metrics = defaultdict(lambda:0)
        cnt = 0
        dataloader = DataLoader(
            TestMixnAudioDataset(self.data_path, filenames=self.filenames, sr=self.train_opt['model']['sr'],
                                sub_id=sub_id, num_threads=self.num_threads, no_GT=self.opt['test']['no_criterion']),
                                batch_size=1, shuffle=False
        )    
        #初始化需要的指标
        snr = SignalNoiseRatio().to(device)
        sisnr = ScaleInvariantSignalNoiseRatio().to(device)
        pesq_wb = PerceptualEvaluationSpeechQuality(16e3, mode='wb').to(device)
        pesq_nb = PerceptualEvaluationSpeechQuality(16e3, mode='nb').to(device)
        stoi = ShortTimeObjectiveIntelligibility(self.sr).to(device)
        opr = (snr, sisnr, pesq_wb, pesq_nb, stoi) #打包,以供传参
        for mixn, spk1, spk2, sr, filename in dataloader:
            cnt += 1
            #首先,将所有的数据移入对应设备.仅当使用评估时才移动GT的数据
            mixn = mixn.to(device)
            if not self.no_criterion:
                spk1 = spk1.to(device)
                spk2 = spk2.to(device)
            #如果仅评估数据集,则相当于预测结果就是mixn本身
            if self.measure_dataset_only:
                spk1_est = mixn
                spk2_est = mixn
            elif gpuid is None:
                spk1_est, spk2_est = model(mixn)
            else:
                spk1_est, spk2_est = torch.nn.parallel.data_parallel(model,mixn,device_ids=gpuid)
            #保存预测的音频文件
            if (not self.measure_dataset_only) and self.save_path is not None:
                torchaudio.save(self.save_spk1(filename[0]), spk1_est.squeeze(0).cpu(), sr.item())
                torchaudio.save(self.save_spk2(filename[0]), spk2_est.squeeze(0).cpu(), sr.item())
            #计算并收集指标
            if not self.no_criterion:
                m = self.metric((spk1_est, spk2_est), (spk1, spk2), sr.item(), opr=opr)
                for k,v in m.items():
                    metrics[k] += v
            logger.info(f"Thread[{sub_id}]: finish test '{filename[0]}'.")
        self.result[sub_id].append(metrics)
        self.result[sub_id].append(cnt)
        del mixn, spk1, spk2, spk1_est, spk2_est

    def run(self):
        logger.info('Test on dataset:', self.data_path)
        d = 0
        self.threads = []
        gpu_dist = None
        #以下,为每个线程分配设备
        #如果设备数少于或等于线程数,则依次分配
        #如果GPU数量超过线程数,则启动data_parallel,为线程分配多个设备
        if len(self.gpuid) > self.num_threads:
            i = 0
            gpu_dist = [[] for _ in range(self.num_threads)]
            for gpu in self.gpuid:
                gpu_dist[i].append(gpu)
                i = (i+1) % self.num_threads
        #启动线程并为每个线程分入一个sub_id,标志其需要处理的数据的起始
        for sub_id in range(self.num_threads):
            t = threading.Thread(target=self.do_test_job, 
                    args=(self.devices[d], sub_id, None if gpu_dist is None else gpu_dist[sub_id]))
            t.start()
            self.threads.append(t)
            d = (d+1) % len(self.devices)
            logger.info(f"Created thread[{sub_id}] {t}")
        #等待所有线程结束
        for t in self.threads:
            t.join()
        #根据配置保存结果
        if not self.no_criterion:
            metrics = defaultdict(lambda:0)
            cnt = 0
            for i,r in enumerate(self.result):
                if len(r) > 0:
                    logger.info(f'Get thread[{i}].')
                    m,c = r
                    cnt += c
                    for k,v in m.items():
                        metrics[k] += v
            print("Result:")
            for k in metrics:
                metrics[k] /= cnt
                print(f'{k}: {metrics[k]}')
            with open(self.save_path+os.sep+'result.csv','w') as f:
                f.write('Metric,Value\n')
                for k in metrics:
                    f.write(f'{k},{metrics[k]}\n')

