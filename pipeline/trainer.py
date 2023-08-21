import sys
import os
sys.path.append(os.path.dirname(__file__))
import torch
from loguru import logger
from shutil import copyfile
import time
from util import Loss,collate_fn
from torch.utils.data import DataLoader
from data import MixnAudioDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

class Trainer:
    def __init__(self, model, opt) -> None:
        self.train_data_dir = opt['datasets']['train_data_dir']
        self.test_data_dir = opt['datasets']['test_data_dir']
        self.save_stat_dir = opt['train']['save_stat_dir']
        self.save_model_dir = opt['train']['save_model_dir']
        self.device = opt['device']
        self.model = model.to(self.device) 
        self.gpuid = opt['gpuid']

        self.batch_size = opt['train']['batch_size']
        self.sub_batch = opt['train']['sub_batch']

        self.train_dataset = MixnAudioDataset(self.train_data_dir, preload=False, sub_batch=self.sub_batch)

        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=collate_fn)
        self.val_dataloader = DataLoader(
            MixnAudioDataset(self.test_data_dir, preload=False),
            batch_size=self.batch_size, shuffle=False, collate_fn=collate_fn
        )
        self.optimizer = self.make_optimizer(self.model.parameters(), opt)
        self.scheduler = self.make_scheduler(self.optimizer, opt)

        self.cur_epoch = 0
        self.print_freq = 10
        self.total_epoch = opt['train']['epoch']
        self.clip_norm = opt['optim']['clip_norm']
        self.early_stop = opt['train']['early_stop']
        if opt['resume']['state']:
            ckp = torch.load(os.path.join(
                opt['resume']['path'], 'best.pt'), map_location=self.device)
            self.cur_epoch = ckp['epoch']
            logger.info("Resume from checkpoint {}: epoch {:.3f}".format(
                opt['resume']['path'], self.cur_epoch))
            model.load_state_dict(
                ckp['model'])
            self.model = model.to(self.device)
            self.optimizer.load_state_dict(ckp['optim']) 
            lr = self.optimizer.param_groups[0]['lr']
            self.adjust_learning_rate(self.optimizer, lr*0.5)

        with open(self.save_stat_dir + os.sep + 'train_stat.csv', 'w') as f:
            f.write('epoch,iter,loss\n')
        with open(self.save_stat_dir + os.sep + 'test_stat.csv', 'w') as f:
            f.write('epoch,iter,loss\n')
    
    def adjust_learning_rate(self, optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    
    @staticmethod
    def make_optimizer(params, opt):
        optimizer = getattr(torch.optim, opt['optim']['name'])
        if opt['optim']['name'] == 'Adam':
            optimizer = optimizer(
                params, lr=opt['optim']['lr'], weight_decay=opt['optim']['weight_decay'])
        else:
            optimizer = optimizer(params, lr=opt['optim']['lr'], weight_decay=opt['optim']
                                ['weight_decay'], momentum=opt['optim']['momentum'])
        return optimizer

    @staticmethod
    def make_scheduler(optimizer, opt):
        return ReduceLROnPlateau(
            optimizer, mode='min',
            factor=opt['scheduler']['factor'],
            patience=opt['scheduler']['patience'],
            verbose=True, min_lr=opt['scheduler']['min_lr'])

    def train(self, epoch):
        logger.info(
            'Start training from epoch: {:d}, iter: {:d}'.format(epoch, 0))
        self.model.train()
        num_batchs = len(self.train_dataloader)
        total_loss = 0.0
        num_index = 1
        start_time = time.time()
        for mixn, spk1, spk2, sr in self.train_dataloader:
            mixn = mixn.to(self.device)
            spk1 = spk1.to(self.device)
            spk2 = spk2.to(self.device)
            self.optimizer.zero_grad()
            if self.gpuid:
                out = torch.nn.parallel.data_parallel(self.model,mixn,device_ids=self.gpuid)
            else:
                out = self.model(mixn)
            l = Loss(out, (spk1, spk2))
            epoch_loss = l
            total_loss += epoch_loss.item()
            epoch_loss.backward()

            if self.clip_norm:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.clip_norm) #也用了这个方式防止梯度爆炸

            self.optimizer.step()
            if num_index % self.print_freq == 0:
                message = '<epoch:{:d}, iter:{:d}, lr:{:.3e}, loss:{:.3f}>'.format(
                    epoch, num_index, self.optimizer.param_groups[0]['lr'], total_loss/num_index)
                logger.info(message)
            num_index += 1
        end_time = time.time()
        total_loss = total_loss/num_index
        message = 'Finished *** <epoch:{:d}, iter:{:d}, lr:{:.3e}, loss:{:.3f}, Total time:{:.3f} min> '.format(
            epoch, num_index, self.optimizer.param_groups[0]['lr'], total_loss, (end_time-start_time)/60)
        logger.info(message)
        with open(self.save_stat_dir + os.sep + 'train_stat.csv', 'a') as f:
            f.write(f'{epoch},{num_index},{total_loss}\n')
        return total_loss

    def validation(self, epoch):
        logger.info(
            'Start Validation from epoch: {:d}, iter: {:d}'.format(epoch, 0))
        self.model.eval()
        num_batchs = len(self.val_dataloader)
        num_index = 1
        total_loss = 0.0
        start_time = time.time()
        with torch.no_grad():
            for mixn, spk1, spk2, sr in self.val_dataloader:
                mixn = mixn.to(self.device)
                spk1 = spk1.to(self.device)
                spk2 = spk2.to(self.device)
                self.optimizer.zero_grad()
                if self.gpuid:
                    out = torch.nn.parallel.data_parallel(self.model,mixn,device_ids=self.gpuid)
                else:
                    out = self.model(mixn)
                l = Loss(out, (spk1, spk2))
                epoch_loss = l
                total_loss += epoch_loss.item()
                if num_index % self.print_freq == 0:
                    message = '<epoch:{:d}, iter:{:d}, lr:{:.3e}, loss:{:.3f}>'.format(
                        epoch, num_index, self.optimizer.param_groups[0]['lr'], total_loss/num_index)
                    logger.info(message)
                del mixn, spk1, spk2
                num_index += 1
        end_time = time.time()
        total_loss = total_loss/num_index
        message = 'Finished *** <epoch:{:d}, iter:{:d}, lr:{:.3e}, loss:{:.3f}, Total time:{:.3f} min> '.format(
            epoch, num_index, self.optimizer.param_groups[0]['lr'], total_loss, (end_time-start_time)/60)
        logger.info(message)
        with open(self.save_stat_dir + os.sep + 'test_stat.csv', 'a') as f:
            f.write(f'{epoch},{num_index},{total_loss}\n')
        return total_loss


    def run(self):
        train_loss = []
        val_loss = []
        self.save_checkpoint(self.cur_epoch, best=False)
        # v_loss = self.validation(self.cur_epoch)
        # best_loss = v_loss
        
        # logger.info("Starting epoch from {:d}, loss = {:.4f}".format(
            # self.cur_epoch, best_loss))
        best_loss = 1e8
        
        no_improve = 0

        # starting training part
        while self.cur_epoch < self.total_epoch:
            self.cur_epoch += 1
            t_loss = self.train(self.cur_epoch)
            self.train_dataset.next_epoch()
            v_loss = self.validation(self.cur_epoch)

            train_loss.append(t_loss)
            val_loss.append(v_loss)

            # schedule here
            self.scheduler.step(v_loss)

            if v_loss >= best_loss:
                no_improve += 1
                logger.info(
                    'No improvement, Best Loss: {:.4f}'.format(best_loss))
            else:
                best_loss = v_loss
                no_improve = 0
                self.save_checkpoint(self.cur_epoch, best=True)
                logger.info('Epoch: {:d}, Now Best Loss Change: {:.4f}'.format(
                    self.cur_epoch, best_loss))

            if no_improve == self.early_stop:
                logger.info(
                    "Stop training cause no improve for {:d} epochs".format(
                        no_improve))
                break
        self.save_checkpoint(self.cur_epoch, best=False)
        logger.info("Training {:d}/{:d} epoches done!".format(
            self.cur_epoch, self.total_epoch))

    
    def save_checkpoint(self, epoch, best=True):
        '''
           save model
           best: the best model
        '''
        os.makedirs(self.save_model_dir, exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model': self.model.state_dict(),
            'optim': self.optimizer.state_dict()
            }, os.path.join(self.save_model_dir, f'epoch_{epoch}.pt'))
        if best:
            copyfile(os.path.join(self.save_model_dir, f'epoch_{epoch}.pt'),
                     os.path.join(self.save_model_dir, f'best.pt'))
    
