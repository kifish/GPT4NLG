import torch, time, os
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F 
from torch import optim
from tqdm import tqdm
import time 
from src.utils.metrics import Metrics

from src.utils.utils import cal_elapsed_time

from transformers import GPT2Tokenizer

import json, math 
from datetime import datetime


# from apex import amp
# 混合精度加速
# TODO



class Trainer(object):
    def __init__(self, config):
                
        # data
        if config.mode == 'run_all' or config.mode == 'run_train':
            self.trainset = config.dataset(**config.train_dataset_config)
            self.valset = config.dataset(**config.val_dataset_config)
            self.testset = config.dataset(**config.test_dataset_config)
            
        elif config.mode == 'run_val':
            self.valset = config.dataset(**config.val_dataset_config)
            
        elif config.mode == 'run_test' or config.mode == 'run_predict':
            self.testset = config.dataset(**config.test_dataset_config)

        
        tokenizer = GPT2Tokenizer.from_pretrained(config.model_config['name'], cache_dir = config.model_config['cache_dir'])        
        SPECIAL_TOKENS_DICT = {'additional_special_tokens': ["<respond>"]}
        tokenizer.add_special_tokens(SPECIAL_TOKENS_DICT)
        self.model = config.model(tokenizer, config.model_config, use_token_type_embedding = False)
        
        
        self.device = torch.device('cuda' if config.use_cuda else 'cpu')
        if config.use_cuda:
            config.logger.info('num of gpus : {}'.format(torch.cuda.device_count()))
            if config.use_multi_gpus:
                self.model = nn.DataParallel(self.model).to(self.device)
                config.logger.info('names of gpus : {}'.format(torch.cuda.get_device_name()))

            else:                
                self.model = self.model.cuda()
                config.logger.info('name of gpus : {}'.format(torch.cuda.get_device_name()))
        # DDP TODO 
        
        self.config = config
        
        if config.mode == 'run_all':
            self.writer = SummaryWriter(self.config.tensorboard_dir)


        self.optimizer = optim.Adam(self.model.parameters(), lr = self.config.lr) 
        
        
        self.criterion = nn.CrossEntropyLoss(ignore_index= -1, reduction = 'mean') # mean token loss
        self.seq_criterion = nn.CrossEntropyLoss(ignore_index= -1, reduction = 'sum') # sum; 方便后续算ppl; 之后除以句子个数

        
        # self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, min_lr=1e-7,
        #                                                  patience=2, verbose=True, threshold=0.0001, eps=1e-8)
        
        # self.optimizer = optim.AdamW(self.model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        # self.optimizer = optim.SGD(self.model.parameters(), lr=self.config.lr, momentum=0.9, dampening=0, weight_decay=self.config.l2_reg, nesterov=False)
        

        self.config.logger.info(self.model)


    def save_checkpoint(self, save_info, 
                        ckpt_file = None, 
                        ckpt_info_log_path = None
                        ):
        # DataParallel wrappers keep raw model object in .module attribute
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        
        # torch.save(raw_model.state_dict(), save_path)
        
        # torch.save({'model_state_dict': raw_model.state_dict(), 
        #             'optimizer_state_dict': self.optimizer.state_dict(),
        #             }, save_path)
        
        
        # save disk space
        if ckpt_file is None:
            ckpt_file = self.config.ckpt_file
            
        torch.save({'model_state_dict': raw_model.state_dict(), 
                    }, ckpt_file)  

        self.config.logger.info('saved model into {} \n'.format(ckpt_file))
        
        # log
        if ckpt_info_log_path is None:
            ckpt_info_log_path = self.config.ckpt_info_log_path
            
        with open(ckpt_info_log_path,'w') as f:
            f.write(save_info + '\n')
            
            
    def load_checkpoint(self):
        ckpt = torch.load(self.config.ckpt_file)
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        raw_model.load_state_dict(ckpt['model_state_dict'])
        self.config.logger.info('loaded the trained model')

    

    
    def train(self, train_data_loader, val_data_loader):
        best_result = {'loss': None, 'ppl': None}
        global_step = 0


        def better(pre_best_result, cur_result):
            
            if pre_best_result['loss'] is None or cur_result['loss'] < pre_best_result['loss']:
                return True
            else:
                return False 
        
        def gen_save_info(epoch_idx, val_result):
            save_info = 'save info : epoch_{}_val_loss_{:.6f}_val_ppl_{:.4f}'. \
                            format(epoch_idx + 1, val_result['loss'], val_result['ppl'])
            return save_info 
        
        
        
        self.config.logger.info('Lr: {}'.format(self.optimizer.param_groups[0]['lr']))
        self.writer.add_scalar('Lr', self.optimizer.param_groups[0]['lr'], global_step)
        
        for epoch in tqdm(range(self.config.num_epoch)):
            self.config.logger.info('>' * 100)
            self.config.logger.info('epoch: {}'.format(epoch + 1))
            
            n_step, n_sample_total, loss_total = 0, 0, 0 # in the epoch
            seq_loss_total = 0
            
            print_cnt = 0
            
            start_time = time.time()

            # switch model to training mode
            self.model.train()

            for batch_idx, batch_samples in enumerate(tqdm(train_data_loader)):
                global_step += 1
                n_step += 1
                
                
                # batchfy
                input_ids, target_ids, _, _ = batch_samples # GPT自动mask padding
                n_sample_total += input_ids.size(0) # 实际的样本个数

                
                ############# train  model #####################
                self.optimizer.zero_grad()
                outputs = self.model(input_ids) 
                pred_scores = outputs[0] # logits
                
                # pred_scores: (b,seq_len,V)
                # target_ids: (b,seq_len)
                
                pred_scores = pred_scores.view(-1, pred_scores.size(-1))
                target_ids = target_ids.view(pred_scores.size(0))
                
                loss = self.criterion(pred_scores, target_ids) # 做了mask; mean token loss 参考了PreTrain4KGC; backward 不要用 mean seq loss
                # https://pytorch.org/docs/stable/nn.functional.html#cross-entropy
                # ignore_index (int, optional) – Specifies a target value that is ignored and does not contribute to the input gradient. When size_average is True, the loss is averaged over non-ignored targets. Default: -100
                
                loss.backward()

                
                if self.config.init_clip_max_norm is not None:                    
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad], \
                            max_norm = self.config.init_clip_max_norm)
                    if grad_norm >= 1e2:
                        self.config.logger.info('WARNING: Exploding Gradients {:.2f}'.format(grad_norm))
            
                self.optimizer.step()
                ####################################################
                
                loss_total += loss.item() 
                train_loss = loss_total / n_step # train_mean_token_loss in the epoch
                
                self.writer.add_scalar('Train/Loss', train_loss , global_step)

                if global_step % self.config.print_every == 0:
                    print_cnt += 1 # 这里为了代码的简洁重复计算了loss, 追求高效的话可以手动计算每个batch实际的需要预测的token数量和实际的句子个数
                    seq_loss_total += self.seq_criterion(pred_scores, target_ids).item() / input_ids.size(0)
                    train_seq_loss = seq_loss_total / print_cnt
                    
                    self.config.logger.info('epoch {}, iteration {}, '
                        'train_mean_token_loss: {:.4f}, '
                        'train_mean_seq_loss: {:.4f}'.format(epoch + 1, batch_idx + 1, train_loss, train_seq_loss))
                    
                    
                # val
                if global_step % self.config.val_every == 0 or \
                        (self.config.force_save_every is not None and
                            global_step % self.config.force_save_every == 0):

                    val_result = self.inference(val_data_loader, mode = 'val')
                    self.writer.add_scalar('Val/loss', val_result['loss'] , global_step) # mean token loss
                    self.writer.add_scalar('Val/seq_loss', val_result['seq_loss'] , global_step) # mean seq loss
                    self.writer.add_scalar('Val/ppl', val_result['ppl'] , global_step)

                    # remember
                    self.model.train()
                    
                    # adjust
                    if self.config.use_scheduler:
                        self.scheduler.step()
                        self.config.logger.info('Lr: {}'.format(self.optimizer.param_groups[0]['lr']))
                        self.writer.add_scalar('Lr', self.optimizer.param_groups[0]['lr'], global_step)
                        

                    # save_best_only
                    if better(best_result, val_result):
                        best_result = val_result
                        save_info = gen_save_info(epoch, val_result)
                        self.save_checkpoint(save_info)
                        

                    elif (self.config.force_save_every is not None and
                            global_step % self.config.force_save_every == 0):
                        
                        
                        save_info = gen_save_info(epoch, val_result)

                        num = global_step // self.config.force_save_every
                        
                        ckpt_info_log_path = os.path.join(self.config.ckpt_dir,'force_{}_{}'.format(
                            num,self.config.save_info_log_name
                        ))
                            
                        ckpt_force_file = os.path.join(self.config.ckpt_dir,'force_{}_{}'.format(
                            num, self.config.model_save_name
                        ))
                        self.config.logger.info('force save ...')
                        self.save_checkpoint(save_info, ckpt_force_file, ckpt_info_log_path)



            # # adjust
            # if self.config.use_scheduler:
            #     self.scheduler.step()
            #     # self.scheduler.step(r1)
                
            train_loss = loss_total / n_step # 一个epoch里的平均loss
            train_seq_loss = 0 # 0表示未计算
            if print_cnt > 0:
                train_seq_loss = seq_loss_total / print_cnt
                
            end_time = time.time()
            epoch_mins, epoch_secs = cal_elapsed_time(start_time, end_time)
            self.config.logger.info('epoch: {}, train_mean_token_loss: {:.4f}, '
                'train_mean_seq_loss: {:.4f}, time: {}m {}s'. \
                    format(epoch + 1, train_loss, train_seq_loss, epoch_mins, epoch_secs))
            


    def inference(self, data_loader, mode = 'val'):
        n_samples, loss_total = 0, 0
        seq_loss_sum = 0
        n_batch = 0
        self.model.eval()

        self.config.logger.info("{} ing ...".format(mode))
        
        with torch.no_grad():
            for batch_idx, batch_samples in enumerate(tqdm(data_loader)):
                n_batch += 1
                
                input_ids, target_ids, _, _ = batch_samples

                
                the_real_batch_size = input_ids.size(0)
                n_samples += the_real_batch_size
                
                mask = target_ids != -1
                the_real_num_target_id = mask.sum().item()
                
                outputs = self.model(input_ids) 
                pred_scores = outputs[0]
                
                
                # reshape
                pred_scores = pred_scores.view(-1, pred_scores.size(-1))
                target_ids = target_ids.view(pred_scores.size(0))
                
                loss = self.criterion(pred_scores, target_ids)
                loss_total += loss.item()
                
                seq_loss_sum += loss.item() * the_real_num_target_id
                
                if self.config.val_num is not None and batch_idx + 1 == self.config.val_num:
                    break
                
        
        mean_token_loss = loss_total / n_batch
        mean_seq_loss = seq_loss_sum / n_samples

        
        result = {
            'loss' : mean_token_loss,
            'seq_loss': mean_seq_loss,
            'ppl' : math.exp(mean_seq_loss) if mean_seq_loss < 300 else float('inf'),
            'token_ppl' : math.exp(mean_token_loss) if mean_token_loss < 300 else float('inf'),
        }
                
        self.config.logger.info("the whole/part of {} dataset:".format(mode))
        self.config.logger.info("n_samples : {}".format(n_samples)) 
        self.config.logger.info('{} loss: {:.4f}'.format(mode, result['loss'])) 
        self.config.logger.info('{} ppl {:.4f}'.format(mode, result['ppl']))
        self.config.logger.info('{} token_ppl {:.4f}'.format(mode, result['token_ppl']))
        return result




    def eval(self, data_loader):
        # TODO
        pass
      

    def predict(self, data_loader):
        # TODO 
        pass
    

    
    def run(self,mode = 'run_all'):
        self.config.logger.info("mode : {}".format(mode))
        
        if mode == 'run_all': 
            need_drop = False
            if self.config.use_cuda and self.config.use_multi_gpus:
                need_drop = True

            if self.config.sample_train_data:
                raise NotImplementedError
            
            else:
                shuffle = False
                if self.config.shuffle_on_the_fly:
                    shuffle = True
                    self.config.logger.info("shuffle_on_the_fly")
                
                train_data_loader = DataLoader(dataset = self.trainset, batch_size = self.config.batch_size, \
                    shuffle = shuffle, collate_fn=self.config.collect_fn, drop_last = need_drop) 
                # https://github.com/pytorch/pytorch/issues/42654
                # https://pytorch.org/docs/stable/data.html#working-with-collate-fn

            val_data_loader = DataLoader(dataset = self.valset, \
                batch_size = self.config.batch_size * self.config.infer_times, \
                shuffle = False, collate_fn=self.config.collect_fn, drop_last = need_drop)
            
            test_data_loader = DataLoader(dataset = self.testset, \
                batch_size = self.config.batch_size * self.config.infer_times, \
                    shuffle = False, collate_fn=self.config.collect_fn, drop_last = need_drop)


            gen_test_data_loader = DataLoader(dataset = self.testset, \
                batch_size = 1, \
                    shuffle = False, collate_fn= self.config.test_collect_fn, drop_last = need_drop)
            

            self.train(train_data_loader, val_data_loader)
            self.load_checkpoint()
            self.inference(test_data_loader, mode = 'test')
            # self.eval(gen_test_data_loader)
            


                        
        elif mode == 'run_val':
            val_data_loader = DataLoader(dataset = self.valset, \
                batch_size = self.config.batch_size * self.config.infer_times, \
                 shuffle=False, collate_fn=self.config.collect_fn)
            
            self.load_checkpoint()
            self.inference(val_data_loader, mode = 'val')
            
        elif mode == 'run_test':
            test_data_loader = DataLoader(dataset = self.testset, \
                batch_size = self.config.batch_size * self.config.infer_times, \
                 shuffle=False, collate_fn= self.config.collect_fn)

            gen_test_data_loader = DataLoader(dataset = self.testset, \
                batch_size = 1, \
                    shuffle = False, collate_fn= self.config.test_collect_fn)
            
            self.load_checkpoint()
            self.inference(test_data_loader, mode = 'test')
            self.eval(gen_test_data_loader)

                        
                        
        elif mode == 'run_predict':
            gen_test_data_loader = DataLoader(dataset = self.testset, \
                batch_size = 1, \
                    shuffle = False, collate_fn= self.config.test_collect_fn)
            self.load_checkpoint()
            # self.predict(gen_test_data_loader) # TODO
            
            

