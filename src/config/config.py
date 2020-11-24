import os 
from shutil import copytree
from time import strftime, localtime

from src.trainer.trainer import Trainer
from src.model.GptForResponseGernation import GptForResponseGernation
from src.dataset.dataset import CachedPersonaDataset
from src.dataset.dataset_utils.persona import Batcher, Batcher_gen
from src.utils.logger import Logger


class Config():
    def __init__(self) -> None:
        self.trainer = Trainer
        self.model = GptForResponseGernation 
        self.mode = 'run_all'
        
        self.infer_times = 1 
        self.save_params = True
        self.preprocess_only = False
        self.save_params = self.save_params and (not self.preprocess_only) and self.mode != 'run_test' and self.mode != 'run_val'

        base_dataset_config = {
            'file_path': 'resource/data/personachat/train_self_original.txt',
            'debug': False,
            'debug_num': 1000,
            'use_cache': True
        }
        
        self.train_dataset_config = base_dataset_config.copy()
        self.train_dataset_config['file_path'] = 'resource/data/personachat/processed/processed_train_self_original.jsonl'

        
        self.val_dataset_config = base_dataset_config.copy()
        self.val_dataset_config['file_path'] = 'resource/data/personachat/processed/processed_valid_self_original.jsonl'

        
        self.test_dataset_config = base_dataset_config.copy()
        self.test_dataset_config['file_path'] = 'resource/data/personachat/processed/processed_test_self_original.jsonl'

        
        self.dataset = CachedPersonaDataset  
        
        self.model_config = {
            'name': 'gpt2',
            'cache_dir' : 'resource/pretrained_models/gpt2',
            'hidden_size': 768,
            'hidden_dropout_prob': 0.1,
            'vocab_size': 50268 # 50267 + 1
        }

         
        self.sample_train_data = False
        self.shuffle_on_the_fly = True

        self.use_cuda = True # GPU
        self.use_multi_gpus = True # 单机多卡
        
        # generation
        self.beam_size = 2
        self.do_sample = False
        self.num_return_sequences = 2

        self.num_epoch = 15
        self.batch_size = 20 # 10
        
        # 多卡的情况下batch_size会均分给每张卡
        # batch_size is the total batch_size when use_multi_gpus is set as True
        # 多卡情况下须用偶数, 否则会报错。且需要drop掉最后一个batch(如果最后一个batch的样本数为奇数)
        
        self.lr = 2e-5
        self.use_scheduler = False
        
        # self.l2_reg = None 
        # self.weight_decay = 1e-5
        self.init_clip_max_norm = 1.0
        
        # print and save
        self.print_every = 10
        self.val_every = 500
        self.force_save_every = None
        self.val_num = None


        run_name = None
        run_index = 1
        if run_name is None:
            run_name = 'run{}'.format(run_index)
        self.save_dir = 'resource/records/personachat/{}'.format(run_name)
        self.save_dir = os.path.abspath(self.save_dir)

        self.check_save_dirs()
        
        self.save_src_and_dst = ('src/', os.path.join(self.save_dir, 'src'))
             
        self.log_dir = os.path.join(self.save_dir,'log') 
        self.ckpt_dir = os.path.join(self.save_dir,'checkpoint')
        self.model_save_name = 'model_param.pt'
        self.ckpt_file = os.path.join(self.ckpt_dir, self.model_save_name)
        self.save_info_log_name = 'model_save_info.log'
        self.ckpt_info_log_path = os.path.join(self.ckpt_dir, self.save_info_log_name)
        

        current_time = strftime("%y%m%d-%H%M", localtime())
        self.log_file = '{}.log'.format(current_time)
        self.log_file = os.path.join(self.log_dir, self.log_file)   
        
        self.generated_results_save_path = os.path.join(self.log_dir, current_time + '_generated_results.json')  
        self.tensorboard_dir = os.path.join(self.log_dir,'tensorboard')      
        
        # check dirs
        self.check_dirs()

        logger = Logger(self.log_file)
        self.logger = logger.get_logger()
        
        
        
        self.batcher_config = {  # 处理数据
            'name': 'gpt2',
            'cache_dir': 'resource/pretrained_models/gpt2',
            'block_size': 256,
            'logger': self.logger,
            'cuda': self.use_cuda
        }
        

        self.collect_fn = Batcher(self.batcher_config)
        self.test_collect_fn = Batcher_gen(self.batcher_config)
        

        if self.save_params:
            self.save_all_program_files()
        

    def check_save_dirs(self):
        if self.mode == 'run_all' or self.mode == 'run_train':
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
            else:
                print('{} was used'.format(self.save_dir))
                save_parent_dir = os.path.split(self.save_dir[:-1])[0]
                paths = [p for p in os.listdir(save_parent_dir) if p[:3] == 'run']
                max_num = 1
                for p in paths:
                    num_suffix = p[3:]
                    if num_suffix:
                        num = int(num_suffix)
                    else:
                        num = 1
                    max_num = max(max_num, num)
                new_num_suffix = str(max_num + 1)
                self.save_dir = os.path.join(save_parent_dir, 'run' + new_num_suffix)
                print('now use {}'.format(self.save_dir))
                os.makedirs(self.save_dir)
        else:
            if not os.path.exists(self.save_dir):
                raise Exception('{} does not exist'.format(self.save_dir))

    def check_dirs(self):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            
        if self.preprocess_only or self.mode == 'run_test':
            return

        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

    def check_params(self):
        assert type(self.infer_times) == int
        assert self.infer_times >= 1 and self.infer_times <= 10

    def save_all_program_files(self):
        src, dst = self.save_src_and_dst
        copytree(src, dst)

