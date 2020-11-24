
import os,json
import random, string
from tqdm import tqdm 
from transformers import GPT2Tokenizer
import torch 
from src.utils.utils import timer_context
from multiprocessing import Pool
import multiprocessing as mp 
import mmap

from functools import wraps
import time

from contextlib import ContextDecorator


def timing(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # start = time.clock()
        start = time.time()
        r = func(*args, **kwargs)
        # end = time.clock()
        end = time.time()
        print('[' + func.__name__ + '] used: {} s'.format(end - start)) # wall time
        return r
    return wrapper

@timing
def get_num_lines(file_path):
    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    n_line = 0
    while buf.readline():
        n_line += 1
    return n_line


def save_examples_to_jsonl(examples, path):
    root_dir = os.path.dirname(path)
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
        
    print('saving data into {} ...'.format(path))
    with open(path, 'w', encoding = 'utf8') as f:
        for example in tqdm(examples):
            line = json.dumps(example)
            f.write(line)
            f.write('\n')
    print('done .')        
    


@timing
def read_cached_data_tqdm(file_path, debug = False, debug_num = 1000):
    examples = []
    print('reading cached data ...')
    with open(file_path, 'r', encoding='utf8') as f:
        for line in tqdm(f, total= get_num_lines(file_path)): # generator
            line = line.strip()
            example = json.loads(line)
            examples.append(example)
            del line 
            if debug and len(examples) >= debug_num:
                break
    print('num of cached examples : {}'.format(len(examples))) 
    return examples



def load_file_multiTurn(file_path, num_cands = 20, shuffle_level= 'examples', shuffle_cand = False, data_format = 'self'):
    """
    return examples, positions
    examples: list
    example: dict;
    {   'document': document,      # document:list
        'context': current_turn['context'], # current_turn['context']:list
        'response': c, # c:str
        'label': l
    } # l:int 0 or 1
    positions: [(start,end)];方便后续shuffle_level的控制
    """
    examples = []
    positions = []
    num_episode = 0
    num_example = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        document, episode = [], []
        start, end = 0, 0
        for i, line in enumerate(lines):
            episode_done = (i + 1 == len(lines) or lines[i + 1].startswith('1 ')) # 一个完整的多轮对话结束

            line = line.lstrip(string.digits).strip()
            if data_format == 'self':
                line = line.replace('your persona:', '').strip()
            else:
                line = line.replace("partner's persona:", '').strip()
            parts = line.split('\t')
            if len(parts) == 1: # 无\t分割的文本说明是document的句子
                document.append(parts[0])
            else:
                response = parts[1]
                query = parts[0]
                cands = parts[-1].split('|')
                if shuffle_cand:
                    random.shuffle(cands)
                    labels = [1 if cand == response else 0 for cand in cands]
                else:
                    labels = [1 if idx + 1 == len(cands) else 0 for idx,c in enumerate(cands)] # 优化判断
                if 'train' in file_path:
                    cands = cands[-num_cands:]
                    labels = labels[-num_cands:]

                num_example += 1
                current_turn = {'context': [query], 'response': response} # 当前对话
                if episode:
                    current_turn['context'] = episode[-1]['context'] + [episode[-1]['response']] + current_turn['context']
                episode.append(current_turn)
                for c, l in zip(cands, labels):
                    examples.append({'document': document,      # document:list
                                     'context': current_turn['context'], # current_turn['context']: list
                                     'response': c, # c:str
                                     'label': l}) # l:int 0 or 1
                    end += 1
                # examples level shuffle !!!
                if shuffle_level == 'examples':
                    positions.append((start, end))
                    start = end

            if episode_done:
                num_episode += 1
                document, episode = [], []
                # session level shuffle !!!
                if shuffle_level == 'session': # 一个完整的多轮对话结束
                    positions.append((start, end))
                    start = end
    
    print('num_episode(session) : {}'.format(num_episode))
    print('num_example : {}'.format(num_example)) # 这里的example没考虑num_cands; 最终的examples cnt是num_example*num_cands
    return examples, positions



def load_positive_examples(file_path):
    examples, _ = load_file_multiTurn(file_path, num_cands = 20, shuffle_level= 'examples', shuffle_cand = False, data_format = 'self')
    pos_examples = []
    for ex in examples:
        if ex['label'] == 1:
            pos_examples.append(ex)
    
    return pos_examples 



class Batcher(object):
    
    def __init__(self, config: dict) -> None:        
        self.device = torch.device('cuda' if config['cuda'] else 'cpu')
        
    def __call__(self, b_examples):

        b_input_ids = [] # 2-d list 
        b_target_ids = [] # 2-d list
        b_context = [] # 3-d list
        b_response = [] # 1-d list

        for ex in b_examples: # 遍历单条样本  
            input_ids = ex['input_ids']
            target_ids = ex['target_ids']   
            b_input_ids.append(input_ids)
            b_target_ids.append(target_ids)
            
            b_context.append(ex['context'])
            b_response.append(ex['response']) 
            

        b_input_ids = torch.tensor(b_input_ids, device = self.device, dtype=torch.long) # 2-d tensor 
        b_target_ids = torch.tensor(b_target_ids, device = self.device, dtype=torch.long) # 2-d tensor 
        
        return b_input_ids, b_target_ids, b_context, b_response
    



class Batcher_gen(object):
    
    def __init__(self, config: dict) -> None:        
        self.device = torch.device('cuda' if config['cuda'] else 'cpu')
        self.respond_id = 50257
        
    def __call__(self, b_examples):

        b_input_ids = [] # 2-d list 
        b_context = [] # 3-d list
        b_response = [] # 1-d list

        for ex in b_examples: # 遍历单条样本  
            input_ids = ex['input_ids'] 
            respond_id_pos = input_ids.index(self.respond_id)
            input_ids = input_ids[:respond_id_pos+1]
            # 长度不同, 2种解决办法: 在前面padding或设定batch size为1
            b_input_ids.append(input_ids)
            b_context.append(ex['context'])
            b_response.append(ex['response']) 
            
            
        b_input_ids = torch.tensor(b_input_ids, device = self.device, dtype=torch.long) # 2-d tensor 
        
        return b_input_ids, b_context, b_response
    





class Processor(object):
    def __init__(self, config: dict) -> None:
        self.tokenizer = GPT2Tokenizer.from_pretrained(config['name'], cache_dir = config['cache_dir'])
        self.block_size = config['block_size']
        self.seq_len = self.block_size - 2
        
        SPECIAL_TOKENS_DICT = {'additional_special_tokens': ["<respond>"]}
        self.tokenizer.add_special_tokens(SPECIAL_TOKENS_DICT)
        self.respond_id = self.tokenizer.convert_tokens_to_ids('<respond>')
        self.pad_id = self.tokenizer.pad_token_id  # 0


    def tokenize(self, text):
        return self.tokenizer.encode(text, text_pair=None, add_special_tokens=False)
    
    
    def __call__(self, ex):
        context_list = ex['context']
        context_ids = [self.tokenize(c) for c in context_list] # 2-d list

        flatten_context_ids = []
        for idx, utterance_ids in enumerate(context_ids):
            flatten_context_ids += utterance_ids
        
        
        resp = ex['response']
        response_ids = self.tokenize(resp) 
        response_ids = [self.respond_id] + response_ids
        
        input_ids = flatten_context_ids + response_ids
        input_ids = input_ids[-self.block_size:]
        input_ids = input_ids + [0] * (self.block_size - len(input_ids)) 

        target_ids = [-1] * len(flatten_context_ids) + response_ids[1:] + [self.tokenizer.eos_token_id] 
        target_ids = target_ids[-self.block_size:]
        target_ids = target_ids + [-1] * (self.block_size - len(target_ids)) # padding token 不需要预测
        
        ret = {
            'context': ex['context'],
            'response': ex['response'],
            'input_ids':input_ids,
            'target_ids': target_ids
        }

        return ret 




def cache_data(src_path, tgt_path):
    examples = load_positive_examples(src_path)
    
    config = {  # 处理数据
        'name': 'gpt2',
        'cache_dir': 'resource/pretrained_models/gpt2',
        'block_size': 256,
    }
    
    
    process_fn = Processor(config)
    process_num = mp.cpu_count()
    chunksize = int(len (examples) / process_num) + 1
    with timer_context('convert examples to ids'):
        with Pool() as p:
            processed_examples = p.map(process_fn, examples, chunksize = chunksize)        
        print('done.')

    save_examples_to_jsonl(processed_examples, tgt_path)
    




 
if __name__ == '__main__':

    cache_data('resource/data/personachat/train_self_original.txt', 'resource/data/personachat/processed/processed_train_self_original.jsonl')
    
    cache_data('resource/data/personachat/valid_self_original.txt', 'resource/data/personachat/processed/processed_valid_self_original.jsonl')
    
    cache_data('resource/data/personachat/test_self_original.txt', 'resource/data/personachat/processed/processed_test_self_original.jsonl')
    
    
    
        