from functools import wraps
import mmap
import time 
from contextlib import ContextDecorator



class timer_context(ContextDecorator):
    '''Elegant Timer via ContextDecorator'''
    def __init__(self, name):
        self.name = name
    def __enter__(self):
        print('{} ...'.format(self.name))
        self.start = time.time()
    def __exit__(self, *args):
        self.end = time.time()
        self.elapse = self.end - self.start
        print("Processing time for [{}] is: {} seconds".format(self.name, self.elapse))




def cal_elapsed_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = float(format((elapsed_time - (elapsed_mins * 60)), '.2f'))
    return elapsed_mins, elapsed_secs




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


def load_examples_from_jsonl(file_path):
    examples = []
    print('reading data ...')
    with open(file_path,'r',encoding='utf8') as f:
        for line in tqdm(f, total= get_num_lines(file_path)): # generator
            line = line.strip()
            example = json.loads(line)
            examples.append(example)
            del line 
    print('num of examples : {}'.format(len(examples))) 
    return examples





