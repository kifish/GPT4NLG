
from torch.utils.data import Dataset
import pprint
from src.dataset.dataset_utils.persona import * 


class PersonaDataset(Dataset):
    def __init__(self, file_path, debug = False, debug_num = 10, verbose = False):
        examples = load_positive_examples(file_path)
        if debug:
            self.data = examples[:debug_num]
        else:
            self.data = examples
        
        if verbose:
            self.display(5, file_path)

            
    def display(self, index, file_path):
        data = self.__getitem__(index)
        print('one sample from {} : '.format(file_path))
        Printer = pprint.PrettyPrinter(indent=4)
        Printer.pprint(data)
        
        
    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)




class CachedPersonaDataset(Dataset):
    def __init__(self, file_path, debug = False, debug_num = 10, verbose = False, **kws):
        
        examples = read_cached_data_tqdm(file_path, debug = debug, debug_num = debug_num)
        
        if debug:
            self.data = examples[:debug_num]
        else:
            self.data = examples
        
        if verbose:
            self.display(5, file_path)

            
    def display(self, index, file_path):
        data = self.__getitem__(index)
        print('one sample from {} : '.format(file_path))
        Printer = pprint.PrettyPrinter(indent=4)
        Printer.pprint(data)
        
        
    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)




 
 