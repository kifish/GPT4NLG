from transformers import GPT2PreTrainedModel, GPT2Model, GPT2Config
import torch.nn as nn


class GptForResponseGernation(GPT2PreTrainedModel):
    '''
    The 'generate' function is so handy :) .
    '''

    def __init__(self, tokenizer, total_config, use_token_type_embedding = False):
        config = GPT2Config.from_pretrained(total_config['name'], cache_dir = total_config['cache_dir']) # https://huggingface.co/transformers/model_doc/gpt2.html#gpt2config
        super(GptForResponseGernation, self).__init__(config)
        self.transformer = GPT2Model.from_pretrained(total_config['name'], cache_dir = total_config['cache_dir'], from_tf = False)
        self.use_token_type_embedding = use_token_type_embedding # 是否使用 token type embedding
        print('initing the model ...')
        print('vocab size: {}'.format(len(tokenizer)))
        self.transformer.resize_token_embeddings(len(tokenizer))
        self.lm_head = nn.Linear(config.n_embd, len(tokenizer), bias=False) # 目前没加dropout
        self.config.vocab_size = len(tokenizer)
        self.tie_weights()


    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, input_ids, past = None, attention_mask = None, token_type_ids=None, **kwargs): 
        # transformer_outputs = self.transformer(input_ids, past = past, token_type_ids = token_type_ids) # 
        transformer_outputs = self.transformer(input_ids, past = past, token_type_ids = token_type_ids, **kwargs) # 与huggingface transformer3.3.1兼容(transformers/generation_utils.py538行左右。
        # 以及656行左右, 用于兼容return_dict这个参数), 
        if 'return_dict' in kwargs: # for generation
            # print(type(transformer_outputs))
            # print(transformer_outputs.keys())
            # print(transformer_outputs['last_hidden_state'].shape)
            transformer_outputs.logits = self.lm_head(transformer_outputs['last_hidden_state'])
            return transformer_outputs
        else: # for train and inference
            hidden_states = transformer_outputs[0] # https://huggingface.co/transformers/model_doc/bert.html#transformers.BertModel.forward
            lm_logits = self.lm_head(hidden_states)
            return (lm_logits,) + transformer_outputs[1:]

    def batch_decode(self, input_ids, max_len, min_len, early_stopping, beam_size, \
                     repetition_penalty, eos_id, length_penalty = 1, do_sample = False, \
                     start_id = None, token_type_ids = None, \
                     no_repeat_ngram_size = None,  \
                     pad_token_id = 0, num_return_sequences = None):
                     
        output_sequences = self.generate( # 用了huggingface gpt2的模型的generate; 支持批量生成回复
            input_ids = input_ids,
            max_length = input_ids.size(1) + max_len, # 30; input_ids.size(1)为输入句子的长度
            # min_length = input_ids.size(1) + min_len, # 0
            do_sample = do_sample,
            # early_stopping = early_stopping, # False
            num_beams = beam_size, # 1
            repetition_penalty = repetition_penalty, # 1.0
            # pad_token_id=0,
            num_return_sequences = num_return_sequences,
            pad_token_id = pad_token_id, # 原始的gpt无 pad token
            eos_token_id = eos_id,
            # no_repeat_ngram_size = no_repeat_ngram_size, # 0
            length_penalty = length_penalty, # 1.0
        )
        
        return output_sequences
    





# Some enhanced models will be released; Stay tuned.


