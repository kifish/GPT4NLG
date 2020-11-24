import numpy as np

# https://github.com/Maluuba/nlg-eval
# benchmark
from nlgeval import NLGEval





class Metrics():
    # TODO
    pass


# https://github.com/facebookresearch/ParlAI/blob/master/parlai/core/metrics.py 




if __name__ == '__main__':
    
    import json 
    metrics = NLGEval()

    with open('resource/records/personachat/run1/log/201124-1259_generated_results.json','r',encoding='utf8') as f:
        items = json.load(f)
    
    # test_ref = [] # 2-d list
    source1_test_ref = []
    test_hyp = [] # 1-d list
    for item in items:
        # test_ref.append([item['response']])
        source1_test_ref.append(item['response'])
        test_hyp.append(item['generated_responses'][0])
    
    test_ref = [source1_test_ref]
    # https://github.com/Maluuba/nlg-eval/blob/master/nlgeval/__init__.py#L19
    # https://github.com/Maluuba/nlg-eval/blob/master/nlgeval/__init__.py#L288
    
    # print(test_ref[:10])
    # print(test_hyp[:10])

    # print(len(test_ref))
    # print(len(test_hyp))

    # for refs in zip(*test_ref):
    #     print(refs)
    
    print('computing metrics')
    eval_result = metrics.compute_metrics(test_ref, test_hyp)
    
    print("test dataset metrics:")
    print(json.dumps(str(eval_result), indent=4, sort_keys=True))
    print('**********************************') 
    
    
    