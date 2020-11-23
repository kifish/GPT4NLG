export PYTHONNOUSERSITE=True
conda activate GPT4NLG
export LANG="C.UTF-8"
export PYTHONPATH=$PYTHONPATH:./


# process data
see src/dataset/dataset_utils/readmd.md

# train
# 单机单卡
CUDA_VISIBLE_DEVICES=2 python src/main.py

# 单机多卡
CUDA_VISIBLE_DEVICES=0,1,6,7 python src/main.py


# val
CUDA_VISIBLE_DEVICES=2,3 python src/main.py



