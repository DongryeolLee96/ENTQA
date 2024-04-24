#!/bin/bash

source activate numeric
set -e
set -f
# Ans-Ans
# val_dir='/home/donaldo9603/workspace/numeric/data/ours/numeric_dev_230926.jsonl'
# train_dir='/home/donaldo9603/workspace/numeric/data/ours/numeric_train_230926.jsonl'

# # Ans-Sen
# val_dir='/home/donaldo9603/workspace/numeric/data/ours/numeric_sentence_dev.jsonl'
# train_dir='/home/donaldo9603/workspace/numeric/data/ours/numeric_sentence_train.jsonl'

# Ans-Ans / Ans-Sen Integrated
# val_dir='/home/donaldo9603/workspace/numeric/data/ours/numeric_dev.jsonl'
# train_dir='/home/donaldo9603/workspace/numeric/data/ours/numeric_integ_train_add_neg_231005.jsonl'


# Only NQ
val_dir='/home/donaldo9603/workspace/numeric/data/ours/only_nq/numeric_dev.jsonl'
train_dir='/home/donaldo9603/workspace/numeric/data/ours/only_nq/numeric_train.jsonl'

output_dir="/home/donaldo9603/workspace/BEM/checkpoint/checkpoint-${SLURM_JOB_ID}"
batch_size=64
epoch=1

model_name='microsoft/deberta-v3-base'
#model_name='bert'
#model_name='roberta'

#BEM model
# val_dir='/home/donaldo9603/workspace/BEM/data/ae_dev.jsonl'
# train_dir='/home/donaldo9603/workspace/BEM/data/train.jsonl'


python train_ae_model.py --train_dir ${train_dir} \
    --validation_dir ${val_dir} \
    --output_dir ${output_dir}\
    --batch_size ${batch_size} \
    --model_name ${model_name} \
    --epoch ${epoch}
