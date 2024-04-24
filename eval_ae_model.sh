#!/bin/bash

source activate numeric
set -e
set -f
#val_dir='/home/donaldo9603/workspace/numeric/data/ours/numeric_dev_230926.jsonl'
val_dir='/home/donaldo9603/workspace/numeric/data/ours/numeric_sentence_dev.jsonl'
#val_dir='/home/donaldo9603/workspace/BEM/data/ae_dev.jsonl'

batch_size=64


# Trained BEM 
checkpoint_dir='/home/donaldo9603/workspace/BEM/checkpoint/checkpoint-32955/checkpoint-1616'
model_name='bert'
# Trained numeric Roberta
# checkpoint_dir='/home/donaldo9603/workspace/BEM/checkpoint/checkpoint-32904/checkpoint-1206'
# model_name='roberta'

# Trained numeric bert
# checkpoint_dir='/home/donaldo9603/workspace/BEM/checkpoint/checkpoint-32905/checkpoint-1608'
# model_name='bert'
python eval_ae_model.py --validation_dir ${val_dir} \
    --batch_size ${batch_size} \
    --model_name ${model_name} \
    --checkpoint_dir ${checkpoint_dir}
