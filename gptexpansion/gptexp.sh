#!/bin/bash

source activate numeric
set -e
set -f
data_type="TQ"

#prediction_file_dir="/home/donaldo9603/workspace/numeric/data/evouna/${data_type}_labeled.json"
prediction_file_dir="/home/donaldo9603/workspace/numeric/data/evouna/evouna_FB/${data_type}_eval_expansion.json"
#few_shot_file_dir='/home/donaldo9603/workspace/numeric/data/evouna/few_shot/nq_train_8shot_1109.json'

# Entity fewshot
#few_shot_file_dir="/home/donaldo9603/workspace/numeric/data/evouna/few_shot/1212${data_type}_train_expansion.json"

# Random fewshot
few_shot_file_dir="/home/donaldo9603/workspace/numeric/data/evouna/few_shot/1201${data_type}_train_expansion.json"

inst_model='gpt-3.5-turbo-instruct'  #'text-davinci-003'  gpt3.5-turbo-instruct gpt-4-1106-preview
entity_type='all' # numeric, nonnumeric, all, unknown,ent

python gptexp.py --validation_dir ${prediction_file_dir} \
    --few_shot_file_dir ${few_shot_file_dir} \
    --jobid ${SLURM_JOB_ID} \
    --inst_model ${inst_model} \
    --entity ${entity_type} \
    --data_type ${data_type}
    
