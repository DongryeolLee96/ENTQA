#!/bin/bash

source activate numeric
set -e
set -f

# data_type="NQ"
# prediction_file_dir="/home/donaldo9603/workspace/numeric/naacl2024/data/nq_integ_0415.json"

data_type="TQ"
prediction_file_dir="/home/donaldo9603/workspace/numeric/naacl2024/data/tq_integ_0415.json"
entity_type='all' # numeric, nonnumeric, all, unknown,ent
#answer_set='inst_expand' # fb wiki inst_expand golden_answer - evaluation
answer_set='inst_entity' # for submitted 
model_type='all' #gpt35 chatgpt newbing fid gpt4 all

python gptexp_eval_rare.py --validation_dir ${prediction_file_dir} \
    --jobid ${SLURM_JOB_ID} \
    --entity ${entity_type} \
    --answer_set ${answer_set} \
    --model_type ${model_type} \
    --data_type ${data_type}
    
