#!/bin/bash

source activate numeric
set -e
set -f
data_type="TQ"

#prediction_file_dir="/home/donaldo9603/workspace/numeric/data/evouna/${data_type}_labeled.json"
prediction_file_dir="/home/donaldo9603/workspace/numeric/data/evouna/evouna_FB/${data_type}_eval_expansion.json"

#### 12/13 insteval-zero - expanded set
#prediction_file_dir="/home/donaldo9603/workspace/numeric/gptexpansion/out/51097/NQ_eval.json"
#prediction_file_dir="/home/donaldo9603/workspace/numeric/gptexpansion/out/51098/TQ_eval.json"

#prediction_file_dir="/home/donaldo9603/workspace/numeric/gpteval_inst/out/39650/${data_type}_eval.json"
#few_shot_file_dir='/home/donaldo9603/workspace/numeric/data/evouna/few_shot/nq_train_8shot_1109.json'
few_shot_file_dir='/home/donaldo9603/workspace/numeric/data/evouna/few_shot/nq_train_fewshot_data_annotated_1106.json'
inference_type="zero" # zero, random, entity
model_type="all" # gpt35, chatgpt, newbing, fid, gpt4
entity_type='all' # numeric, nonnumeric, all, unknown,ent
extraction='false'
few_shot_type='short' # short, sent
inst_model='gpt-3.5-turbo-instruct' #'text-davinci-003' # gpt-3.5-turbo-instruct	

python gpteval_conf.py --validation_dir ${prediction_file_dir} \
    --inference_type ${inference_type} \
    --few_shot_file_dir ${few_shot_file_dir} \
    --jobid ${SLURM_JOB_ID} \
    --model_type ${model_type} \
    --data_type ${data_type} \
    --extracted ${extraction} \
    --entity ${entity_type} \
    --few_shot_type ${few_shot_type} \
    --inst_model ${inst_model}
    
# if BEM_score, NEM_score model changes-> override_prediction=True