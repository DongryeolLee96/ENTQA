#!/bin/bash

source activate numeric
set -e
set -f
# data_type="NQ"
# prediction_file_dir="/home/donaldo9603/workspace/numeric/gptexpansion/out/NQ_llama2_70b_expansion/${data_type}_eval.json"

data_type="NQ"
# prediction_file_dir="/home/donaldo9603/workspace/numeric/gptexpansion/out/51097/${data_type}_eval.json"
prediction_file_dir="/home/donaldo9603/workspace/numeric/naacl2024/data/nq_integ.json"

data_type="TQ"
#prediction_file_dir="/home/donaldo9603/workspace/numeric/gptexpansion/out/51098/${data_type}_eval.json"
#prediction_file_dir="/home/donaldo9603/workspace/numeric/naacl2024/data/tq_integ.json"

# data_type="TQ"
# prediction_file_dir="/home/donaldo9603/workspace/numeric/gptexpansion/out/TQ_llama2_70ilab_expansion/${data_type}_eval.json"

# prediction_file_dir="/home/donaldo9603/workspace/numeric/data/evouna/${data_type}_labeled.json"
#prediction_file_dir="/home/donaldo9603/workspace/numeric/data/evouna/evouna_FB/${data_type}_eval_expansion.json"
# prediction_file_dir="/home/donaldo9603/workspace/numeric/gptexpansion/out/51569/${data_type}_eval.json"
entity_type='all' # numeric, nonnumeric, all, unknown,ent
#answer_set='inst_expand' # fb wiki inst_expand golden_answer - evaluation
answer_set='inst_entity' # for submitted 
model_type='all' #gpt35 chatgpt newbing fid gpt4 all

python gptexp_eval.py --validation_dir ${prediction_file_dir} \
    --jobid ${SLURM_JOB_ID} \
    --entity ${entity_type} \
    --answer_set ${answer_set} \
    --model_type ${model_type} \
    --data_type ${data_type}
    
