#!/bin/bash

source activate numeric
set -e
set -f
#EMDR
#prediction_file_dir='/home/donaldo9603/workspace/numeric/data/nq/500samples/emdr/8502_nq_num_500_emdr.json'
#prediction_file_dir='/home/donaldo9603/workspace/numeric/data/nq/500samples/emdr/8501_nq_non_500_emdr.json'
#InstGPT_few
#prediction_file_dir='/home/donaldo9603/workspace/numeric/data/nq/500samples/instgpt/8504_nq_num_500_inst3_few.json'
#prediction_file_dir='/home/donaldo9603/workspace/numeric/data/nq/500samples/instgpt/8503_nq_non_500_inst3_few.json'

#InstGPT_zero
#prediction_file_dir='/home/donaldo9603/workspace/numeric/data/nq/500samples/instgpt/8506_nq_num_500_inst3_zero.json'
prediction_file_dir='/home/donaldo9603/workspace/numeric/data/nq/500samples/instgpt/8505_nq_non_500_inst3_zero.json'

#LLAMA2_few
#prediction_file_dir='/home/donaldo9603/workspace/numeric/data/nq/500samples/llama2/8508_nq_dev_num_500_llama2_few.json'
#prediction_file_dir='/home/donaldo9603/workspace/numeric/data/nq/500samples/llama2/8507_nq_dev_non_500_llama2_few.json'

#LLAMA2_zero
#prediction_file_dir='/home/donaldo9603/workspace/numeric/data/nq/500samples/llama2/8510_nq_dev_num_500_llama2_zero.json'
#prediction_file_dir='/home/donaldo9603/workspace/numeric/data/nq/500samples/llama2/8509_nq_dev_non_500_llama2_zero.json'

#R2D2
#prediction_file_dir='/home/donaldo9603/workspace/numeric/data/nq/500samples/r2d2/8512_nq_num_500_r2d2.json'
#prediction_file_dir='/home/donaldo9603/workspace/numeric/data/nq/500samples/r2d2/8511_nq_non_500_r2d2.json'
python gpteval_ours.py --validation_dir ${prediction_file_dir} \
    --jobid ${SLURM_JOB_ID} \
    
