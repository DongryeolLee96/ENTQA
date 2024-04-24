import argparse
import json
import sys
import os
from os import path

from collections import defaultdict
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from eval import spearman, return_bertscore, f1_thereshold, f1_score, hard_exact_match_score, soft_exact_match_score, metric_max_over_ground_truths, bem_score

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--validation_dir', required=True)
    parser.add_argument('--jobid', required=True)
    args =parser.parse_args()
    return args
def main(args):
    # data load
    print(args)
    with open(args.validation_dir) as f:
        val=json.load(f)
    print('Loaded {} data'.format(len(val)))
    type_dic=defaultdict(list)
    for d in val:
        # if d['ans_type']!='unknown' and type(d['ans_type'])==str:
        type_dic[str(d['ans_type'])].append(d)
    for key in type_dic.keys():
        print(key, len(type_dic[key]))
    
    human_label_cnt=0
    human_sum=0
    soft_em_sum=0
    hard_em_sum=0
    
    f1_sum=0
    f1_th_sum=0
    bem_sum=0
    
    human_pred=[]
    soft_em_pred=[]
    hard_em_pred=[]
    f1_pred=[]
    bem_pred=[]
    
    # Evaluation result for each metric
    for d in val:
        human_label_cnt+=1
        human_sum+=d['human']
        human_pred.append(d['human'])
       
        # EM    
        d['soft_em']=metric_max_over_ground_truths(soft_exact_match_score, d['prediction'], d['answer'])
        soft_em_sum+=d['soft_em']
        
        d['hard_em']=metric_max_over_ground_truths(hard_exact_match_score, d['prediction'], d['answer'])
        hard_em_sum+=d['hard_em']
        
        # F1
        d['f1']=metric_max_over_ground_truths(f1_score, d['prediction'], d['answer'])
        f1_sum+=d['f1']    

        # F1 threshold
        d['f1_th']=metric_max_over_ground_truths(f1_thereshold, d['prediction'], d['answer'])
        f1_th_sum+=d['f1_th']    
        
        # BEM

        d['bem']=bem_score(d['prediction'], d['answer'], d['question'])
        bem_sum+=d['bem']

        soft_em_pred.append(d['soft_em'])
        hard_em_pred.append(d['hard_em'])
        f1_pred.append(d['f1'])
        bem_pred.append(d['bem'])
            
            
            
        
    print('Evaluating prediction file: {}'.format(args.validation_dir))
    print('Evaluated {} annotated samples'.format(human_label_cnt))
    print()
    print('Surface accuracy')
    print('Human: %5.3f' %(human_sum/human_label_cnt))
    print('Soft EM: %5.3f' %(soft_em_sum/human_label_cnt))
    print('Hard EM: %5.3f' %(hard_em_sum/human_label_cnt))

    print('F1: %5.3f' %(f1_sum/human_label_cnt))   
    print('F1_0.5_threshold: %5.3f' %(f1_th_sum/human_label_cnt)) 
    print('BEM: %5.3f' %(bem_sum/human_label_cnt))
    print()
    
    # Metric against human           
    hard_em_hum=0
    soft_em_hum=0
    f1_hum=0
    f1_th_hum=0
    bem_hum=0
    bs_hum=0
    bs_th_hum=0
    for d in val:     
        if d['human']==d['soft_em']:
            soft_em_hum+=1
            
        if d['human']==d['hard_em']:
            hard_em_hum+=1
        if d['human']==d['bem']:
            bem_hum+=1
        if d['human']==d['f1_th']:
            f1_th_hum+=1
    print('Metric accuracy against human label')
    print('Soft EM accuracy against human: %5.3f' %(soft_em_hum/human_label_cnt))
    print('Hard EM accuracy against human: %5.3f' %(hard_em_hum/human_label_cnt))
    print('F1 threshold 0.5 accuracy against human: %5.3f' %(f1_th_hum/human_label_cnt))
    print('BEM accuracy against human: %5.3f' %(bem_hum/human_label_cnt))
    # print('F1, BERTSCORE are computed based on the distance against human label EX) Human:1 F1:0.7 -> 0.7 / Human: 0 F1:0.7 -> 0.3')
    print()

if __name__=='__main__':
    args=get_args()
    main(args)