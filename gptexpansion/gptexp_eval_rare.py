import argparse
import json
from collections import defaultdict, Counter
import sys
import os
from os import path
import csv
import openai
from pathlib import Path
from tqdm import tqdm

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from eval import f1_score, hard_exact_match_score, soft_exact_match_score, metric_max_over_ground_truths, bem_score, insteval


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--validation_dir')
    parser.add_argument('--answer_set', type=str)
    parser.add_argument('--jobid')
    parser.add_argument('--data_type')
    parser.add_argument('--entity')
    parser.add_argument('--model_type')
    args =parser.parse_args()
    return args

def main(args):
    print(args)
    # data load
    with open(args.validation_dir) as f:
        val=json.load(f)
    type_dic=defaultdict(list)
    for d in val:
        rel_doc_cnt=max(d['a_entities_docs'])
        if rel_doc_cnt==0:
            type_dic[str(0)].append(d)
        elif rel_doc_cnt >0 and rel_doc_cnt<10:
            type_dic['0-1'].append(d)
        elif rel_doc_cnt >=10 and rel_doc_cnt<100:
            type_dic['1-2'].append(d)
        elif rel_doc_cnt >=100 and rel_doc_cnt<1000:
            type_dic['2-3'].append(d)
        elif rel_doc_cnt >=1000 and rel_doc_cnt<10000:
            type_dic['3-4'].append(d)
        elif rel_doc_cnt>=10000 and rel_doc_cnt<100000:
            type_dic['4-5'].append(d)
        elif rel_doc_cnt>=100000 and rel_doc_cnt<1000000:
            type_dic['5-6'].append(d)

    # QA model type
    if args.model_type=='all':
        model_types=['fid','gpt35', 'chatgpt', 'gpt4','newbing']
    else:
        model_types=[args.model_type]
    
    print('Evaluation on {}-{} entity types'.format(args.data_type, args.entity))
    
    for key, _ in sorted(type_dic.items()):
        print(key, len(type_dic[key]))
    
    surface_acc=[]
    against_human_acc=[]
    surface_acc.append(['Entity', 'Number of data', 'Hard EM','BEM', 'F1', 'Soft EM', 'Soft EM expansion by {}'.format(args.answer_set),'Insteval','Human'])
    against_human_acc.append(['Entity', 'Number of data', 'Hard EM','BEM', 'F1', 'Soft EM', 'Soft EM expansion by {}'.format(args.answer_set), 'Insteval'])
    
    for model in model_types:
        print('Evaluation result on {}'.format(model))
        surface_acc.append([model for i in range(7)])
        against_human_acc.append([model for i in range(6)])
        
        
        # Data total sum
        total_sum=0
        
        # Surface accuracy
        total_surf_hard_em=0
        total_surf_bem=0
        total_surf_f1=0
        total_surf_soft_em=0
        total_surf_expand=0
        total_surf_hum=0
        total_surf_insteval=0
        
        # Against Human acc
        total_hard_em_hum=0
        total_bem_hum=0
        total_f1_hum=0
        total_soft_em_hum=0
        total_expand_hum=0
        total_insteval_hum=0
        
        for key, _ in sorted(type_dic.items()):                
            sum=0
            # surface acc
            surf_hard_em=0
            surf_bem=0
            surf_f1=0
            surf_soft_em=0
            surf_expand=0
            surf_hum=0
            surf_insteval=0
            
            # Against Human acc
            hard_em_hum=0
            bem_hum=0
            f1_hum=0
            soft_em_hum=0
            expand_hum=0
            insteval_hum=0
            
            for d in type_dic[key]:
                sum+=1
                
                # Surface
                hard_em=metric_max_over_ground_truths(hard_exact_match_score, d['answer_{}'.format(model)], d['golden_answer'])
                surf_hard_em+=hard_em
                
                bem=bem_score(d['answer_{}'.format(model)], d['golden_answer'], d['question'])
                surf_bem+=bem
                
                f1=metric_max_over_ground_truths(f1_score, d['answer_{}'.format(model)], d['golden_answer'])
                surf_f1+=f1
                
                soft_em=metric_max_over_ground_truths(soft_exact_match_score, d['answer_{}'.format(model)], d['golden_answer'])
                surf_soft_em+=soft_em
                
                expand=metric_max_over_ground_truths(soft_exact_match_score, d['answer_{}'.format(model)], d[args.answer_set])
                surf_expand+=expand
                
                human= 1 if d['judge_{}'.format(model)] else 0
                surf_hum+=human

                inst=1 if ('Yes' in d['insteval_{}'.format(model)] or 'yes' in d['insteval_{}'.format(model)]) else 0
                surf_insteval+=inst
                
                # Against Human
                if hard_em==human:
                    hard_em_hum+=1
                if bem==human:
                    bem_hum+=1
                if abs(f1-human)<=0.5:
                    f1_hum+=1
                if soft_em==human:
                    soft_em_hum+=1
                if expand==human:
                    expand_hum+=1
                if inst==human:
                    insteval_hum+=1
                    
            # Surface accuracy save
            surface = []
            surface.append(key)
            surface.append(sum)
            surface.append(surf_hard_em/sum)
            surface.append(surf_bem/sum)
            surface.append(surf_f1/sum)
            surface.append(surf_soft_em/sum)
            surface.append(surf_expand/sum)
            surface.append(surf_insteval/sum)
            surface.append(surf_hum/sum)
            
            surface_acc.append(surface)
            # Against human acc save 
            against_human=[]
            against_human.append(key)
            against_human.append(sum)
            against_human.append(hard_em_hum/sum)
            against_human.append(bem_hum/sum)
            against_human.append(f1_hum/sum)
            against_human.append(soft_em_hum/sum)
            against_human.append(expand_hum/sum)
            against_human.append(insteval_hum/sum)
            
            against_human_acc.append(against_human)
            
            total_sum+=sum
            total_surf_hard_em+=surf_hard_em
            total_surf_bem+=surf_bem
            total_surf_f1+=surf_f1
            total_surf_soft_em+=surf_soft_em
            total_surf_expand+=surf_expand
            total_surf_insteval+=surf_insteval
            total_surf_hum+=surf_hum
            
            total_hard_em_hum+=hard_em_hum
            total_bem_hum+=bem_hum
            total_f1_hum+=f1_hum
            total_soft_em_hum+=soft_em_hum
            total_expand_hum+=expand_hum
            total_insteval_hum+=insteval_hum
            
            print('Surface accuracy for {} type {}'.format(sum, key))
            print('Hard EM: %5.3f' %(surf_hard_em/sum))
            print('BEM: %5.3f' %(surf_bem/sum))
            print('F1: %5.3f' %(surf_f1/sum))
            print('Soft EM: %5.3f' %(surf_soft_em/sum))
            print('Soft EM on {} exapansion: {}'.format(args.answer_set, surf_expand/sum))
            print('Insteval: %5.3f' %(surf_insteval/sum))
            print('Human: %5.3f' %(surf_hum/sum))
            print()
            
            print('Accuracy against human for {} type {}'.format(sum, key))
            print('Hard EM against human: %5.3f' %(hard_em_hum/sum))
            print('BEM against human: %5.3f' %(bem_hum/sum))
            print('F1 against human: %5.3f' %(f1_hum/sum))
            print('Soft EM against human: %5.3f' %(soft_em_hum/sum))
            print('Soft EM on {} exapansion against human: {}'.format(args.answer_set, expand_hum/sum))
            print('Insteval against human: %5.3f' %(insteval_hum/sum))
            print()
        
        
        
        print('Evaluation Result on QA model {} on {} data.'.format(model, total_sum))
        print('Surface accuracy')
        print('Hard EM: %5.3f' %(total_surf_hard_em/total_sum))
        print('BEM: %5.3f' %(total_surf_bem/total_sum))
        print('F1: %5.3f' %(total_surf_f1/total_sum))
        print('Soft EM: %5.3f' %(total_surf_soft_em/total_sum))
        print('Soft EM on {} exapansion: {}'.format(args.answer_set, total_surf_expand/total_sum))
        print('Insteval: %5.3f' %(total_surf_insteval/total_sum))
        print('Human: %5.3f' %(total_surf_hum/total_sum))
        print()
        
        print('Accuracy against human')
        print('Hard EM against human: %5.3f' %(total_hard_em_hum/total_sum))
        print('BEM against human: %5.3f' %(total_bem_hum/total_sum))
        print('F1 against human: %5.3f' %(total_f1_hum/total_sum))
        print('Soft EM against human: %5.3f' %(total_soft_em_hum/total_sum))
        print('Soft EM on {} exapansion against human: {}'.format(args.answer_set, total_expand_hum/total_sum))
        print('Insteval against human: %5.3f' %(total_insteval_hum/total_sum))
        print()
        
        surface=[]
        surface.append('Overall')
        surface.append(total_sum)
        surface.append(total_surf_hard_em/total_sum)
        surface.append(total_surf_bem/total_sum)
        surface.append(total_surf_f1/total_sum)
        surface.append(total_surf_soft_em/total_sum)
        surface.append(total_surf_expand/total_sum)
        surface.append(total_surf_insteval/total_sum)
        surface.append(total_surf_hum/total_sum)
        
        surface_acc.append(surface)
        
        against_human=[]
        against_human.append('Overall')
        against_human.append(total_sum)
        against_human.append(total_hard_em_hum/total_sum)
        against_human.append(total_bem_hum/total_sum)
        against_human.append(total_f1_hum/total_sum)
        against_human.append(total_soft_em_hum/total_sum)
        against_human.append(total_expand_hum/total_sum)
        against_human.append(total_insteval_hum/total_sum)
        
        against_human_acc.append(against_human)
        
    # Save
    directory=os.getcwd()
    if not (Path.cwd() / 'eval' / '{}'.format(args.jobid)).exists():
        os.mkdir('{}/eval/{}'.format(directory, args.jobid))
    
    with open('{}/eval/{}/{}_{}_expansion_surface_result.csv'.format(directory, args.jobid, args.data_type, args.answer_set), 'w', newline="") as f:
        writer = csv.writer(f)
        writer.writerows(surface_acc)
    
    with open('{}/eval/{}/{}_{}_expansion_against_human_result.csv'.format(directory, args.jobid, args.data_type, args.answer_set), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(against_human_acc)

    
if __name__=='__main__':
    args=get_args()
    main(args)  
    

