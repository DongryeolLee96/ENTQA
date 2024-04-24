import argparse
import json
from collections import defaultdict, Counter
import sys
import os
from os import path
import openai
import csv
from pathlib import Path
from tqdm import tqdm
import random
#openai.api_key ='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'
openai.api_key='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'
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from eval import f1_score, hard_exact_match_score, soft_exact_match_score, metric_max_over_ground_truths, bem_score, insteval


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--validation_dir')
    parser.add_argument('--few_shot_file_dir', type=str)
    parser.add_argument('--jobid')
    parser.add_argument('--data_type')
    parser.add_argument('--inst_model')
    parser.add_argument('--entity')
    args =parser.parse_args()
    return args

def main(args):
    print(args)
    # data load
    with open(args.validation_dir) as f:
        val=json.load(f)
    type_dic=defaultdict(list)
    for d in val:
        if type(d['ans_type'])==str:
            type_dic[str(d['ans_type'])].append(d)
        else:
            type_dic['multiple'].append(d)
    #numeric_type=['DATE']
    numeric_type=['DATE', 'CARDINAL','QUANTITY', 'ORDINAL', 'MONEY', 'PERCENT', 'TIME']
    non_numeric_type=['PERSON','GPE','ORG'] 
    others=['NORP', 'WORK_OF_ART','FAC','PRODUCT','EVENT','LAW','LANGUAGE', 'LOC']
    unknown=['unknown']
    if args.entity=="unknown":
        keys=unknown
    elif args.entity=='numeric':
        keys=numeric_type
    elif args.entity=='nonnumeric':
        keys=non_numeric_type
    elif args.entity=='ent':
        keys=numeric_type
        keys.extend(non_numeric_type)
        keys.extend(others)
    elif args.entity=='all':
        keys=numeric_type
        keys.extend(non_numeric_type)
        keys.extend(others)
        keys.extend(unknown)
    else:
        raise ('Invalid entity type!')
    print('Evaluation on {}-{} entity types'.format(args.data_type, args.entity))
    
    
    for key in keys:
        print(key, len(type_dic[key]))
    with open(args.few_shot_file_dir, 'r') as f:
        few_shot=json.load(f)
    
    
    try:
        for key in keys:
            instruction="You are a given a question and a set of gold-standard reference answers (split with /) written by experts. Your task is to provide other forms of gold reference answers that can also be correct for the given question. Split your answers with /.\n"
            
            # # ##### RANDOM FEW-SHOT #####
            random_pool=[]
            random.seed(1004)
            for k in few_shot.keys():
                if k!='unknown' and k!='others':
                    random_pool.extend(random.sample(few_shot[k][:8], 8))
            random_pool=random.sample(random_pool,8)
            for few in random_pool:
                answer=''
                for ans in few['expand']:
                    answer+=ans+'/'
                answer=answer[:-1]
                instruction+="Question: {}\nGold Answers: {}\n".format(few['question'], answer)
            
            ### Ours
            # if key in others:
            #     for few in few_shot['others'][:8]:
            #         answer=''
            #         for ans in few['expand']:
            #             answer+=ans+'/'
            #         answer=answer[:-1]
            #         instruction+="Question: {}\nGold Answers: {}\n".format(few['question'], answer)
            # elif key=='unknown':
            #     random_pool=[]
            #     random.seed(1004)
            #     for k in few_shot.keys():
            #         if k!='unknown' and k!='others':
            #             random_pool.extend(random.sample(few_shot[k][:8], 8))
            #     random_pool=random.sample(random_pool,8)
            #     for few in random_pool:
            #         answer=''
            #         for ans in few['expand']:
            #             answer+=ans+'/'
            #         answer=answer[:-1]
            #         instruction+="Question: {}\nGold Answers: {}\n".format(few['question'], answer)
            # else:
            #     for few in few_shot[key][:8]:
            #         answer=''
            #         for ans in few['expand']:
            #             answer+=ans+'/'
            #         answer=answer[:-1]
            #         instruction+="Question: {}\nGold Answers: {}\n".format(few['question'], answer)

        ##### 8 Entity+ 8 Random #####
            # few-shot example 생성
            # 
            # few_shot_pool=[]
            # if key in others:
            #     for few in few_shot['others'][:8]:
            #         few_shot_pool.append(few)
            # else:
            #     for few in few_shot[key][:8]:
            #         few_shot_pool.append(few)
            
            # random_pool=[]
            # random.seed(1004)
            # for k in few_shot.keys():
            #     if k!=key:
            #         random_pool.append(random.choice(few_shot[k][:8]))
            # random_pool=random.sample(random_pool,8)
            # for i in range(16):
            #     if i%2==0:
            #         few=few_shot_pool[i//2]
            #     else:
            #         few=random_pool[i//2]
            #     answer=''
            #     for ans in few['expand']:
            #         answer+=ans+'/'
            #     answer=answer[:-1]
            #     instruction+="Question: {}\nGold Answers: {}\n".format(few['question'], answer)
                
            # for few in random_pool:
            #     answer=''
            #     for ans in few['expand']:
            #         answer+=ans+'/'
            #     answer=answer[:-1]
            #     instruction+="Question: {}\nGold Answers: {}\n".format(few['question'], answer)
        
        
            for d in tqdm(type_dic[key]):
                print(key)
                print(instruction)
                golden_answer=''
                for ans in d['golden_answer']:
                    golden_answer+=ans+'/'
                prompt=instruction+"Question: {}\nGold Answers: {}".format(d['question'], golden_answer)
                print(prompt)
                response = openai.Completion.create(
                            model=args.inst_model,
                                temperature=0,
                                max_tokens=200,
                                prompt=prompt,
                                top_p=1,
                                frequency_penalty=0.0,
                                presence_penalty=0.0)
                result=response['choices'][0]['text']
                d['inst_expand_input']=prompt
                if '/' not in result:
                    d['inst_expand']=d['golden_answer'][:]
                    if len(result)<3:
                        continue
                    else:
                        d['inst_expand'].append(result)
                        continue
                else:
                    if result[-1]=='/':
                        result=result[:-1]
                    result=result.split('/')
                d['inst_expand']=d['golden_answer'][:]
                d['inst_expand'].extend(result)

    except Exception as e:
        print('Error Occur: ', e)
        directory=os.getcwd()
        if not (Path.cwd() / 'out' / '{}'.format(args.jobid)).exists():
            os.mkdir('{}/out/{}'.format(directory, args.jobid))
        with open('{}/out/{}/{}_eval.json'.format(directory, args.jobid, args.data_type), 'w') as f:
            json.dump(val, f)

    directory=os.getcwd()
    if not (Path.cwd() / 'out' / '{}'.format(args.jobid)).exists():
        os.mkdir('{}/out/{}'.format(directory, args.jobid))
    with open('{}/out/{}/{}_eval.json'.format(directory, args.jobid, args.data_type), 'w') as f:
        json.dump(val, f)
    
    
if __name__=='__main__':
    args=get_args()
    main(args)  
    

