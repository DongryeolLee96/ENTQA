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
openai.api_key ='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'

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from eval import f1_score, hard_exact_match_score, soft_exact_match_score, metric_max_over_ground_truths, bem_score, insteval, insteval_conf
openai.api_key ='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'

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--validation_dir')
    parser.add_argument('--inference_type', type=str)
    parser.add_argument('--few_shot_file_dir', type=str)
    parser.add_argument('--jobid')
    parser.add_argument('--model_type')
    parser.add_argument('--data_type')
    parser.add_argument('--extracted')
    parser.add_argument('--entity')
    parser.add_argument('--few_shot_type')
    parser.add_argument('--inst_model')
    args =parser.parse_args()
    return args
def main(args):
    print(args)
    # data load
    with open(args.validation_dir) as f:
        val=json.load(f)
    print('Loaded {} data'.format(len(val)))
    type_dic=defaultdict(list)
    for d in val:
        if type(d['ans_type'])==str:
            type_dic[str(d['ans_type'])].append(d)
        else:
            type_dic['multiple'].append(d)
    
    #surface_acc=[['Entity', 'Number of data', 'Hard EM', 'Soft EM', 'F1', 'BEM','Human']]
    #against_human_acc=[['Entity', 'Number of data', 'Hard EM', 'Soft EM', 'F1', 'BEM']]
    surface_acc=[['Entity', 'Number of data', 'Hard EM', 'Soft EM', 'F1', 'BEM', 'Inst{}'.format(args.inference_type),'Human']]
    against_human_acc=[['Entity', 'Number of data', 'Hard EM', 'Soft EM', 'F1', 'BEM', 'Inst{}'.format(args.inference_type)]]
    numeric_type=['DATE', 'CARDINAL','QUANTITY', 'ORDINAL', 'MONEY', 'PERCENT', 'TIME']
    non_numeric_type=['PERSON','GPE','ORG'] #'WORK_OF_ART','FAC','PRODUCT','EVENT','LAW','LANGUAGE'
    others=['NORP', 'LOC', 'WORK_OF_ART','FAC','PRODUCT','EVENT','LAW','LANGUAGE']
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
        raise ('Invalide entity type!')
    print('Evaluation on {}-{} entity types'.format(args.data_type, args.entity))
    for key in keys:
        print(key, len(type_dic[key]))
    
    # QA model type
    if args.model_type=='all':
        model_types=['fid','gpt35', 'chatgpt', 'gpt4','newbing']
    else:
        model_types=[args.model_type]
    
    for model_type in model_types:
        # Evaluation setup
        hum_sum_all=0
        # Total average
        hard_em_sum_all=0
        soft_em_sum_all=0
        f1_sum_all=0
        inst_sum_all=0
        data_sum_all=0
        bem_sum_all=0
        nem_sum_all=0
        # Total accuracy against Human
        hard_em_hum_all=0
        soft_em_hum_all=0
        f1_hum_all=0
        inst_hum_all=0
        bem_hum_all=0
        nem_hum_all=0
        
        surface_acc.append([model_type for i in range(8)])
        against_human_acc.append([model_type for i in range(7)])
        for key in keys[:3]:
            data_sum=0
            hum_sum=0
            hard_em_sum=0
            soft_em_sum=0
            f1_sum=0
            inst_sum=0
            bem_sum=0
            nem_sum=0
            
            soft_em_hum=0
            hard_em_hum=0
            f1_hum=0
            bem_hum=0
            inst_hum=0
            nem_hum=0       
                    
            for d in tqdm(type_dic[key][:1]):
                # Golden answer preprocess
                if 'inst{}_{}'.format(args.inference_type,model_type) in d.keys():
                    continue
                else:
                
                    # if '/' in d['golden_answer']:
                    #     answer_set=d['golden_answer'].split('/')
                    # else:
                    #     answer_set=[d['golden_answer']]
                    answer_set=''
                    for ans in d['golden_answer']:
                        answer_set+=ans+'/'
                    answer_set=answer_set[:-1]
                    # answer_set=''
                    # for ans in d['inst_expand']:
                    #     answer_set+=ans+'/'
                    # answer_set=answer_set[:-1]
                    
                    # Using GPT Extracted Candidate answer or raw model-generated candidate answer
                    # try:
                    if args.extracted=='true':
                        instzero, prompt=insteval_conf(d['answer_{}_extracted'.format(model_type)], answer_set, d['question'], d['ans_type'], inst_model=args.inst_model, few_shot_type=args.few_shot_type,few_shot_dir= args.few_shot_file_dir,eval_type=args.inference_type)
                        hard_em=metric_max_over_ground_truths(hard_exact_match_score, d['answer_{}_extracted'.format(model_type)],d['golden_answer'])
                        soft_em=metric_max_over_ground_truths(soft_exact_match_score, d['answer_{}_extracted'.format(model_type)],d['golden_answer'])
                        f1=metric_max_over_ground_truths(f1_score, d['answer_{}_extracted'.format(model_type)],d['golden_answer'])
                        bem=bem_score(d['answer_{}_extracted'.format(model_type)], d['golden_answer'], d['question'])
                    elif args.extracted=='false':
                        instzero, prompt=insteval_conf(d['answer_{}'.format(model_type)], answer_set, d['question'], d['ans_type'], inst_model=args.inst_model, few_shot_type=args.few_shot_type,few_shot_dir= args.few_shot_file_dir,eval_type=args.inference_type)
                        hard_em=metric_max_over_ground_truths(hard_exact_match_score, d['answer_{}'.format(model_type)],d['golden_answer'])
                        soft_em=metric_max_over_ground_truths(soft_exact_match_score, d['answer_{}'.format(model_type)],d['golden_answer'])
                        f1=metric_max_over_ground_truths(f1_score, d['answer_{}'.format(model_type)],d['golden_answer'])
                        bem=bem_score(d['answer_{}'.format(model_type)], d['golden_answer'], d['question'])
                    else:
                        raise ('Error')
                    # except:
                    #     directory=os.getcwd()
                    #     if not (Path.cwd() / 'out' / '{}'.format(args.jobid)).exists():
                    #         os.mkdir('{}/out/{}'.format(directory, args.jobid))
                    #     with open('{}/out/{}/{}_eval.json'.format(directory, args.jobid, args.data_type), 'w') as f:
                    #         json.dump(val, f)
                        
                    #     with open('{}/out/{}/{}_{}_surface_result.csv'.format(directory, args.jobid, args.data_type,model_type), 'w', newline="") as f:
                    #         writer = csv.writer(f)
                    #         writer.writerows(surface_acc)
                            
                    #     with open('{}/out/{}/{}_{}_against_human_result.csv'.format(directory, args.jobid, args.data_type, model_type), "w", newline="") as f:
                    #         writer = csv.writer(f)
                    #         writer.writerows(against_human_acc)
                    
                    
                    data_sum+=1
                    d['hardem_{}'.format(model_type)]=hard_em
                    d['softem_{}'.format(model_type)]=soft_em

                    d['f1_{}'.format(model_type)]=f1                
                    d['bem_{}'.format(model_type)]=bem
                    d['inst{}_{}'.format(args.inference_type,model_type)]=instzero
                    d['inst{}_{}_prompt'.format(args.inference_type,model_type)]=prompt
                    
                    human=1 if d['judge_{}'.format(model_type)] else 0
                    inst=1 if ('Yes' in instzero or 'yes' in instzero) else 0
                    
                    hard_em_sum+=hard_em
                    soft_em_sum+=soft_em
                    f1_sum+=f1
                    hum_sum+=human
                    bem_sum+=bem
                    inst_sum+=inst
                    
                    if human==soft_em:
                        soft_em_hum+=1
                    if human==hard_em:
                        hard_em_hum+=1
                    if abs(human-f1)<=0.5:
                        f1_hum+=1
                    if human==bem:
                        bem_hum+=1
                    if human==inst:
                        inst_hum+=1

            
            surface=[]
            against_human=[]
            surface.append(key)
            surface.append(len(type_dic[key]))
            surface.append(hard_em_sum/data_sum)
            surface.append(soft_em_sum/data_sum)
            surface.append(f1_sum/data_sum)
            surface.append(bem_sum/data_sum)
            surface.append(inst_sum/data_sum)
            surface.append(hum_sum/data_sum)
            
            
            against_human.append(key)
            against_human.append(len(type_dic[key]))
            against_human.append(hard_em_hum/data_sum)
            against_human.append(soft_em_hum/data_sum)
            against_human.append(f1_hum/data_sum)
            against_human.append(bem_hum/data_sum)
            against_human.append(inst_hum/data_sum)
            
            surface_acc.append(surface)
            against_human_acc.append(against_human)
            
            data_sum_all+=data_sum
            hard_em_sum_all+=hard_em_sum
            soft_em_sum_all+=soft_em_sum
            f1_sum_all+=f1_sum
            bem_sum_all+=bem_sum
            inst_sum_all+=inst_sum
            hum_sum_all+=hum_sum
            
            
            hard_em_hum_all+=hard_em_hum
            soft_em_hum_all+=soft_em_hum
            f1_hum_all+=f1_hum
            bem_hum_all+=bem_hum
            inst_hum_all+=inst_hum
        
            # For single entity type
            print("QA Model: {}".format(model_type))
            print("{} data for Entity type: {}".format(len(type_dic[key]), key))
            print("Surface accuracy")
            print("Hard EM: %5.3f" %(hard_em_sum/data_sum))
            print("Soft EM: %5.3f" %(soft_em_sum/data_sum))
            print("F1: %5.3f" %(f1_sum/data_sum))
            print("BEM: %5.3f" % (bem_sum/data_sum))
            print("NEM: %5.3f" % (nem_sum/data_sum))
            print("Inst{}: ".format(args.inference_type), "%5.3f" % (inst_sum/data_sum))
            print("Human: %5.3f" % (hum_sum/data_sum))
            print()
            print("Accuracy against human")
            print("Hard EM accuracy against human %5.3f"%(hard_em_hum/data_sum))
            print("Soft EM accuracy against human %5.3f"%(soft_em_hum/data_sum))
            print("F1 th 0.5 accuracy against human %5.3f" % (f1_hum/data_sum))
            print("BEM accuracy against human %5.3f" %(bem_hum/data_sum))
            print("Inst{} accuracy against human ".format(args.inference_type), "%5.3f" %(inst_hum/data_sum))
            
        # For entire entity types
        print("All NQ {} data inference by model: {}".format(data_sum_all, model_type))
        print("Surface Accuracy")
        print("Hard EM: %5.3f" %(hard_em_sum_all/data_sum_all))
        print("Soft EM: %5.3f" %(soft_em_sum_all/data_sum_all))
        print("F1: %5.3f" %(f1_sum_all/data_sum_all))
        print("BEM: %5.3f" % (bem_sum_all/data_sum_all))
        print()
        print("Accuracy against human")
        print("Hard EM accuracy against human %5.3f"%(hard_em_hum_all/data_sum_all))
        print("Soft EM accuracy against human %5.3f"%(soft_em_hum_all/data_sum_all))
        print("F1 th 0.5 accuracy against human %5.3f" % (f1_hum_all/data_sum_all))
        print("BEM accuracy against human %5.3f" %(bem_hum_all/data_sum_all))
        print("Inst{} accuracy against human ".format(args.inference_type), "%5.3f" %(inst_hum_all/data_sum_all))
        
        surface=[]
        against_human=[]
        surface.append("Overall")
        surface.append(data_sum_all)
        surface.append(hard_em_sum_all/data_sum_all)
        surface.append(soft_em_sum_all/data_sum_all)
        surface.append(f1_sum_all/data_sum_all)
        surface.append(bem_sum_all/data_sum_all)
        surface.append(inst_sum_all/data_sum_all)
        surface.append(hum_sum_all/data_sum_all)
        
        against_human.append("Overall")
        against_human.append(data_sum_all)
        against_human.append(hard_em_hum_all/data_sum_all)
        against_human.append(soft_em_hum_all/data_sum_all)
        against_human.append(f1_hum_all/data_sum_all)
        against_human.append(bem_hum_all/data_sum_all)
        against_human.append(inst_hum_all/data_sum_all)
        
        surface_acc.append(surface)
        against_human_acc.append(against_human)    
    
    # Save
    directory=os.getcwd()
    if not (Path.cwd() / 'out' / '{}'.format(args.jobid)).exists():
        os.mkdir('{}/out/{}'.format(directory, args.jobid))
    with open('{}/out/{}/{}_eval.json'.format(directory, args.jobid, args.data_type), 'w') as f:
        json.dump(val, f)
    
    with open('{}/out/{}/{}_{}_surface_result.csv'.format(directory, args.jobid, args.data_type,model_type), 'w', newline="") as f:
        writer = csv.writer(f)
        writer.writerows(surface_acc)
        
    with open('{}/out/{}/{}_{}_against_human_result.csv'.format(directory, args.jobid, args.data_type, model_type), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(against_human_acc)
    
    
if __name__=='__main__':
    args=get_args()
    main(args)  
    

