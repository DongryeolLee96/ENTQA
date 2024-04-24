import argparse
import json
from eval import spearman, return_bertscore, f1_thereshold, f1_score, hard_exact_match_score, soft_exact_match_score, metric_max_over_ground_truths, bem_score, num_integ_score

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--validation_dir', required=True)
    parser.add_argument('--override_prediction', default=False)
    
    args =parser.parse_args()
    return args
def main(args):
    # data load
    print(args)
    with open(args.validation_dir) as f:
        val=json.load(f)
    print('Loaded {} data'.format(len(val)))
    human_label_cnt=0
    human_sum=0
    
    em_sum=0
    f1_sum=0
    f1_th_sum=0
    bem_sum=0
    bs_sum=0
    bs_th_sum=0
    # emnem_sum=0
    
    human_pred=[]
    # emnem_pred=[]
    em_pred=[]
    f1_pred=[]
    bs_pred=[]
    bem_pred=[]
    # nem_ans_pred=[]
    # nem_sen_pred=[]
    nem_integ_pred=[]
    # Evaluation result for each metric
    for d in val:
        if 'human' in d.keys():
            human_label_cnt+=1
            human_sum+=d['human']
            # BERTSCORE
            if args.override_prediction:
                d['bs']=return_bertscore(d['prediction'], d['answer'])
                bs_sum+=d['bs']
                
                d['bs_th']=1 if d['bs']>=0.5 else 1
                bs_th_sum+=d['bs_th']
            else:
                if 'bs' in d.keys():
                    bs_sum+=d['bs']
                    
                    d['bs_th']=1 if d['bs']>=0.5 else 1
                    bs_th_sum+=d['bs_th']
                
                else:
                    d['bs']=return_bertscore(d['prediction'], d['answer'])
                    bs_sum+=d['bs']
                    
                    d['bs_th']=1 if d['bs']>=0.5 else 1
                    bs_th_sum+=d['bs_th']
                    
            # EM
            if args.override_prediction:
                d['em']=metric_max_over_ground_truths(hard_exact_match_score, d['prediction'], d['answer'])
                em_sum+=d['em']
            else:
                if 'em' in d.keys():
                    em_sum+=d['em']
                else:
                    d['em']=metric_max_over_ground_truths(hard_exact_match_score, d['prediction'], d['answer'])
                    em_sum+=d['em']
            
            # F1
            if args.override_prediction:
                d['f1']=metric_max_over_ground_truths(f1_score, d['prediction'], d['answer'])
                f1_sum+=d['f1']   
            else:
                if 'f1' in d.keys():
                    f1_sum+=d['f1']
                else:
                    d['f1']=metric_max_over_ground_truths(f1_score, d['prediction'], d['answer'])
                    f1_sum+=d['f1']    

            # F1 threshold
            if args.override_prediction:
                d['f1_th']=metric_max_over_ground_truths(f1_thereshold, d['prediction'], d['answer'])
                f1_th_sum+=d['f1_th']
            else:
                if 'f1_th' in d.keys():
                    f1_th_sum+=d['f1_th']
                else:
                    d['f1_th']=metric_max_over_ground_truths(f1_thereshold, d['prediction'], d['answer'])
                    f1_th_sum+=d['f1_th']    
            
            # BEM
            if args.override_prediction:
                d['bem']=bem_score(d['prediction'], d['answer'], d['question'])
                bem_sum+=d['bem']
            else:
                if 'bem' in d.keys():
                    bem_sum+=d['bem']
                else:
                    d['bem']=bem_score(d['prediction'], d['answer'], d['question'])
                    bem_sum+=d['bem']

            human_pred.append(d['human'])
    
            em_pred.append(d['em'])
            f1_pred.append(d['f1'])
            bs_pred.append(d['bs'])
            bem_pred.append(d['bem'])
            
            
            
        
    print('Evaluating prediction file: {}'.format(args.validation_dir))
    print('Evaluated {} annotated samples'.format(human_label_cnt))
    print()
    print('Evaluation result for each metrics')
    print('Human: %5.3f' %(human_sum/human_label_cnt))
    print('EM: %5.3f' %(em_sum/human_label_cnt))
    print('F1: %5.3f' %(f1_sum/human_label_cnt))   
    print('F1_0.5_threshold: %5.3f' %(f1_th_sum/human_label_cnt)) 
    print('BEM: %5.3f' %(bem_sum/human_label_cnt))
    print('BERTSCORE: %5.3f' %(bs_sum/human_label_cnt))
    print('BERTSCORE_0.5_threshold: %5.3f' %(bs_th_sum/human_label_cnt))
    print()
    if 'num' in args.validation_dir.split('/')[-1]:
        # num_ans_sum=0
        # num_sen_sum=0
        num_integ_sum=0
        for d in val:
            if 'human' in d.keys():
                
                if args.override_prediction:
                    # d['nem_ans']=num_ans_score(d['prediction'], d['answer'], d['question'])
                    # num_ans_sum+=d['nem_ans']
                    
                    # d['nem_sen']=num_sen_score(d['prediction'], d['answer'], d['question'])
                    # num_sen_sum+=d['nem_sen']
                    
                    d['nem_integ']=num_integ_score(d['prediction'], d['answer'], d['question'])
                    num_integ_sum+=d['nem_integ']
                    
                    # if d['em']==1:
                    #     d['emnem']=1
                    #     emnem_sum+=d['emnem']
                    # else:
                    #     d['emnem']=num_integ_score(d['prediction'], d['answer'], d['question'])
                    #     emnem_sum+=d['emnem']
                else:
                    # NEM_ans
                    # if 'nem_ans' in d.keys():
                    #     num_ans_sum+=d['nem_ans']
                    # else:
                    #     d['nem_ans']=num_ans_score(d['prediction'], d['answer'], d['question'])
                    #     num_ans_sum+=d['nem_ans']
                    
                    # # NEM_sen
                    # if 'nem_sen' in d.keys():
                    #     num_sen_sum+=d['nem_sen']
                    # else:
                    #     d['nem_sen']=num_sen_score(d['prediction'], d['answer'], d['question'])
                    #     num_sen_sum+=d['nem_sen']

                    # NEM_integ
                    if 'nem_integ' in d.keys():
                        num_integ_sum+=d['nem_integ']
                    else:
                        d['nem_integ']=num_integ_score(d['prediction'], d['answer'], d['question'])
                        num_integ_sum+=d['nem_integ']

                    # EMNEM_integ
                    # if 'emnem' in d.keys():
                    #     emnem_sum+=d['emnem']
                    # else:
                    #     if d['em']==1:
                    #         d['emnem']=1
                    #         emnem_sum+=d['emnem']
                    #     else:
                    #         d['emnem']=num_integ_score(d['prediction'], d['answer'], d['question'])
                    #         emnem_sum+=d['emnem']
                    
                # nem_ans_pred.append(d['nem_ans'])
                # nem_sen_pred.append(d['nem_sen'])
                nem_integ_pred.append(d['nem_integ'])
                # emnem_pred.append(d['emnem'])
                
        # print('NEM_ans: %5.3f' %(num_ans_sum/human_label_cnt))
        # print('NEM_sen: %5.3f' %(num_sen_sum/human_label_cnt))   
        print('NEM_integ: %5.3f' %(num_integ_sum/human_label_cnt))
        # print('EM+NEM_integ: %5.3f' %(emnem_sum/human_label_cnt))
    
    # Metric against human           
    em_hum=0
    f1_hum=0
    f1_th_hum=0
    bem_hum=0
    bs_hum=0
    bs_th_hum=0
    for d in val:
        if 'human' in d.keys():
            if d['human']==1:
               f1_hum+=d['f1']
               bs_hum+=d['bs']
            else:
                f1_hum+=1-d['f1']
                bs_hum+=1-d['bs']
                
    
            if d['human']==d['em']:
                em_hum+=1
            if d['human']==d['bem']:
                bem_hum+=1
            if d['human']==d['f1_th']:
                f1_th_hum+=1
            if d['human']==d['bs_th']:
                bs_th_hum+=1
    print('Metric accuracy against human label')
    print('EM accuracy against human: %5.3f' %(em_hum/human_label_cnt))
    print('F1 accuracy against human: %5.3f' %(f1_hum/human_label_cnt))
    print('F1 threshold 0.5 accuracy against human: %5.3f' %(f1_th_hum/human_label_cnt))
    print('BEM accuracy against human: %5.3f' %(bem_hum/human_label_cnt))
    print('BERTSCORE accuracy against human: %5.3f' %(bs_hum/human_label_cnt))
    print('BERTSCORE threshold 0.5 accuracy against human: %5.3f' %(bs_th_hum/human_label_cnt))
    print('F1, BERTSCORE are computed based on the distance against human label EX) Human:1 F1:0.7 -> 0.7 / Human: 0 F1:0.7 -> 0.3')
    print()

    if 'num' in args.validation_dir.split('/')[-1]:
        # num_ans_hum=0
        # num_sen_hum=0
        num_integ_hum=0
        # emnem_hum=0
        for d in val:
            if 'human' in d.keys():
                # if d['human']==d['nem_ans']:
                #     num_ans_hum+=1
                # if d['human']==d['nem_sen']:
                #     num_sen_hum+=1
                if d['human']==d['nem_integ']:
                    num_integ_hum+=1
                # if d['human']==d['emnem']:
                #     emnem_hum+=1
        # print('NEM_ans accuracy against human: %5.3f' %(num_ans_hum/human_label_cnt))
        # print('NEM_sen accuracy against human: %5.3f' %(num_sen_hum/human_label_cnt))
        print('NEM_integ accuracy against human: %5.3f' %(num_integ_hum/human_label_cnt))
        # print('EM+NEM_integ accuracy against human: %5.3f' %(emnem_hum/human_label_cnt))
        print()
        print('Metric Spearsman Rho against human label')
        print('EM  Spearsman Rho against human: %5.3f' %(spearman(human_pred, em_pred)))
        print('F1 Spearsman Rho against human: %5.3f' %(spearman(human_pred, f1_pred)))
        print('BEM Spearsman Rho against human: %5.3f' %(spearman(human_pred, bem_pred)))
        print('BERTSCORE Spearsman Rho against human: %5.3f' %(spearman(human_pred, bs_pred)))
        # print('NEM_ans  Spearsman Rho against human: %5.3f' %(spearman(human_pred, nem_ans_pred)))
        # print('NEM_sen Spearsman Rho against human: %5.3f' %(spearman(human_pred, nem_sen_pred)))
        print('NEM_integ Spearsman Rho against human: %5.3f' %(spearman(human_pred, nem_integ_pred)))
        # print('EM+NEM_integ Spearsman Rho against human: %5.3f' %(spearman(human_pred, emnem_pred)))
    # Save prediction file
    with open(args.validation_dir, 'w') as f:
        json.dump(val, f)
        
if __name__=='__main__':
    args=get_args()
    main(args)