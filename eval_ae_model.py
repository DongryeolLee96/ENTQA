import argparse
import jsonlines
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
import torch
from datasets import load_metric, Dataset
import evaluate
import numpy as np
from eval import f1_threshold, return_bertscore

accuracy = evaluate.load('accuracy')
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    print(predictions)
    print(labels)
    predictions = np.argmax(predictions, axis=1)
    print(len(predictions))
    print(len(labels))
    return accuracy.compute(predictions=predictions, references=labels)    

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--validation_dir', required=True)
    parser.add_argument('--batch_size', required=True, type=int)
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--checkpoint_dir', type=str)
    
    args =parser.parse_args()
    return args
def main(args):
    # data load
    val=[]
    with jsonlines.open(args.validation_dir) as f:
        for line in f:
            val.append(line)
    # [{'question': , 'reference': , 'candidate': , 'score': 1.0 / 0.0}]
    if 'bert' in args.model_name:
        val_data={'id':[i for i in range(len(val))],
                'text':[d['candidate']+' [SEP] '+d['reference']+' [SEP] '+d['question'] for d in val],
                'label':[1 if d['score']==1.0 else 0 for d in val]}
        val_data=Dataset.from_dict(val_data)
        # List to Dataset conversion
        
        id2label = {0: "NEGATIVE", 1: "POSITIVE"}
        label2id = {"NEGATIVE": 0, "POSITIVE": 1}
        
        # Preprocessing
        if args.model_name=='bert':
            tokenizer=AutoTokenizer.from_pretrained('bert-base-uncased')    
            model = AutoModelForSequenceClassification.from_pretrained(args.checkpoint_dir, num_labels=2, id2label=id2label, label2id=label2id)
        elif args.model_name == 'roberta':
            tokenizer=AutoTokenizer.from_pretrained('roberta-base')
            model = AutoModelForSequenceClassification.from_pretrained(args.checkpoint_dir, num_labels=2, id2label=id2label, label2id=label2id)
        
        def preprocess_function(example):
            return tokenizer(example['text'], padding = 'max_length',truncation=True)
        
        tokenized_val=val_data.map(preprocess_function, batched=True)
        training_args = TrainingArguments("test_trainer")
        
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        model.to(device)
        cnt=0
        for d in val:
            text='{} [SEP] {} [SEP] {}'.format(d['candidate'], d['reference'], d['question'])
            inputs = tokenizer(text, return_tensors="pt")
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
            inputs=inputs.to(device)
            with torch.no_grad():
                logits= model(**inputs).logits                    
            predicted_class_id = logits.argmax().item()
            
            if predicted_class_id == int(d['score']):
                cnt+=1

        print('Evaluation on {} data: {}'.format(len(val),cnt/len(val)))
        
        trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=tokenized_val,
        compute_metrics=compute_metrics,
        )
        print(trainer.evaluate())
    
    elif args.model_name=="f1":
        cnt=0
        for d in val:
            f1=f1_threshold(d['candidate'], d['reference'])
    elif args.model_name=='bertscore':
        pass
    
    
if __name__=='__main__':
    args=get_args()
    main(args)