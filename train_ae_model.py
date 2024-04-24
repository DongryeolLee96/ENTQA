import argparse
import jsonlines
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import DataCollatorWithPadding, TrainingArguments, Trainer, EarlyStoppingCallback
from datasets import Dataset, load_dataset
import torch
import numpy as np
import evaluate
import wandb
from transformers.trainer_utils import set_seed

def compute_metrics(eval_pred):
    accuracy=evaluate.load('accuracy')
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions = predictions, references = labels)
    
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--train_dir', required=True)
    parser.add_argument('--validation_dir', required=True)
    parser.add_argument('--batch_size', required=True, type=int)
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--epoch', type=int)


    args =parser.parse_args()
    return args
def main(args):
    set_seed(1001)
    print(args)
    # data load
    wandb.init(project='NEM', entity='donaldo9603')
    wandb.config.update(args)
    train=[]
    val=[]
    with jsonlines.open(args.train_dir) as f:
        for line in f:
            train.append(line)
    with jsonlines.open(args.validation_dir) as f:
        for line in f:
            val.append(line)
    
    # List to Dataset conversion
    val_data={'id': [i for i in range(len(val))],
              'text': ['{} [SEP] {} [SEP] {}'.format(d['candidate'], d['reference'], d['question']) for d in val],
              'label': [1 if d['score']==1.0 else 0 for d in val]}
    train_data={'id': [i for i in range(len(train))],
              'text': ['{} [SEP] {} [SEP] {}'.format(d['candidate'], d['reference'], d['question']) for d in train],
              'label': [1 if d['score']==1.0 else 0 for d in train]}
    
    # val_data={'id': [i for i in range(len(val))],
    #           'text': ['{} [SEP] {}'.format(d['candidate'], d['reference']) for d in val],
    #           'label': [1 if d['score']==1.0 else 0 for d in val]}
    # train_data={'id': [i for i in range(len(train))],
    #           'text': ['{} [SEP] {}'.format(d['candidate'], d['reference']) for d in train],
    #           'label': [1 if d['score']==1.0 else 0 for d in train]}
    
    val_data=Dataset.from_dict(val_data)
    train_data=Dataset.from_dict(train_data)
    # train_data=train_data.shuffle(seed=1001)
    id2label = {0: "NEGATIVE", 1: "POSITIVE"}
    label2id = {"NEGATIVE": 0, "POSITIVE": 1}
        
    # Preprocessing
    if args.model_name=='bert':
        tokenizer=AutoTokenizer.from_pretrained('bert-base-uncased')    
        model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2, id2label=id2label, label2id=label2id)
    elif args.model_name == 'roberta':
        tokenizer=AutoTokenizer.from_pretrained('roberta-base')
        model = AutoModelForSequenceClassification.from_pretrained('roberta-base', num_labels=2, id2label=id2label, label2id=label2id)
    elif args.model_name =='microsoft/deberta-v3-base':
        tokenizer=AutoTokenizer.from_pretrained('microsoft/deberta-v3-base')
        model = AutoModelForSequenceClassification.from_pretrained('microsoft/deberta-v3-base', num_labels=2, id2label=id2label, label2id=label2id)   
    elif args.model_name =='microsoft/deberta-v3-base':
        tokenizer=AutoTokenizer.from_pretrained('microsoft/deberta-v3-large')
        model = AutoModelForSequenceClassification.from_pretrained('microsoft/deberta-v3-large', num_labels=2, id2label=id2label, label2id=label2id)     
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer)
    
    def preprocess_function(example):
        return tokenizer(example['text'], truncation=True)
    
    
    
    
    tokenized_train=train_data.map(preprocess_function, batched=True)
    tokenized_val=val_data.map(preprocess_function, batched=True)
    
    # Training Argument definition
    training_args= TrainingArguments(
        output_dir= args.output_dir,
        learning_rate=1e-4,
        per_device_eval_batch_size=args.batch_size,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epoch,
        weight_decay=0.01,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True
    )
    trainer = Trainer(
        model=model,
        args = training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks = [EarlyStoppingCallback(early_stopping_patience=2)]
    )
    trainer.train()
if __name__=='__main__':
    args=get_args()
    main(args)