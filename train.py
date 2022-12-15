import numpy as np
import torch
import evaluate
import os
import sys
import hydra
from pathlib import Path
from omegaconf import OmegaConf
from transformers import logging

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from datasets import load_dataset, Dataset, concatenate_datasets

MODEL_NAME = 'cardiffnlp/twitter-roberta-base-sentiment-latest'
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
metric = evaluate.load('recall')  # default metric for sentiment dataset is recall (macro)

def tokenize_text(examples):
    return tokenizer(examples['text'], padding='max_length', max_length=512, truncation=True)


def tokenize_label(examples):
    new_examples = []
    for example in examples:
        new_ex = {'text': example['text']}
        if example['labels'] == 'negative':
            new_ex['labels'] = 0
        elif example['labels'] == 'neutral':
            new_ex['labels'] = 1
        elif example['labels'] == 'positive':
            new_ex['labels'] = 2
        new_examples.append(new_ex)
    dataset = Dataset.from_list(new_examples)
    return dataset


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels, average='macro')


def load_data(cfg):
    print('='*10, end=' ')
    print(f'Load data: {cfg.preprocess.paths.relabel_aug_train}, {cfg.preprocess.paths.relabel_aug_test}')
    
    train_ds = load_dataset('json',
                data_files={os.path.abspath(cfg.preprocess.paths.relabel_aug_train)})['train']
    eval_ds = load_dataset('json',
                data_files={os.path.abspath(cfg.preprocess.paths.relabel_aug_test)})['train']

    train_ds = train_ds.rename_column('exp_label', 'labels')
    train_ds = train_ds.remove_columns(['target_group', 'factual?', 'ingroup_effect', 'lewd', 'framing', \
        'predicted_group', 'stereotyping', 'intent', 'toxicity_ai', 'toxicity_human', \
        'predicted_author', 'actual_method', 'pred_label', 'classes'])
    train_ds = tokenize_label(train_ds)
    
    eval_ds = eval_ds.rename_column('exp_label', 'labels')
    eval_ds = eval_ds.remove_columns(['target_group', 'factual?', 'ingroup_effect', 'lewd', 'framing', \
        'predicted_group', 'stereotyping', 'intent', 'toxicity_ai', 'toxicity_human', \
        'predicted_author', 'actual_method', 'pred_label', 'classes'])
    eval_ds = tokenize_label(eval_ds)

    return train_ds, eval_ds


def main(cfg) -> None:

    print(OmegaConf.to_yaml(cfg))
    logging.set_verbosity_error()
    device = torch.device('cuda', cfg.device) if cfg.device > -1 else torch.device('cpu')

    train_ds, eval_ds = load_data(cfg)
    tok_train_ds = train_ds.map(tokenize_text, batched=True)
    tok_eval_ds = eval_ds.map(tokenize_text, batched=True)
    
    tok_train_ds.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
    tok_eval_ds.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)
    model.to(device)
    training_args = TrainingArguments(**cfg.train)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tok_train_ds,
        eval_dataset=tok_eval_ds,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.create_model_card()
    trainer.save_model('saved_model')

if __name__ == '__main__':

    root_path = Path(__file__).resolve().parent
    cfg_path = sys.argv[1]
    sys.argv = sys.argv[1:] # NOTE: without this line, hydra reports mismatched input, expecting ID
    cfg_path = root_path / cfg_path

    if cfg_path.is_file():
        lst = str(cfg_path).split('/')
        cfg_dir = cfg_path.parent
        cfg_file = cfg_path.name
    else: 
        raise Exception('Missing training config......')

    print(cfg_dir, cfg_file)
    hydra.main(config_path=cfg_dir, config_name=cfg_file)(main)()