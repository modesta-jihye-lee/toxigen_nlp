import torch
import numpy as np
import json
import csv
import os
import sys
import hydra
import random
import itertools

from datasets import load_dataset
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
from transformers import logging

from scipy.special import softmax
from pathlib import Path
from omegaconf import OmegaConf
from tqdm import tqdm
from typing import List, Dict
from collections import Counter


####################################################################################################
### DATASET INFORMATION
# new: toxigen dataset with predicted labels from tweet_eval model and (naively) expected labels
# unexp: toxigen dataset where predicted labels and expected labels differ
# exp: toxigen dataset where predicted labels and expected labels equal
# new = unexp + exp

# cf_pos: unexp dataset where exp_label = 'positive' if text directly refers to minority groups
# cf_neut: unexp dataset where exp_label = 'neutral' if text directly refers to minority groups
# relabel: unexp dataset where exp_label = manually decided label
# relabel_aug: relabel + exp
####################################################################################################


def load_data(cfg):
    print('='*10, end=' ')
    print('Load toxigen annotated')
    access_token = '' # you need to put your own huggingface access token here
    dataset = load_dataset(cfg.preprocess.data, use_auth_token=access_token, split='train')
    print('length of loaded dataset: ', len(dataset))
    return dataset


def save_json(dataset, path):
    print('='*10, end=' ')
    print(f'Save json ({path})')
    with open(path, 'w') as f:
        for x in dataset:
            f.write(json.dumps(x) + '\n')


@torch.no_grad()
def infer(cfg, dataset, device): # create custom dataset with inferred results
    print('='*10, end=' ')
    print('Infer sentiment')

    # load latest tweeteval model
    MODEL = cfg.preprocess.model
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    config = AutoConfig.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    model.to(device)
    new_dataset: List[Dict] = []
    unexp_dataset: List[Dict] = []
    
    for data in tqdm(dataset):
        new_data = {k: v for k, v in data.items()}

        exp_label = '' # expected label
        # 1) factual to neutral
        if new_data['factual?'] == 'factual':
            exp_label = 'neutral'
        # 2) positive stereotyping to positive
        elif new_data['stereotyping'] == 'uses positive stereotyping':
            exp_label = 'positive'
        # 3) toxicity to sentiment
        elif new_data['toxicity_human'] <= 1.6:
            exp_label = 'positive'
        elif new_data['toxicity_human'] <= 3.3:
            exp_label = 'neutral'
        else:
            exp_label = 'negative'
        new_data['exp_label'] = exp_label

        inputs = tokenizer(data['text'], return_tensors='pt')
        inputs.to(device)
        outputs = model(**inputs)
        scores = outputs[0][0].detach().cpu().numpy()
        scores = softmax(scores)

        ranking = np.argsort(scores)
        ranking = ranking[::-1]
        classes = dict()
        for i in range(scores.shape[0]):
            l = config.id2label[ranking[i]]
            s = scores[ranking[i]]
            classes[l] = np.round(float(s), 4)
        new_data['pred_label'] = config.id2label[ranking[0]] # predicted label from model
        new_data['classes'] = classes
        new_dataset.append(new_data)
        if new_data['exp_label'] != new_data['pred_label']:
            unexp_dataset.append(new_data)
    
    return new_dataset, unexp_dataset


def split_dataset(cfg): # split train test dataset (80:20)
    print('='*10, end=' ')
    print(f'Split train / test dataset from {cfg.preprocess.paths.relabel}')

    dataset = load_dataset('json',
            data_files={os.path.abspath(cfg.preprocess.paths.relabel)})['train'] # put custom data path 
    dataset = sorted(dataset, key=lambda x: x['exp_label'])
    grouped = itertools.groupby(dataset, key=lambda x: x['exp_label'])
    train_ds = []
    eval_ds = []
    for key, group in grouped:
        group = list(group)
        random.shuffle(group)
        split = int(len(group) * 0.8)
        train_ds.extend(group[:split])
        eval_ds.extend(group[split:])

    print(f'train_dataset: {len(train_ds)}, eval_dataset: {len(eval_ds)}')
    save_json(train_ds, cfg.preprocess.paths.cf_neut_train)
    save_json(eval_ds, cfg.preprocess.paths.cf_neut_test)


def create_expected(cfg): # create expected dataset
    print('='*10, end=' ')
    print('Save expected dataset')

    new_dataset = load_dataset('json',
            data_files={os.path.abspath(cfg.preprocess.paths.new)})['train']
    exp_dataset = []
    for d in new_dataset:
        if d['pred_label'] == d['exp_label']:
            exp_dataset.append({k: v for k, v in d.items()})

    save_json(exp_dataset, cfg.preprocess.paths.exp)


def create_counterfactual(cfg): # create counterfactual dataset
    print('='*10, end=' ')
    print('Save counterfactual dataset')
    pos_dataset = []
    neut_dataset = []

    with open(cfg.preprocess.paths.unexp, 'r') as input_file:
        json_dataset = [json.loads(line) for line in input_file]

    for data in json_dataset:
        pos_data = {k: v for k, v in data.items()}
        neut_data = {k: v for k, v in data.items()}
        if type(data['predicted_group']) == type([]):
            for d in data['predicted_group']:
                if d.find('the text directly references') != -1:
                    pos_data['exp_label'] = 'positive'
                    neut_data['exp_label'] = 'neutral'
                    break
        elif type(data['predicted_group']) == type(''):
            if data['predicted_group'].find('the text directly references') != -1:
                pos_data['exp_label'] = 'positive'
                neut_data['exp_label'] = 'neutral'
        pos_dataset.append(pos_data)
        neut_dataset.append(neut_data)

    save_json(pos_dataset, cfg.preprocess.paths.cf_pos)
    save_json(neut_dataset, cfg.preprocess.paths.cf_neut)


def create_relabel(cfg): # create manually relabelled dataset
    print('='*10, end=' ')
    print('Save relabelled dataset')
    relabel_dataset = []

    with open(cfg.preprocess.paths.unexp, 'r') as input_file:
        json_dataset = [json.loads(line) for line in input_file]

    with open(cfg.preprocess.paths.man_label_csv, 'r', encoding='utf-8', newline='') as output_file:
        csvreader = csv.reader(output_file)
        for c, data in zip(csvreader, json_dataset):
            if int(c[0]) == 3:
                continue
            new_data = {k: v for k, v in data.items()}
            if int(c[0]) == 0:
                val = 'negative'
            elif int(c[0]) == 1:
                val = 'positive'
            elif int(c[0]) == 2:
                val = 'neutral'
            new_data['exp_label'] = val
            relabel_dataset.append(new_data)

    save_json(relabel_dataset, cfg.preprocess.paths.relabel)


def print_tweeteval_data_info(): # check tweeteval dataset label count and distribution
    print('='*10, end=' ')
    print('Load tweeteval dataset')
    dataset = load_dataset('tweet_eval', 'sentiment')
    train_ds = dataset['train']
    test_ds = dataset['test']
    train_count = dict(Counter(train_ds['label'])) # 0: negative, 1: neutral, 2: positive
    test_count = dict(Counter(test_ds['label']))
    print(f'total train dataset: {len(train_ds)}, label count: {train_count}, \
        negative: {train_count[0]/len(train_ds):.2f}, neutral: {train_count[1]/len(train_ds):.2f}, positive: {train_count[2]/len(train_ds):.2f}')
    print(f'total test dataset: {len(test_ds)}, label count: {test_count}, \
        negative: {test_count[0]/len(test_ds):.2f}, neutral: {test_count[1]/len(test_ds):.2f}, positive: {test_count[2]/len(test_ds):.2f}')
    # total train dataset: 45615, label count: {2: 17849, 1: 20673, 0: 7093}, negative: 0.16, neutral: 0.45, positive: 0.39
    # total test dataset: 12284, label count: {1: 5937, 2: 2375, 0: 3972}, negative: 0.32, neutral: 0.48, positive: 0.19


def print_custom_data_info(cfg): # check custom dataset label count and distribution
    print('='*10, end=' ')
    print(f'Load {cfg.preprocess.paths.relabel} dataset')
    _ds = load_dataset('json',
            data_files={os.path.abspath(cfg.preprocess.paths.relabel)})['train'] # put custom data path 
    ds = []
    for d in _ds:
        if d['exp_label'] == 'negative':
            label = 0
        elif d['exp_label'] == 'neutral':
            label = 1
        elif d['exp_label'] == 'positive':
            label = 2
        ds.append(label)
    ds_count = dict(Counter(ds)) # 0: negative, 1: neutral, 2: positive
    print(f'total dataset: {len(ds)}, label count: {ds_count}, \
        negative: {ds_count[0]/len(ds):.2f}, neutral: {ds_count[1]/len(ds):.2f}, positive: {ds_count[2]/len(ds):.2f}')


def json_to_csv(cfg):
    csv_dataset = []
    json_dataset = []

    with open(cfg.preprocess.paths.unexp, 'r') as input_file:
        json_dataset = [json.loads(line) for line in input_file]

    for d in json_dataset:
        csv_dataset.append([val for val in d.values()])
        
    with open(cfg.preprocess.paths.unexp_csv, 'w', encoding='utf-8', newline='') as output_file:
        csvwriter = csv.writer(output_file, delimiter='\t')
        csvwriter.writerow([key for key in json_dataset[0].keys()])
        for c in csv_dataset:
            csvwriter.writerow(c)


def main(cfg) -> None:
    print(OmegaConf.to_yaml(cfg))
    logging.set_verbosity_error()
    device = torch.device('cuda', cfg.device) if cfg.device > -1 else torch.device('cpu')

    raw_dataset = load_data(cfg)
    new_dataset, unexp_dataset = infer(cfg, raw_dataset, device)
    save_json(new_dataset, cfg.preprocess.paths.new)
    save_json(unexp_dataset, cfg.preprocess.paths.unexp)

    create_expected(cfg)
    create_counterfactual(cfg)
    create_relabel(cfg)
    split_dataset(cfg)


if __name__ == '__main__':
    root_path = Path(__file__).resolve().parent
    cfg_path = 'confs/basic.yaml'
    # cfg_path = sys.argv[1]
    # sys.argv = sys.argv[1:] # NOTE: without this line, hydra reports mismatched input, expecting ID
    cfg_path = root_path / cfg_path

    if cfg_path.is_file():
        lst = str(cfg_path).split('/')
        cfg_dir = cfg_path.parent
        cfg_file = cfg_path.name
    else: 
        raise Exception('Missing training config......')

    print(cfg_dir, cfg_file)
    hydra.main(config_path=cfg_dir, config_name=cfg_file)(main)()