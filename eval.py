import evaluate
import datasets
from datasets import load_dataset
from transformers import pipeline
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm.auto import tqdm

import torch
import csv
import sys
import hydra
from pathlib import Path
from omegaconf import OmegaConf
from transformers import logging


def baseline_eval(cfg): # evaluation for baseline model

    # load model and tokenizer
    MODEL_NAME = 'cardiffnlp/twitter-roberta-base-sentiment-latest'
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    
    # load dataset
    dataset = load_dataset('tweet_eval', 'sentiment', split='test')

    pipe = pipeline('text-classification', model=model, tokenizer=tokenizer, device=0)
    predictions = []
    references = []

    for out in tqdm(pipe(dataset['text'])):
        if out['label'] == 'negative':
            label = 0
        elif out['label'] == 'neutral':
            label = 1
        elif out['label'] == 'positive':
            label = 2
        predictions.append(label)
    references = list(dataset['label'])
    
    # load metrics (f1 and recall)
    f1_metric = evaluate.load('f1')
    f1_result = f1_metric.compute(average='macro', predictions=predictions, references=references)
    recall_metric = evaluate.load('recall')
    recall_result = recall_metric.compute(average='macro', predictions=predictions, references=references)
    
    print('================ Baseline Results')
    print('f1 result: ', f1_result)
    print('recall result: ', recall_result)


def custom_eval(cfg): # evaluation for custom finetuned model

    # load tokenizer
    MODEL_NAME = 'cardiffnlp/twitter-roberta-base-sentiment-latest'
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # load dataset
    dataset = load_dataset('tweet_eval', 'sentiment', split='test')

    # load model
    print('='*10, f'Loaded model: {cfg.eval.models.custom}')
    model = AutoModelForSequenceClassification.from_pretrained(cfg.eval.models.custom)

    pipe = pipeline('text-classification', model=model, tokenizer=tokenizer, device=0)
    predictions = []
    references = []

    for out in tqdm(pipe(dataset['text'])):
        if out['label'] == 'negative':
            label = 0
        elif out['label'] == 'neutral':
            label = 1
        elif out['label'] == 'positive':
            label = 2
        predictions.append(label)
    references = list(dataset['label'])
    
    # load metrics
    f1_metric = evaluate.load('f1')
    f1_result = f1_metric.compute(average='macro', predictions=predictions, references=references)
    recall_metric = evaluate.load('recall')
    recall_result = recall_metric.compute(average='macro', predictions=predictions, references=references)

    print('================ Custom Results')
    print('f1 result: ', f1_result)
    print('recall result: ', recall_result)


def export_to_csv(cfg): # if you want to export predictions to csv file

    # load tokenizer
    MODEL_NAME = 'cardiffnlp/twitter-roberta-base-sentiment-latest'
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    dataset = load_dataset('tweet_eval', 'sentiment', split='test')

    # load model
    base_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    relabel_aug_model = AutoModelForSequenceClassification.from_pretrained(cfg.eval.models.custom)
    base_pipe = pipeline('text-classification', model=base_model, tokenizer=tokenizer, device=0)
    relabel_aug_pipe = pipeline('text-classification', model=relabel_aug_model, tokenizer=tokenizer, device=0)
    
    infer = []
    for data in dataset:
        base_pred = base_pipe(data['text'])[0]['label']
        relabel_pred = relabel_aug_pipe(data['text'])[0]['label']
        gold_label = data['label']
        if gold_label == 0:
            label = 'negative'
        elif gold_label == 1:
            label = 'neutral'
        elif gold_label == 2:
            label = 'positive'
        infer.append({'text': data['text'], 'gold': label, 'baseline': base_pred, 'ours': relabel_pred})
    
    with open(cfg.preprocess.paths.infer_csv, 'w', encoding='utf-8', newline='') as output_file:
        csvwriter = csv.DictWriter(output_file, fieldnames=infer[0].keys(), delimiter='\t')
        csvwriter.writeheader()
        csvwriter.writerows(infer)


def text_eval(cfg): # you can put your own text, compare baseline and custom model

    # load model and tokenizer
    MODEL_NAME = 'cardiffnlp/twitter-roberta-base-sentiment-latest'
    base_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    custom_model = AutoModelForSequenceClassification.from_pretrained(cfg.eval.models.custom)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # load pipeline
    base_pipe = pipeline('text-classification', model=base_model, tokenizer=tokenizer, device=0)
    custom_pipe = pipeline('text-classification', model=custom_model, tokenizer=tokenizer, device=0)

    # example texts
    text1 = "The #Obama admin. has carried out a deliberately deceptive campaign accusing Iran of sending arms to #Houthis#yemenhttps://t.co/KYu02C73ww" # neutral (1)
    text2 = "#Israel fire updates: Major blaze breaks out near #Jerusalem, firebombing suspected..." # neutral (1)
    text3 = "#Putin has had it with #Obama and has decided to throw #America out of #Syria negotiationshttps://t.co/z4CBIDDU11" # neutral (1)
    text4 = "@user Wow...this is one deadly statement! Reminding me of the Greta Muhammad Ali." # positive (2)
    text5 = "¡Gobble gobble, idiot hookers! #Thanksgiving #ScreamQueens" # negative (0)
    text6 = "@user @user @user @user @user @user I'm getting tired of these people and their gun control" # negative (0)
    text7 = "The latest Smart LifeStyle Food! Thanks to @user #weightloss #nationalfastfoodday" # positive (2)
    text8 = "Can't wait to try this - Google Earth VR - this stuff really is the future of exploration...." # positive (2)

    print("="*10, "Baseline results")
    print("text: ", text1, ", predicted label: ", base_pipe(text1), ", gold label: neutral")
    print("text: ", text2, ", predicted label: ", base_pipe(text2), ", gold label: neutral")
    print("text: ", text3, ", predicted label: ", base_pipe(text3), ", gold label: neutral")
    print("text: ", text4, ", predicted label: ", base_pipe(text4), ", gold label: positive\n")

    print("="*10, "Relabel Augmented results")
    print("text: ", text1, ", predicted label: ", custom_pipe(text1), ", gold label: neutral")
    print("text: ", text2, ", predicted label: ", custom_pipe(text2), ", gold label: neutral")
    print("text: ", text3, ", predicted label: ", custom_pipe(text3), ", gold label: neutral")
    print("text: ", text4, ", predicted label: ", custom_pipe(text4), ", gold label: positive\n")

    print("="*10, "Baseline results")
    print("text: ", text5, ", predicted label: ", base_pipe(text5), ", gold label: negative")
    print("text: ", text6, ", predicted label: ", base_pipe(text6), ", gold label: negative")
    print("text: ", text7, ", predicted label: ", base_pipe(text7), ", gold label: positive")
    print("text: ", text8, ", predicted label: ", base_pipe(text8), ", gold label: positive\n")

    print("="*10, "Relabel Augmented results")
    print("text: ", text5, ", predicted label: ", custom_pipe(text5), ", gold label: negative")
    print("text: ", text6, ", predicted label: ", custom_pipe(text6), ", gold label: negative")
    print("text: ", text7, ", predicted label: ", custom_pipe(text7), ", gold label: positive")
    print("text: ", text8, ", predicted label: ", custom_pipe(text8), ", gold label: positive\n")


def main(cfg) -> None:

    print(OmegaConf.to_yaml(cfg))
    logging.set_verbosity_error()
    device = torch.device('cuda', cfg.device) if cfg.device > -1 else torch.device('cpu')

    baseline_eval(cfg)
    custom_eval(cfg)
    # text_eval(cfg)
    # export_to_csv(cfg)


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