# How much are NLP models stereotyped? : A Case Study with Multi-Class Sentiment Classification 

## Motivation

Previous works have pointed out that stereotypes affect NLP systems by making false correlations.  
For example, most sentiment classifiers assign more negative emotions with African Americans and toxicity classifiers incorrectly flag texts with minority groups.  

We aim to break these false correlations in multi-class sentiment classification using ToxiGen dataset (Hartvigsen et al. 2022).

## Task

Sentiment Analysis from SemEval-2017 Task 4: Sentiment Analysis in Twitter (Rosenthal et al. 2017) 

## Dataset

ToxiGen from ToxiGen: A Large-Scale Machine-Generated Dataset for Adversarial and Implicit Hate Speech Detection (Hartvigsen et al. 2022)

## Baseline Model

TimeLM from TimeLMs: Diachronic Language Models from Twitter (Loureiro et al. 2022)

## Fine-tuned Model

Fintuned TimeLM using manually relabelled ToxiGen

## Command Script

Preprocess: CUDA_VISIBLE_DEVICES=0 ./scripts/docker_run.sh python3 /code/preprocess.py 

Train: CUDA_VISIBLE_DEVICES=0 ./scripts/docker_run.sh ./scripts/train.sh 

Eval: CUDA_VISIBLE_DEVICES=0 ./scripts/docker_run.sh python3 /code/eval.py 
