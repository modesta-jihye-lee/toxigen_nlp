#!/bin/bash
pip install wandb && pip install evaluate &&
pip install --upgrade transformers && pip install --upgrade datasets &&
python3 /code/train.py confs/basic.yaml hydra.run.dir=/outputs/relabel_aug_lr_2e-6