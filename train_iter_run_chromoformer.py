"""Main module to load and train the model. This should be the program entry point."""

##TO DO:
#Update data loader to output just y_type
#Update chromoformer to also wrok with just y_type
#Update metrics being monitored to match otr model - MSE & Pearson R

#generic imports
import os
import pathlib
import random
from datetime import datetime
import time
import numpy as np
import pandas as pd
import math
import argparse
import itertools
import os.path

#track models
import wandb

#import constants
from epi_to_express.constants import (
    CHROM_LEN, 
    CHROMOSOMES, 
    SAMPLES,
    SAMPLE_IDS,
    CHROMOSOME_DATA,
    SRC_PATH,
    ASSAYS,
    PROJECT_PATH)

#model imports
#data loading imports
from epi_to_express.utils import pearsonR
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

import numpy as np
import pandas as pd

from chromoformer.chromoformer.data import Roadmap3D
from chromoformer.chromoformer.net import Chromoformer
#from collections import defaultdict
from tqdm import tqdm
from pathlib import Path
from sklearn import metrics
from scipy import stats

from chromoformer.chromoformer.util import seed_everything

#pass inputs
# argv
def get_args():
    parser = argparse.ArgumentParser(description="train")
    parser.add_argument('-c', '--CELL', default='', type=str, help='Cell to train in')
    parser.add_argument('-m', '--MARK', default='', type=str, help='Mark to train on')
    args = parser.parse_args()
    return args

args=get_args()

CELL=args.CELL
#leading and trailing whitespace
CELL=CELL.strip()
#assert it's a valid choice
assert CELL in SAMPLE_IDS, f"{CELL} not valid. Must choose valid cell: {SAMPLE_IDS}"

MARK=args.MARK.lower()
MARK=MARK.strip()
#assert it's a valid choice
assert MARK in ASSAYS, f"{MARK} not valid. Must choose valid assay: {ASSAYS}"

print("---------------------------------")
print(CELL)
print(MARK)
print("---------------------------------")

seed_everything(101)

SAVE_PATH = pathlib.Path("./model_results")
SAVE_PATH.mkdir(parents=True, exist_ok=True)

MOD_SAVE_PATH = pathlib.Path("./model_results/models")
MOD_SAVE_PATH.mkdir(parents=True, exist_ok=True)

# 1. --- SETUP PARAMETERS ------------------------------------------------

#what will be used to predict expression
features = [MARK]
#what cell will we predict in
cell = CELL
# 1 Mb of the assay will be considered for the prediction of gene expression
window_size = 40_000
#number of k-fold cross validation
k_fold = 4
#seed
seed = 123
#regression problem
y_type = 'log2RPKM'

# Model specifics - similar to https://www.nature.com/articles/s42256-022-00570-9
batch_size = 64
n_epochs = 10

lr = 3e-5
gamma = 0.87
i_max = 8
embed_n_layers = 1
embed_n_heads = 2
embed_d_model = 128
embed_d_ff = 128
pw_int_n_layers = 2
pw_int_n_heads = 2
pw_int_d_model = 128
pw_int_d_ff = 256
reg_n_layers = 6
reg_n_heads = 8
reg_d_model = 256
reg_d_ff = 256
head_n_feats = 128

# 2. --- Dataset parameters -------------------------------
train_dir = PROJECT_PATH/'chromoformer'/'preprocessing'
train_meta = train_dir / 'train.csv'
meta = pd.read_csv(train_meta) \
    .sample(frac=1, random_state=seed) \
    .reset_index(drop=True) # load and shuffle.

#filter metadat to cell type of interest
meta = meta[meta.eid == CELL]

# Split genes into two sets (train/val).
genes = set(meta.gene_id.unique())
n_genes = len(genes)
print('Target genes:', len(genes))

#get data for folds separated
qs = [
    meta[meta.split == 1].gene_id.tolist(),
    meta[meta.split == 2].gene_id.tolist(),
    meta[meta.split == 3].gene_id.tolist(),
    meta[meta.split == 4].gene_id.tolist(),
]

# 3. --- Train models -------------------------------
# loop through each fold
for ind,fold in enumerate([x+1 for x in range(k_fold)]):
    if not os.path.exists(str(f"{MOD_SAVE_PATH}/chromoformer_{cell}_{'-'.join(features)}_kfold{fold}")):
        print(f"K-fold Cross-Validation - blind test: {ind}")
        #get fold specific data ----
        train_genes = qs[(fold + 0) % 4] + qs[(fold + 1) % 4] + qs[(fold + 2) % 4]
        val_genes = qs[(fold + 3) % 4]

        #split val_genes in two to get validation and test set
        # train/val split by chrom so do the same for val test
        val_test_genes = val_genes
        val_test_chrom = list(set(meta[meta.gene_id.isin(val_test_genes)]['chrom']))
        val_chrom = val_test_chrom[0:len(val_test_chrom)//2]
        test_chrom = val_test_chrom[len(val_test_chrom)//2:len(val_test_chrom)]
        val_genes = meta[meta.gene_id.isin(val_test_genes) & meta.chrom.isin(val_chrom)]['gene_id'].tolist()
        test_genes = meta[meta.gene_id.isin(val_test_genes) & meta.chrom.isin(test_chrom)]['gene_id'].tolist()
        #----
        # 2. --- Dataset parameters -------------------------------
        train_dataset = Roadmap3D(cell, train_genes, i_max, window_size, window_size,
                                  marks=features,train_dir=train_dir)
        val_dataset = Roadmap3D(cell, val_genes, i_max, window_size, window_size,
                                marks=features,train_dir=train_dir)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, 
                                                   num_workers=8, shuffle=True, drop_last=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, 
                                                 num_workers=8)

        model = Chromoformer(
            1, embed_n_layers, embed_n_heads, embed_d_model, embed_d_ff, #1 feature input
            pw_int_n_layers, pw_int_n_heads, pw_int_d_model, pw_int_d_ff,
            reg_n_layers, reg_n_heads, reg_d_model, reg_d_ff, head_n_feats,
        )
        model.cuda()

        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr))
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=gamma)

        optimizer.zero_grad()
        optimizer.step()
        #----
        #save to wandb if ind = 1
        if fold==1:
            readable_features = '-'.join(features)
            wandb.init(
                name=f'chromoformer_{cell}_{readable_features}_{fold}',
                entity="al-murphy",
                project="Epi_to_Express",
            )
        
        best_val_auc = 0
        for epoch in range(1, n_epochs):

            # Prepare train.
            bar = tqdm(enumerate(train_loader, 1), total=len(train_loader))
            running_loss = 0.0
            train_out, train_label = [], []

            # Train.
            model.train()
            for batch, d in bar:
                for k, v in d.items():
                    d[k] = v.cuda()

                optimizer.zero_grad()

                out = model(
                    d['x_p_2000'], d['pad_mask_p_2000'], d['x_pcre_2000'], 
                    d['pad_mask_pcre_2000'], d['interaction_mask_2000'],
                    d['x_p_500'], d['pad_mask_p_500'], d['x_pcre_500'], 
                    d['pad_mask_pcre_500'], d['interaction_mask_2000'],
                    d['x_p_100'], d['pad_mask_p_100'], d['x_pcre_100'], 
                    d['pad_mask_pcre_100'], d['interaction_mask_2000'],
                    d['interaction_freq'],
                )
                loss = criterion(out, d['label'])

                loss.backward()
                optimizer.step()

                loss = loss.detach().cpu().item()
                running_loss += loss

                train_out.append(out.detach().cpu())
                train_label.append(d['label'].cpu())

                if batch % 10 == 0:
                    batch_loss = running_loss / 10.

                    train_out, train_label = map(torch.cat, (train_out, train_label))
                    train_score = train_out.softmax(axis=1)[:, 1]
                    train_pred = train_out.argmax(axis=1)
                    
                    batch_mse = metrics.mean_squared_error(train_label, train_pred)
                    batch_corr = stats.pearsonr(train_label, train_score)
                    
                    bar.set_description(f'E{epoch} {batch_loss:.4f}, lr={get_lr(optimizer)}, mse={batch_mse:.4f}, corr={batch_corr:.4f}')

                    running_loss = 0.0
                    train_out, train_label = [], []
                    if fold==1:
                        wandb.log({
                            'loss': batch_loss,
                            'mse': batch_mse,
                            'correlation': batch_corr,
                            'lr': get_lr(optimizer),
                            'epoch': epoch,
                        })

            # Prepare validation.
            bar = tqdm(enumerate(val_loader, 1), total=len(val_loader))
            val_out, val_label = [], []

            # Validation.
            model.eval()
            with torch.no_grad():
                for batch, d in bar:
                    for k, v in d.items():
                        d[k] = v.cuda()

                    out = model(
                        d['x_p_2000'], d['pad_mask_p_2000'], d['x_pcre_2000'], 
                        d['pad_mask_pcre_2000'], d['interaction_mask_2000'],
                        d['x_p_500'], d['pad_mask_p_500'], d['x_pcre_500'], 
                        d['pad_mask_pcre_500'], d['interaction_mask_2000'],
                        d['x_p_100'], d['pad_mask_p_100'], d['x_pcre_100'], 
                        d['pad_mask_pcre_100'], d['interaction_mask_2000'],
                        d['interaction_freq'],
                    )
                    val_out.append(out.cpu())

                    val_label.append(d['label'].cpu())

            val_out = torch.cat(val_out)
            val_label = torch.cat(val_label)

            val_loss = criterion(val_out, val_label)

            # Metrics.
            val_label = val_label.numpy()
            val_score = val_out.softmax(axis=1)[:, 1].numpy()
            val_pred = val_out.argmax(axis=1).numpy()
            
            val_mse = metrics.mean_squared_error(val_label, val_pred)
            val_corr = stats.pearsonr(val_label, val_score)

            print(f'Validation loss={val_loss:.4f}, mse={val_mse:.4f}, corr={val_corr:.4f}')
            if fold==1:
                wandb.log({
                    'val_loss': val_loss,
                    'val_mse': val_mse,
                    'val_correlation': val_corr,
                    'val_epoch': epoch,
                })

            ckpt = {
                'net': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'last_val_loss': val_loss,
                'last_val_auc': val_auc,
                'val_score': val_score,
                'val_label': val_label,
            }
            torch.save(ckpt, f"{MOD_SAVE_PATH}/chromoformer_{cell}_{'-'.join(features)}_kfold{fold}")
            scheduler.step()
        if fold==1:
            wandb.summary.update({
                'last_val_loss': val_loss,
                'last_val_auc': val_auc,
            })

