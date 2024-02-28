"""Main module to load and train the model. This should be the program entry point."""
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

#import constants
from chromexpress.constants import (
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
#from scipy import stats
from torchmetrics.functional import pearson_corrcoef

from chromoformer.chromoformer.util import(seed_everything,
                                           EarlyStopping,
                                           LRScheduler)

#TO ADD - cell, mark and wandb details if using
CELL='E003'
MARK='h3k4me1
track_wandb=True
wandb_entity=''
wandb_project=''


#leading and trailing whitespace
CELL=CELL.strip()
#assert it's a valid choice
assert CELL in SAMPLE_IDS, f"{CELL} not valid. Must choose valid cell: {SAMPLE_IDS}"

MARK=MARK.strip()
#assert it's a valid choice
assert MARK in ASSAYS, f"{MARK} not valid. Must choose valid assay: {ASSAYS}"


if track_wandb:
    #track models
    import wandb

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

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

# Model specifics - similar to https://www.nature.com/articles/s42256-022-00570-9
batch_size = 64
n_epochs = 100#10
init_learning_rate = 0.001#3e-5
lr_decay_factor = 0.2
lr_patience = 3
es_patience = 12
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
        #set values for early stopping
        min_val_mse = 1_000_000
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
                                  marks=features,train_dir=train_dir,train_meta=train_meta)
        val_dataset = Roadmap3D(cell, val_genes, i_max, window_size, window_size,
                                marks=features,train_dir=train_dir,train_meta=train_meta)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, 
                                                   num_workers=8, shuffle=True, drop_last=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, 
                                                 num_workers=8)

        model = Chromoformer(
            n_feats=len(features), embed_n_layers=embed_n_layers, #1 feature input
            embed_n_heads = embed_n_heads, embed_d_model=embed_d_model, 
            embed_d_ff=embed_d_ff, pw_int_n_layers=pw_int_n_layers, 
            pw_int_n_heads=pw_int_n_heads, pw_int_d_model=pw_int_d_model, 
            pw_int_d_ff=pw_int_d_ff,reg_n_layers=reg_n_layers, 
            reg_n_heads=reg_n_heads, reg_d_model=reg_d_model, 
            reg_d_ff=reg_d_ff, head_n_feats=head_n_feats
        )
        model.cuda()

        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(init_learning_rate))
        
        #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=gamma)
        # Initialising learning rate scheduler
        lr_scheduler = LRScheduler(optimizer, patience=lr_patience, factor=lr_decay_factor)
        #Initialising early stopping
        early_stopping = EarlyStopping(patience = es_patience)

        optimizer.zero_grad()
        optimizer.step()
        #----
        #save to wandb if ind = 1
        if fold==1 and track_wandb:
            readable_features = '-'.join(features)
            wandb.init(
                name=f'chromoformer_{cell}_{readable_features}_{fold}',
                entity=f"{wandb_entity}",
                project=f"{wandb_project}",
            )
       
        for epoch in range(0, n_epochs):
            print('epoch',epoch)
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
                y = d['log2RPKM'].float().unsqueeze(1)
                loss = criterion(out,y)
                
                loss.backward()
                optimizer.step()

                loss = loss.detach().cpu().item()
                running_loss += loss

                train_out.append(out.detach().cpu())
                train_label.append(d['log2RPKM'].unsqueeze(1).cpu())
                
                #save training error at end of epoch
                if batch == len(train_loader):
                    print('batch == len(train_loader):')
                    print('batch',batch)
                    batch_loss = running_loss / len(train_loader)

                    train_out, train_label = map(torch.cat, (train_out, train_label))
                    #train_score = train_out.softmax(axis=1)[:, 1]
                    #train_pred = train_out.argmax(axis=1)
                    
                    batch_mse = metrics.mean_squared_error(train_label, train_out)
                    
                    #batch_corr = stats.pearsonr(train_label, train_out)
                    batch_corr = pearson_corrcoef(train_label[:,0], train_out[:,0])
                    bar.set_description(f'E{epoch} {batch_loss:.4f}, lr={get_lr(optimizer)}, mse={batch_mse}, corr={batch_corr}')

                    running_loss = 0.0
                    train_out, train_label = [], []
                        

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

                    val_label.append(d['log2RPKM'].unsqueeze(1).cpu())

            val_out = torch.cat(val_out)
            val_label = torch.cat(val_label)

            val_loss = criterion(val_out, val_label)

            # Metrics.
            #val_label = val_label.numpy()
            #val_score = val_out.softmax(axis=1)[:, 1].numpy()
            #val_pred = val_out.argmax(axis=1).numpy()
            
            val_mse = metrics.mean_squared_error(val_label, val_out)
            val_corr = pearson_corrcoef(val_label[:,0], val_out[:,0])
            
            print(f'Validation loss={val_loss:.4f}, mse={val_mse}, corr={val_corr}')
            if fold==1 and track_wandb:
                wandb.log({
                    'loss': batch_loss,
                    'mse': batch_mse,
                    'correlation': batch_corr,
                    'lr': get_lr(optimizer),
                    'val_loss': val_loss,
                    'val_mse': val_mse,
                    'val_correlation': val_corr,
                    'epoch': epoch,
                })
            if val_mse < min_val_mse:
                min_val_mse = val_mse   
                ckpt = {
                    'net': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'last_val_loss': val_loss,
                    'last_val_mse': val_mse,
                    'last_val_corr': val_corr,
                    'val_act': val_label,
                    'val_pred': val_out,
                }
                torch.save(ckpt, f"{MOD_SAVE_PATH}/chromoformer_{cell}_{'-'.join(features)}_kfold{fold}")
            #learning rate
            lr_scheduler(val_loss)
            #early stopping
            early_stopping(val_loss)
            if early_stopping.early_stop:
                break
            #scheduler.step()
        #save result after 100 epochs or es
        print('epoch',epoch)
        print(aa)

