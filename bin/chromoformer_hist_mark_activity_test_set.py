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
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import pearson_corrcoef
#data loading imports
from chromoformer.chromoformer.data import Roadmap3D
#model arch
from chromoformer.chromoformer.net import Chromoformer

#params
pred_resolution = 100
#seed
seed = 123
#regression problem
y_type = 'log2RPKM'
# window to be considered for the prediction of gene expression
window_size = 40_000

#paths
SAVE_PATH = pathlib.Path("./model_results")
MOD_SAVE_PATH = pathlib.Path("./model_results/models")
PRED_PATH = pathlib.Path("./model_results/predictions")
PRED_PATH.mkdir(parents=True, exist_ok=True)
train_dir = PROJECT_PATH/'chromoformer'/'preprocessing'
train_meta = train_dir / 'train.csv'


cells_ = []
assays_ = []
activity_ = []
expression_ = []
exp_label_ = []
fold_ = []
with torch.no_grad():
    for assay_i in ASSAYS:
        print(assay_i)
        for cell_i in SAMPLE_IDS:
            print(cell_i)
            # 2. --- Dataset parameters -------------------------------
            #use k-fold cross-validation to retrain each model k times and hold out 
            train_dir = PROJECT_PATH/'chromoformer'/'preprocessing'
            train_meta = train_dir / 'train.csv'
            meta = pd.read_csv(train_meta) \
                .sample(frac=1, random_state=seed) \
                .reset_index(drop=True) # load and shuffle.

            #filter metadat to cell type of interest
            meta = meta[meta.eid == cell_i]

            # Split genes into two sets (train/val).
            genes = set(meta.gene_id.unique())
            n_genes = len(genes)
            #get data for folds separated
            qs = [
                meta[meta.split == 1].gene_id.tolist(),
                meta[meta.split == 2].gene_id.tolist(),
                meta[meta.split == 3].gene_id.tolist(),
                meta[meta.split == 4].gene_id.tolist(),
            ]
            #loop through folds
            for ind,fold in enumerate([x+1 for x in range(k_fold)]):
                print('K fold: ',fold)
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
                #data loaders ----
                test_dataset = Roadmap3D(cell_i, test_genes,w_prom=window_size, w_max=window_size,
                                         marks = [assay_i],train_dir=train_dir,train_meta=train_meta)
                test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, 
                                                          num_workers=8, shuffle=False, drop_last=False)
                for item in test_loader:
                    for k, v in item.items():
                        item[k] = v.cuda()
                    #get cell
                    cells_.append(cell_i)
                    #get mark
                    assays_.append(assay_i)
                    #get fold
                    fold_.append(fold)
                    #get expression
                    expression_.append(item['log2RPKM'].cpu().float().unsqueeze(1).numpy()[0][0])
                    exp_label_.append(item['label'].cpu().float().unsqueeze(1).numpy()[0][0])
                    #get hist mark activity
                    activity_.append(item[f'x_p_{pred_resolution}'].cpu().numpy().mean())    

#all res kept in order so can just make df and save
res = pd.DataFrame({"cell":cells_,
                    "assay":assays_,
                    "fold":fold_,
                    "log2RPKM":expression_,
                    "exp_label":exp_label_,
                    "hist_mark_activity":activity_})
#save res
res.to_csv(str(PROJECT_PATH/'model_results/checkpoints/agg_hist_exp_chromo_test_folds.csv'), 
           sep='\t',index=False)