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
with torch.no_grad():
    for assay_i in ASSAYS:
        print(assay_i)
        for cell_i in SAMPLE_IDS:
            print(cell_i)
            # 2. --- Dataset parameters -------------------------------
            train_dir = PROJECT_PATH/'chromoformer'/'preprocessing'
            train_meta = train_dir / 'train.csv'
            meta = pd.read_csv(train_meta) \
                .sample(frac=1, random_state=seed) \
                .reset_index(drop=True) # load and shuffle.
            #filter metadat to cell type of interest
            meta = meta[meta.eid == cell_i]
            # get genes
            genes = set(meta.gene_id.unique()).tolist()
            n_genes = len(genes)
            #data loaders ----
            all_dataset = Roadmap3D(cell_i, genes,w_prom=window_size, w_max=window_size,
                                     marks = [assay_i],train_dir=train_dir,train_meta=train_meta)
            all_loader = torch.utils.data.DataLoader(all_dataset, batch_size=1, num_workers=8, 
                                                     shuffle=False, drop_last=False)
            for item in all_loader:
                for k, v in item.items():
                    item[k] = v.cuda()
                #get cell
                cells_.append(cell_i)
                #get mark
                assays_.append(mark_i)
                #get expression
                print(item['log2RPKM'].float().unsqueeze(1))
                expression_.append(item['log2RPKM'].float().unsqueeze(1))
                #get hist mark activity
                print(item['x_p_pred_res'].numpy().mean())
                activity_.append(item['x_p_pred_res'].numpy().mean())
                
#all res kept in order so can just make df and save
res = pd.DataFrame({"cell":cells_,
                    "assay":assays_,
                    "log2RPKM":expression_,
                    "hist_mark_activity":activity_})
#save res
res.to_csv(str(PROJECT_PATH/'model_results/checkpoints/agg_hist_exp_chromo.csv'), sep='\t',index=False)