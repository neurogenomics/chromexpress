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
from chromoformer.chromoformer.util import seed_everything

#params
pred_resolution = 100
#number of k-fold cross validation
k_fold = 4
#seed
seed = 123
#regression problem
y_type = 'log2RPKM'
#pred in batches
batch_size = 64
# window to be considered for the prediction of gene expression
window_size = 40_000

#model params
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

#paths
SAVE_PATH = pathlib.Path("./model_results")
MOD_SAVE_PATH = pathlib.Path("./model_results/models")
PRED_PATH = pathlib.Path("./model_results/predictions")
PRED_PATH.mkdir(parents=True, exist_ok=True)
train_dir = PROJECT_PATH/'chromoformer'/'preprocessing'
train_meta = train_dir / 'train.csv'

loss_fn = pearson_corrcoef
losses = []
indic = 0
with torch.no_grad():
    for assay_i in ASSAYS:
        print(assay_i)
        for cell_i in SAMPLE_IDS:
            print(cell_i)
            print("K fold:")
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
                #set seed
                seed_everything(101)
                
                print(fold)
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
                                         marks = assay_i,train_dir=train_dir,train_meta=train_meta,
                                         return_gene_ids = True)
                test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, 
                                                          num_workers=8, shuffle=False, drop_last=False)
                #get model
                mod_pth = f"{MOD_SAVE_PATH}/chromoformer_{cell_i}_{'-'.join(assay_i)}_kfold{fold}"
                model = Chromoformer(
                            n_feats=len(assay_i), embed_n_layers=embed_n_layers, #1 feature input
                            embed_n_heads = embed_n_heads, embed_d_model=embed_d_model, 
                            embed_d_ff=embed_d_ff, pw_int_n_layers=pw_int_n_layers, 
                            pw_int_n_heads=pw_int_n_heads, pw_int_d_model=pw_int_d_model, 
                            pw_int_d_ff=pw_int_d_ff,reg_n_layers=reg_n_layers, 
                            reg_n_heads=reg_n_heads, reg_d_model=reg_d_model, 
                            reg_d_ff=reg_d_ff, head_n_feats=head_n_feats
                        )
                model.cuda()
                model.load_state_dict(torch.load(mod_pth)['net'])
                model.eval()
                act_all = []
                pred_all = []
                genes = []
                for item in test_loader:
                    for k, v in item.items():
                        #note gene ids aren't tensors
                        if k=='gene':
                            item[k] = v
                        else:
                            item[k] = v.cuda()
                    #predict
                    out = model(item['x_p_2000'], item['pad_mask_p_2000'], item['x_pcre_2000'], 
                                item['pad_mask_pcre_2000'], item['interaction_mask_2000'],
                                item['x_p_500'], item['pad_mask_p_500'], item['x_pcre_500'], 
                                item['pad_mask_pcre_500'], item['interaction_mask_2000'],
                                item['x_p_100'], item['pad_mask_p_100'], item['x_pcre_100'], 
                                item['pad_mask_pcre_100'], item['interaction_mask_2000'],
                                item['interaction_freq'])
                    #eval
                    y = item['log2RPKM'].float().unsqueeze(1)
                    evalu = loss_fn(y[:,0], out[:,0])
                    #append
                    genes.extend(item['gene'])
                    act_all.extend(y[:,0].cpu().numpy().tolist())
                    pred_all.extend(out[:,0].cpu().numpy().tolist())
                losses.append(pd.DataFrame({"fold":[fold]*len(pred_all),
                                            "assay":['-'.join(assay_i)]*len(pred_all),
                                            "cell":[cell_i]*len(pred_all),
                                            "gene":genes,
                                            "pred":pred_all,
                                            "act":act_all}))
                
#concat to single dataframe
losses = pd.concat(losses)
#save res
losses.to_csv(f"{PRED_PATH}/chromoformer_gene_pred_act.csv", sep='\t',index=False)                
