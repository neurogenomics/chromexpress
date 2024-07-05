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
import tensorflow as tf
#data loading imports
from chromexpress.utils import Roadmap3D_tf
from chromexpress.utils import pearsonR

# 1. --- SETUP PARAMETERS ------------------------------------------------
#paths
SAVE_PATH = pathlib.Path("./model_results")
MOD_SAVE_PATH = pathlib.Path("./model_results/models")
PRED_PATH = pathlib.Path("./model_results/predictions")
PRED_PATH.mkdir(parents=True, exist_ok=True)

#params
pred_resolution = 100# choice of 100, 500, 2000
# 1 Mb of the assay will be considered for the prediction of gene expression
window_size_full = 6_000
window_size_half =3_000
window_size_quart = 1_500
res_full = window_size_full//100
res_half = window_size_half//100
buff_half = (res_full-res_half)//2
res_quart = window_size_quart//100
buff_quart = (res_full-res_quart)//2

batch_size = 256
#number of k-fold cross validation
k_fold = 4
#seed
seed = 123
#regression problem
y_type = 'log2RPKM'

loss_fn = pearsonR()#tf.keras.losses.mse
losses = []
indic = 0

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
            # Set random seeds.
            np.random.seed(101)
            tf.random.set_seed(101)
            random.seed(101)
            
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
            train_generator_full = Roadmap3D_tf(cell_i, train_genes, batch_size=batch_size,
                                               w_prom=window_size_full, w_max=window_size_full,
                                               marks = [assay_i],y_type=y_type,
                                               pred_res = pred_resolution,
                                               return_pcres=False,return_gene=True)
            
            all_hist_act = []
            genes = []
            all_y = []
            for ind in range(len(train_generator_full)):
                #get data
                X,y = train_generator_full[ind]
                #get hist mark activity
                #just store activity to compare across hist marks
                all_hist_act.append(X['x_p_pred_res'][:,:,0])
                genes.extend(X['gene_ids'])
                y = [item for sublist in y.numpy().tolist() for item in sublist]
                all_y.extend(y)
            dat_DF = pd.DataFrame(np.concatenate(all_hist_act))
            #rename cols to match pos
            dat_DF = dat_DF.set_axis(['pos_'+str(i) for i in range(0,dat_DF.shape[1])], axis=1)
            meta_dat = pd.DataFrame({"fold":[fold]*len(all_y),
                                     "assay":[assay_i]*len(all_y),
                                     "cell":[cell_i]*len(all_y),
                                     "gene":genes,
                                     "act":all_y})
            #col bind to meta data
            losses.append(pd.concat([meta_dat.reset_index(drop=True), dat_DF], axis=1))

#concat to single dataframe
losses = pd.concat(losses)
#save res
losses.to_csv(f"{PRED_PATH}/train_hist_activity.csv", sep='\t',index=False)
