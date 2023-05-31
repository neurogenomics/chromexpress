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
import tensorflow as tf
#data loading imports
from epi_to_express.utils import Roadmap3D_tf
from epi_to_express.utils import pearsonR

# 1. --- SETUP PARAMETERS ------------------------------------------------
#paths
SAVE_PATH = pathlib.Path("./model_results")
MOD_SAVE_PATH = pathlib.Path("./model_results/models")
PRED_PATH = pathlib.Path("./model_results/predictions")
PRED_PATH.mkdir(parents=True, exist_ok=True)

#params
pred_resolution = 100# choice of 100, 500, 2000
# 1 Mb of the assay will be considered for the prediction of gene expression
window_size = 6_000
#number of k-fold cross validation
k_fold = 4
#seed
seed = 123
#regression problem
y_type = 'log2RPKM'

# Model specifics - similar to https://www.nature.com/articles/s42256-022-00570-9
batch_size = 64
n_epochs = 100
init_learning_rate = 0.001
lr_decay_factor = 0.2
lr_patience = 3
es_patience = 12
pool_factor=4
kernel_size_factor=3

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
            test_generator = Roadmap3D_tf(cell_i, test_genes, batch_size=1,
                                          w_prom=window_size, w_max=window_size,
                                          marks = [assay_i],y_type=y_type,
                                          pred_res = pred_resolution,
                                          return_pcres=False,return_gene=True)
            
            #get model
            mod_pth = f"{MOD_SAVE_PATH}/covnet_{cell_i}_{'-'.join([assay_i])}_kfold{fold}"
            model = tf.keras.models.load_model(mod_pth, custom_objects={'pearsonR': pearsonR})
            #compile
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=init_learning_rate),
                          loss=tf.keras.losses.mean_squared_error,
                          metrics=['mse',pearsonR()])
            scores = []
            genes = []
            for ind in range(len(test_generator)):
                #get data
                X,y = test_generator[ind]
                genes.extend(X['gene_ids'])
                #predict
                output = model.predict(X['x_p_pred_res'])
                #eval
                evalu = loss_fn(y, output)
                scores.append(evalu)
            #keep all res in a list index is assay-cell
            scores = [score.numpy() for score in scores]
            losses.append(pd.DataFrame({"fold":[fold]*len(scores),
                                        "assay":[assay_i]*len(scores),
                                        "cell":[cell_i]*len(scores),
                                        "gene":genes,
                                        "Pearson_R":scores}))

#concat to single dataframe
losses = pd.concat(losses)
#save res
losses.to_csv(f"{PRED_PATH}/blind_test.csv", sep='\t',index=False)            
