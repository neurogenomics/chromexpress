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

#track models
import wandb
#from omegaconf import OmegaConf
from wandb.keras import WandbCallback


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
from epi_to_express.utils import pearsonR
#model imports
import tensorflow as tf
#data loading imports
from epi_to_express.utils import Roadmap3D_tf
from epi_to_express.model import conv_profile_task_base
from epi_to_express.model import covnet

CELL='E003'#args.CELL
#leading and trailing whitespace
CELL=CELL.strip()
#assert it's a valid choice
assert CELL in SAMPLE_IDS, f"{CELL} not valid. Must choose valid cell: {SAMPLE_IDS}"

MARK='h3k4me1'#args.MARK.lower()
MARK=MARK.strip()
#assert it's a valid choice
assert MARK in ASSAYS, f"{MARK} not valid. Must choose valid assay: {ASSAYS}"

print("---------------------------------")
print(CELL)
print(MARK)
print("---------------------------------")

# Set random seeds.
np.random.seed(101)
tf.random.set_seed(101)
random.seed(101)
    
SAVE_PATH = pathlib.Path("./model_results")
SAVE_PATH.mkdir(parents=True, exist_ok=True)

MOD_SAVE_PATH = pathlib.Path("./model_results/models")
MOD_SAVE_PATH.mkdir(parents=True, exist_ok=True)

# 1. --- SETUP PARAMETERS ------------------------------------------------

#what will be used to predict expression
features = [MARK]
#what cell will we predict in
cell = CELL
#resolution for training assay
pred_resolution = 100# choice of 100, 500, 2000
# 1 Mb of the assay will be considered for the prediction of gene expression
window_size = 6_000
#number of k-fold cross validation
k_fold = 4
#which fold to run 
fold = 0
readable_features = '-'.join(features)
wandb.init(
    name=f'covnet_6k_{cell}_{readable_features}_{fold}',
    entity="al-murphy",
    project="Epi_to_Express",
)

#seed
seed = 123
#regression problem
y_type = 'log2RPKM'

# Model specifics
batch_size = 64
n_epochs = 100
init_learning_rate = 0.001
lr_decay_factor = 0.2
lr_patience = 3
es_patience = 12
pool_factor=4
kernel_size_factor=3

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
print(f"K-fold Cross-Validation - blind test: {fold}")
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
training_generator = Roadmap3D_tf(cell, train_genes, batch_size=batch_size,
                                  w_prom=window_size, w_max=window_size,
                                  marks = features,y_type=y_type,
                                  pred_res = pred_resolution,
                                  return_pcres=False)
validation_generator = Roadmap3D_tf(cell, val_genes, batch_size=batch_size,
                                    w_prom=window_size, w_max=window_size,
                                    marks = features,y_type=y_type,
                                    pred_res = pred_resolution,
                                    return_pcres=False)
#----
#train ----
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# import conv model
#model = conv_profile_task_base(output_shape=[1,1],window_size=window_size,
#                               pred_res=pred_resolution,pool_factor=pool_factor,
#                               kernel_size_factor=kernel_size_factor)
model = covnet(window_size=window_size,pred_res=pred_resolution)

#learning rate schedule
lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", 
                                                 factor=lr_decay_factor, 
                                                 patience=lr_patience)
#early stopping
es = tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=es_patience,
                                      #save best weights
                                      restore_best_weights=True)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=init_learning_rate),
              loss=tf.keras.losses.mean_squared_error,
              metrics=['mse',pearsonR()])

# Train model on dataset
model.fit(training_generator,
          validation_data=validation_generator,
          epochs=n_epochs,
          verbose=2,
          use_multiprocessing=False,#started getting errors when set to True...
          callbacks=[es,lr_schedule,WandbCallback(save_model=False)]
         )

model.save(f"{MOD_SAVE_PATH}/covnet_{cell}_{'-'.join(features)}_kfold{ind}")
