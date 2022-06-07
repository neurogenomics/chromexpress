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

#import constants
from epi_to_express.constants import (
    CHROM_LEN, 
    CHROMOSOMES, 
    SAMPLES,
    CHROMOSOME_DATA,
    SRC_PATH,
    ASSAYS,
    DATA_PATH)

#model imports
import tensorflow as tf
#data loading imports
from epi_to_express.utils import(
    train_valid_split,
    DataGenerator)


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
assert CELL in SAMPLES, f"{CELL} not valid. Must choose valid cell: {SAMPLES}"

MARK=args.MARK.lower()
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
cell = [CELL]
#resolution for training assay
pred_resolution = 25
# 1 Mb of the assay will be considered for the prediction of gene expression
window_size = 100_000

# Model specifics
batch_size = 128
n_epochs = 100
init_learning_rate = 0.001
lr_decay_factor = 0.2
lr_patience = 3
es_patience = 12

# Dataset parameters
valid_frac = 0.2
# Generic split func - Train test split over chromosomes
split = "CHROM"
# Exclude chromosomes, save for test set (when not predicting across cell types)
# hold out chromosomes 3,7,20
blind_test = [2,6,19]#positions not chroms
train_len = np.delete(CHROM_LEN, blind_test)
train_chrom = np.delete(CHROMOSOMES, blind_test)
test_len = CHROM_LEN[np.ix_(blind_test)]
test_chrom = CHROMOSOMES[np.ix_(blind_test)]
train_valid_samples = np.asarray(cell)

#Split the data into training and validation set - split by mix chrom and sample
#set seed so get the same split
(s_train_index, s_valid_index, c_train_index, c_valid_index, s_train_dist,
 s_valid_dist, c_train_dist, c_valid_dist) = train_valid_split(train_chrom,
                                                            train_len,
                                                            train_valid_samples,
                                                            valid_frac,
                                                            split)
# Training
train_cells = train_valid_samples[np.ix_(s_train_index)]
train_chromosomes = CHROMOSOMES[np.ix_(c_train_index)]
train_cell_probs = s_train_dist # equal probabilities
train_chromosome_probs = c_train_dist #weighted by chrom size
# Validation
#using wandb.config. converted their class (v 0.12.9, issue raised) so not storing for now
valid_cells = train_valid_samples[np.ix_(s_valid_index)]
valid_chromosomes = CHROMOSOMES[np.ix_(c_valid_index)]
valid_cell_probs = s_valid_dist
valid_chromosome_probs = c_valid_dist

# 2. --- Data loaders ---------------------------------------------------
training_generator = DataGenerator(cell=train_cells, 
                                   chromosomes=train_chromosomes,
                                   features=features, 
                                   window_size=window_size,
                                   pred_res=pred_resolution, 
                                   batch_size=batch_size, shuffle=True,
                                   test_subset=False)
validation_generator = DataGenerator(cell=valid_cells, 
                                   chromosomes=valid_chromosomes,
                                   features=features, 
                                   window_size=window_size,
                                   pred_res=pred_resolution,
                                   batch_size=batch_size, shuffle=True,
                                   test_subset=False)

# 3. --- Training ---------------------------------------------------
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

from epi_to_express.model import conv_profile_task_base

# import conv model
model = conv_profile_task_base(input_shape=[window_size//pred_resolution,len(features)],
                               output_shape=[1,1])

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
              metrics='mse')

# Train model on dataset
model.fit(training_generator,
          validation_data=validation_generator,
          epochs=n_epochs,
          verbose=2,
          use_multiprocessing=False,#started getting errors when set to True...
          callbacks=[es,lr_schedule]
         )

model.save(f"{MOD_SAVE_PATH}/mod_{'-'.join(cell)}_{'-'.join(features)}")

