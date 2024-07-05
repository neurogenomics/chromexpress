#generic imports
import os
import pathlib
from datetime import datetime
import time
import numpy as np
import pandas as pd
import math
import itertools

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

from chromexpress.utils import pearsonR
#model imports
import tensorflow as tf
#data loading imports
from chromexpress.utils import Roadmap3D_tf

# 1. --- SETUP PARAMETERS ------------------------------------------------

#what will be used to predict expression
features = ASSAYS

#resolution for training assay
pred_resolution = 100# choice of 100, 500, 2000
# 1 Mb of the assay will be considered for the prediction of gene expression
window_size = 40_000#6_000

readable_features = '-'.join(features)

y_type = 'both'
#number of k-fold cross validation
k_fold = 4
#seed
seed = 123

# 2. --- Dataset parameters -------------------------------
train_dir = PROJECT_PATH/'chromoformer'/'preprocessing'
train_meta = train_dir / 'train.csv'
meta = pd.read_csv(train_meta) \
    .reset_index(drop=True)


#two cell groups
rm_meta = pd.read_csv(str(PROJECT_PATH/'metadata/roadmap_metadata.csv'))
cell_grp = rm_meta[['Epigenome ID (EID)','ANATOMY']]
cell_grp['cell_anatomy_grp'] = np.where(cell_grp['ANATOMY'].isin(['ESC', 'ESC_DERIVED']),
                                        'ESC derived','Primary tissue')
#add cell anatomy type 2
cell_grp['cell_anatomy_grp2'] = np.where(cell_grp['ANATOMY'].isin(['ESC']),
                                         'ESC',np.where(cell_grp['ANATOMY'].isin(['ESC_DERIVED']),'ESC derived',
                                                        np.where(cell_grp['Epigenome ID (EID)'].isin(['E118','E114']),
                                                               'Cancer line','Primary tissue')))
cells = cell_grp['Epigenome ID (EID)']
# 3. --- load data with loader -------------------------------
cells_generators = dict()
genes = list(set(meta.gene_id.unique()))
n_genes = len(genes)

#loop through folds
cells_agg_hist_act = {i:[] for i in features}
for cell_i in cells:
    print("cell: ",cell_i)    
    cell_lst = []
    cell_gene = []
    cell_gene_lab = []
    cell_gene_id = []
    hists_act = {i:[] for i in features}
    #----
    #data loaders, one for each cell ----
    cells_generators[cell_i] = Roadmap3D_tf(cell_i, genes, batch_size=1,
                                            w_prom=window_size, w_max=window_size,
                                            marks = features,y_type=y_type,
                                            pred_res = pred_resolution,
                                            shuffle = False,#want order maintained to view all genes
                                            return_pcres=False,
                                            return_gene=True)
    for tss_i in range(len(cells_generators[cell_i])):
        hist_dat_i = cells_generators[cell_i].__getitem__(tss_i)
        #store gene activity - log2-transformed RPKM
        cell_gene.append(hist_dat_i[1]['log2RPKM'][0][0].numpy())
        cell_gene_lab.append(hist_dat_i[1]['label'][0][0].numpy())
        cell_gene_id.extend(hist_dat_i[0]['gene_ids'])
        cell_lst.append(cell_i)
        #store hist mark activity
        for hist_ind, hist_i in enumerate(features):
            hists_act[hist_i].append(hist_dat_i[0]['x_p_pred_res'][0,:,hist_ind].numpy())
    #split saving per hist mark
    for hist_i in features:
        #join into on dataframe
        hist_i_act = pd.DataFrame(hists_act[hist_i])
        hist_i_act['log2RPKM'] = cell_gene
        hist_i_act['label'] = cell_gene_lab
        hist_i_act['gene'] = cell_gene_id
        hist_i_act['cell'] = cell_lst
        #save foir each cell type
        cells_agg_hist_act[hist_i].append(hist_i_act)

for hist_i in features:            
    #join into one dataframe
    cells_agg_hist_act_i = pd.concat(cells_agg_hist_act[hist_i])
    #save
    cells_agg_hist_act_i.to_csv(str(PROJECT_PATH/f'model_results/checkpoints/{hist_i}_40k_hist_act.csv'), 
                                sep='\t',index=False)                                                                