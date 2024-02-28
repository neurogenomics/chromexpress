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
window_size = 6_000

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
esc_cells = cell_grp[cell_grp['cell_anatomy_grp']=='ESC derived']['Epigenome ID (EID)']
prim_tiss_cells = cell_grp[cell_grp['cell_anatomy_grp']=='Primary tissue']['Epigenome ID (EID)']

# 3. --- load data with loader -------------------------------
cells_generators = dict()
genes = list(set(meta.gene_id.unique()))
#go through each group separately
esc_gene = []
esc_gene_lab = []
esc_gene_id = []
esc_cell = []
esc_hists = {i:[] for i in features}
#subset 4 groups
sub_esc_gene = []
sub_esc_gene_lab = []
sub_esc_gene_id = []
sub_esc_cell = []
sub_esc_hists = {i:[] for i in features}
sub_esc_der_gene = []
sub_esc_der_gene_lab = []
sub_esc_der_gene_id = []
sub_esc_der_cell = []
sub_esc_der_hists = {i:[] for i in features}
#get activity by fold so it can be compared to performance
#k-fold cross-validation
train_dir = PROJECT_PATH/'chromoformer'/'preprocessing'
train_meta = train_dir / 'train.csv'

#loop through folds
folds_data_esc = []
folds_data_esc_sub = []
for ind,fold in enumerate([x+1 for x in range(k_fold)]):
    print(fold)
    for cell_i in esc_cells:
        print("esc cells: ",cell_i)
        #filter metadat to cell type of interest
        meta = pd.read_csv(train_meta).sample(frac=1, random_state=seed).reset_index(drop=True) # load and shuffle.
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
        #data loaders, one for each cell ----
        cells_generators[cell_i] = Roadmap3D_tf(cell_i, test_genes, batch_size=1,
                                                w_prom=window_size, w_max=window_size,
                                                marks = features,y_type=y_type,
                                                pred_res = pred_resolution,
                                                shuffle = False,#want order maintained to view all genes
                                                return_pcres=False,
                                                return_gene=True)
        for tss_i in range(len(cells_generators[cell_i])):
            hist_dat_i = cells_generators[cell_i].__getitem__(tss_i)
            #store gene activity - log2-transformed RPKM
            esc_gene.append(hist_dat_i[1]['log2RPKM'][0][0].numpy())
            esc_gene_lab.append(hist_dat_i[1]['label'][0][0].numpy())
            esc_gene_id.extend(hist_dat_i[0]['gene_ids'])
            esc_cell.append(cell_i)
            if(cell_grp[cell_grp['Epigenome ID (EID)']==cell_i]['cell_anatomy_grp2'].tolist()[0]=='ESC'):
                sub_esc_gene.append(hist_dat_i[1]['log2RPKM'][0][0].numpy())
                sub_esc_gene_lab.append(hist_dat_i[1]['label'][0][0].numpy())
                sub_esc_gene_id.extend(hist_dat_i[0]['gene_ids'])
                sub_esc_cell.append(cell_i)
            else: #esc deriv
                sub_esc_der_gene.append(hist_dat_i[1]['log2RPKM'][0][0].numpy())
                sub_esc_der_gene_lab.append(hist_dat_i[1]['label'][0][0].numpy())
                sub_esc_der_gene_id.extend(hist_dat_i[0]['gene_ids'])
                sub_esc_der_cell.append(cell_i)
            #store hist mark activity - mean -log 10 p-val for 6k bp around TSS
            for hist_ind, hist_i in enumerate(features):
                esc_hists[hist_i].append(hist_dat_i[0]['x_p_pred_res'][0,:,hist_ind].numpy().mean())
                if(cell_grp[cell_grp['Epigenome ID (EID)']==cell_i]['cell_anatomy_grp2'].tolist()[0]=='ESC'):
                    sub_esc_hists[hist_i].append(hist_dat_i[0]['x_p_pred_res'][0,:,hist_ind].numpy().mean())
                else:
                    #esc derv
                    sub_esc_der_hists[hist_i].append(hist_dat_i[0]['x_p_pred_res'][0,:,hist_ind].numpy().mean())
    #join into on dataframe
    agg_hist_exp_esc = pd.DataFrame.from_dict(esc_hists)
    agg_hist_exp_esc['log2RPKM'] = esc_gene
    agg_hist_exp_esc['label'] = esc_gene_lab
    agg_hist_exp_esc['gene'] = esc_gene_id
    agg_hist_exp_esc['cell'] = esc_cell
    agg_hist_exp_esc['cell_group'] = 'ESC derived'
    folds_data_esc.append(agg_hist_exp_esc)
    #join 4 grp into on dataframe
    #esc & esc deriv
    agg_hist_exp_esc = pd.DataFrame.from_dict(sub_esc_hists)
    agg_hist_exp_esc['log2RPKM'] = sub_esc_gene
    agg_hist_exp_esc['label'] = sub_esc_gene_lab
    agg_hist_exp_esc['gene'] = sub_esc_gene_id
    agg_hist_exp_esc['cell'] = sub_esc_cell
    agg_hist_exp_esc['cell_group'] = 'ESC'
    agg_hist_exp_esc['fold'] = fold
    agg_hist_exp_esc_derv = pd.DataFrame.from_dict(sub_esc_der_hists)
    agg_hist_exp_esc_derv['log2RPKM'] = sub_esc_der_gene
    agg_hist_exp_esc_derv['label'] = sub_esc_der_gene_lab
    agg_hist_exp_esc_derv['gene'] = sub_esc_der_gene_id
    agg_hist_exp_esc_derv['cell'] = sub_esc_der_cell
    agg_hist_exp_esc_derv['cell_group'] = 'ESC derived'
    agg_hist_exp_esc_derv['fold'] = fold
    #join everything
    agg_hist_exp_sub_i = pd.concat([agg_hist_exp_esc, agg_hist_exp_esc_derv])
    folds_data_esc_sub.append(agg_hist_exp_sub_i)
        
prim_tiss_gene = []
prim_tiss_gene_lab = []
prim_tiss_gene_id = []
prim_tiss_cell = []
prim_tiss_hists = {i:[] for i in features}
#subset 4 groups
sub_prim_tiss_gene = []
sub_prim_tiss_gene_lab = []
sub_prim_tiss_gene_id = []
sub_prim_tiss_cell = []
sub_prim_tiss_hists = {i:[] for i in features}
sub_cncr_gene = []
sub_cncr_gene_lab = []
sub_cncr_gene_id = []
sub_cncr_cell = []
sub_cncr_hists = {i:[] for i in features}

#loop through folds
folds_data_prim = []
folds_data_prim_sub = []
for ind,fold in enumerate([x+1 for x in range(k_fold)]):
    print(fold)
    for cell_i in prim_tiss_cells:
        print("prim tiss cells:",cell_i)
        #filter metadat to cell type of interest
        meta = pd.read_csv(train_meta).sample(frac=1, random_state=seed).reset_index(drop=True) # load and shuffle.
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
        #data loaders, one for each cell ----
        cells_generators[cell_i] = Roadmap3D_tf(cell_i, test_genes, batch_size=1,
                                                w_prom=window_size, w_max=window_size,
                                                marks = features,y_type=y_type,
                                                pred_res = pred_resolution,
                                                shuffle = False,#want order maintained to view all genes
                                                return_pcres=False,
                                                return_gene=True)
        for tss_i in range(len(cells_generators[cell_i])):
            hist_dat_i = cells_generators[cell_i].__getitem__(tss_i)
            #store gene activity
            prim_tiss_gene.append(hist_dat_i[1]['log2RPKM'][0][0].numpy())
            prim_tiss_gene_lab.append(hist_dat_i[1]['label'][0][0].numpy())
            prim_tiss_gene_id.extend(hist_dat_i[0]['gene_ids'])
            prim_tiss_cell.append(cell_i)
            if(cell_grp[cell_grp['Epigenome ID (EID)']==cell_i]['cell_anatomy_grp2'].tolist()[0]=='Cancer line'):
                sub_cncr_gene.append(hist_dat_i[1]['log2RPKM'][0][0].numpy())
                sub_cncr_gene_lab.append(hist_dat_i[1]['label'][0][0].numpy())
                sub_cncr_gene_id.extend(hist_dat_i[0]['gene_ids'])
                sub_cncr_cell.append(cell_i)
            else:#Primary tissue
                sub_prim_tiss_gene.append(hist_dat_i[1]['log2RPKM'][0][0].numpy())
                sub_prim_tiss_gene_lab.append(hist_dat_i[1]['label'][0][0].numpy())
                sub_prim_tiss_gene_id.extend(hist_dat_i[0]['gene_ids'])
                sub_prim_tiss_cell.append(cell_i)

            #store hist mark activity - mean -log 10 p-val for 6k bp around TSS
            for hist_ind, hist_i in enumerate(features):
                prim_tiss_hists[hist_i].append(hist_dat_i[0]['x_p_pred_res'][0,:,hist_ind].numpy().mean()) 
                if(cell_grp[cell_grp['Epigenome ID (EID)']==cell_i]['cell_anatomy_grp2'].tolist()[0]=='Cancer line'):
                    sub_cncr_hists[hist_i].append(hist_dat_i[0]['x_p_pred_res'][0,:,hist_ind].numpy().mean()) 
                else:#Primary tissue
                    sub_prim_tiss_hists[hist_i].append(hist_dat_i[0]['x_p_pred_res'][0,:,hist_ind].numpy().mean())    
    #join into on dataframe
    agg_hist_exp_prim = pd.DataFrame.from_dict(prim_tiss_hists)
    agg_hist_exp_prim['log2RPKM'] = prim_tiss_gene
    agg_hist_exp_prim['label'] = prim_tiss_gene_lab
    agg_hist_exp_prim['gene'] = prim_tiss_gene_id
    agg_hist_exp_prim['cell'] = prim_tiss_cell
    agg_hist_exp_prim['cell_group'] = 'Primary tissue'
    agg_hist_exp_prim['fold'] = fold
    folds_data_prim.append(agg_hist_exp_prim)
    #join 4 grp into on dataframe
    #prim & cncr
    agg_hist_exp_prim = pd.DataFrame.from_dict(sub_prim_tiss_hists)
    agg_hist_exp_prim['log2RPKM'] = sub_prim_tiss_gene
    agg_hist_exp_prim['label'] = sub_prim_tiss_gene_lab
    agg_hist_exp_prim['gene'] = sub_prim_tiss_gene_id
    agg_hist_exp_prim['cell'] = sub_prim_tiss_cell
    agg_hist_exp_prim['cell_group'] = 'Primary tissue'
    agg_hist_exp_prim['fold'] = fold
    agg_hist_exp_cncr = pd.DataFrame.from_dict(sub_cncr_hists)
    agg_hist_exp_cncr['log2RPKM'] = sub_cncr_gene
    agg_hist_exp_cncr['label'] = sub_cncr_gene_lab
    agg_hist_exp_cncr['gene'] = sub_cncr_gene_id
    agg_hist_exp_cncr['cell'] = sub_cncr_cell
    agg_hist_exp_cncr['cell_group'] = 'Cancer line'
    agg_hist_exp_cncr['fold'] = fold
    agg_hist_exp_sub_i = pd.concat([agg_hist_exp_prim, agg_hist_exp_cncr])
    folds_data_prim_sub.append(agg_hist_exp_sub_i)
        
#join into one dataframe
folds_data_esc = pd.concat(folds_data_esc)
folds_data_prim = pd.concat(folds_data_prim)
agg_hist_exp = pd.concat([folds_data_prim, folds_data_esc])
#save as checkpoint
#PROJECT_PATH
agg_hist_exp.to_csv(str(PROJECT_PATH/'model_results/checkpoints/agg_hist_exp.csv'), 
                    sep='\t',index=False)

#join 4 grp into on dataframe
folds_data_esc_sub = pd.concat(folds_data_esc_sub)
folds_data_prim_sub = pd.concat(folds_data_prim_sub)
agg_hist_exp_sub = pd.concat([folds_data_prim_sub,folds_data_esc_sub])
#save as checkpoint
#PROJECT_PATH
agg_hist_exp_sub.to_csv(str(PROJECT_PATH/'model_results/checkpoints/agg_hist_exp_4_grps.csv'), 
                    sep='\t',index=False)                                                                