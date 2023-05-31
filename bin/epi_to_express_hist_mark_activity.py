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

# 1. --- SETUP PARAMETERS ------------------------------------------------

#what will be used to predict expression
features = ASSAYS

#resolution for training assay
pred_resolution = 100# choice of 100, 500, 2000
# 1 Mb of the assay will be considered for the prediction of gene expression
window_size = 6_000

readable_features = '-'.join(features)

y_type = 'both'


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
esc_hists = {i:[] for i in features}
#subset 4 groups
sub_esc_gene = []
sub_esc_gene_lab = []
sub_esc_hists = {i:[] for i in features}
sub_esc_der_gene = []
sub_esc_der_gene_lab = []
sub_esc_der_hists = {i:[] for i in features}
for cell_i in esc_cells:
    print("esc cells:",cell_i)
    #data loaders, one for each cell ----
    cells_generators[cell_i] = Roadmap3D_tf(cell_i, genes, batch_size=1,
                                            w_prom=window_size, w_max=window_size,
                                            marks = features,y_type=y_type,
                                            pred_res = pred_resolution,
                                            shuffle = True,#want order maintained to view all genes
                                            return_pcres=False)
    for tss_i in range(len(cells_generators[cell_i])):
        hist_dat_i = cells_generators[cell_i].__getitem__(tss_i)
        #store gene activity - log2-transformed RPKM
        esc_gene.append(hist_dat_i[1]['log2RPKM'][0][0].numpy())
        esc_gene_lab.append(hist_dat_i[1]['label'][0][0].numpy())
        if(cell_grp[cell_grp['Epigenome ID (EID)']==cell_i]['cell_anatomy_grp2'].tolist()[0]=='ESC'):
            sub_esc_gene.append(hist_dat_i[1]['log2RPKM'][0][0].numpy())
            sub_esc_gene_lab.append(hist_dat_i[1]['label'][0][0].numpy())
        else: #esc deriv
            sub_esc_der_gene.append(hist_dat_i[1]['log2RPKM'][0][0].numpy())
            sub_esc_der_gene_lab.append(hist_dat_i[1]['label'][0][0].numpy())
        #store hist mark activity - mean -log 10 p-val for 6k bp around TSS
        for hist_ind, hist_i in enumerate(features):
            esc_hists[hist_i].append(hist_dat_i[0]['x_p_pred_res'][0,:,hist_ind].numpy().mean())
            if(cell_grp[cell_grp['Epigenome ID (EID)']==cell_i]['cell_anatomy_grp2'].tolist()[0]=='ESC'):
                sub_esc_hists[hist_i].append(hist_dat_i[0]['x_p_pred_res'][0,:,hist_ind].numpy().mean())
            else:
                #esc derv
                sub_esc_der_hists[hist_i].append(hist_dat_i[0]['x_p_pred_res'][0,:,hist_ind].numpy().mean())

prim_tiss_gene = []
prim_tiss_gene_lab = []
prim_tiss_hists = {i:[] for i in features}
#subset 4 groups
sub_prim_tiss_gene = []
sub_prim_tiss_gene_lab = []
sub_prim_tiss_hists = {i:[] for i in features}
sub_cncr_gene = []
sub_cncr_gene_lab = []
sub_cncr_hists = {i:[] for i in features}
for cell_i in prim_tiss_cells:
    print("prim tiss cells:",cell_i)
    #data loaders, one for each cell ----
    cells_generators[cell_i] = Roadmap3D_tf(cell_i, genes, batch_size=1,
                                            w_prom=window_size, w_max=window_size,
                                            marks = features,y_type=y_type,
                                            pred_res = pred_resolution,
                                            shuffle = True,#want order maintained to view all genes
                                            return_pcres=False)
    for tss_i in range(len(cells_generators[cell_i])):
        hist_dat_i = cells_generators[cell_i].__getitem__(tss_i)
        #store gene activity
        prim_tiss_gene.append(hist_dat_i[1]['log2RPKM'][0][0].numpy())
        prim_tiss_gene_lab.append(hist_dat_i[1]['label'][0][0].numpy())                                  
        if(cell_grp[cell_grp['Epigenome ID (EID)']==cell_i]['cell_anatomy_grp2'].tolist()[0]=='Cancer line'):
            sub_cncr_gene.append(hist_dat_i[1]['log2RPKM'][0][0].numpy())
            sub_cncr_gene_lab.append(hist_dat_i[1]['label'][0][0].numpy())
        else:#Primary tissue
            sub_prim_tiss_gene.append(hist_dat_i[1]['log2RPKM'][0][0].numpy())
            sub_prim_tiss_gene_lab.append(hist_dat_i[1]['label'][0][0].numpy())
                                  
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
agg_hist_exp_prim['cell_group'] = 'Primary tissue'
agg_hist_exp_esc = pd.DataFrame.from_dict(esc_hists)
agg_hist_exp_esc['log2RPKM'] = esc_gene
agg_hist_exp_esc['label'] = esc_gene_lab
agg_hist_exp_esc['cell_group'] = 'ESC derived'
agg_hist_exp = pd.concat([agg_hist_exp_prim, agg_hist_exp_esc])
#save as checkpoint
#PROJECT_PATH
agg_hist_exp.to_csv(str(PROJECT_PATH/'model_results/checkpoints/agg_hist_exp.csv'), 
                    sep='\t',index=False)
#join 4 grp into on dataframe
#prim & cncr
agg_hist_exp_prim = pd.DataFrame.from_dict(sub_prim_tiss_hists)
agg_hist_exp_prim['log2RPKM'] = sub_prim_tiss_gene
agg_hist_exp_prim['label'] = sub_prim_tiss_gene_lab
agg_hist_exp_prim['cell_group'] = 'Primary tissue'
agg_hist_exp_cncr = pd.DataFrame.from_dict(sub_cncr_hists)
agg_hist_exp_cncr['log2RPKM'] = sub_cncr_gene
agg_hist_exp_cncr['label'] = sub_cncr_gene_lab
agg_hist_exp_cncr['cell_group'] = 'Cancer line'
#esc & esc deriv
agg_hist_exp_esc = pd.DataFrame.from_dict(sub_esc_hists)
agg_hist_exp_esc['log2RPKM'] = sub_esc_gene
agg_hist_exp_esc['label'] = sub_esc_gene_lab
agg_hist_exp_esc['cell_group'] = 'ESC'
agg_hist_exp_esc_derv = pd.DataFrame.from_dict(sub_esc_der_hists)
agg_hist_exp_esc_derv['log2RPKM'] = sub_esc_der_gene
agg_hist_exp_esc_derv['label'] = sub_esc_der_gene_lab
agg_hist_exp_esc_derv['cell_group'] = 'ESC derived'
#join everything
agg_hist_exp_sub = pd.concat([agg_hist_exp_prim, agg_hist_exp_cncr, 
                              agg_hist_exp_esc, agg_hist_exp_esc_derv])
#save as checkpoint
#PROJECT_PATH
agg_hist_exp_sub.to_csv(str(PROJECT_PATH/'model_results/checkpoints/agg_hist_exp_4_grps.csv'), 
                    sep='\t',index=False)                                                                