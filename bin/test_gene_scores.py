#implement archR Gene Scores - https://doi.org/10.1038/s41588-021-00790-6
#Follows same approach apart from excluding regions of other genes
#Used here for bulk histone mark signal rather than single-cell ATAC
#

#uses an asymmetric aggregation function that decays with distance upstream of the TSS, 
#and downstream of the end of the gene body
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



#Function to calc gene scores
def calc_gene_score(hist_act, gene_ends, gene_lens, bin_size = 500, 
                    res=100, upsteam_tss=5_000, gene_size_weight=5):
    #reaverage based on bin size
    hist_act = hist_act.T.reshape(-1, bin_size//res, hist_act.T.shape[1]).mean(axis = 1).T
    #loop through each gene
    gs_weight_all = []
    for gene_pos in range(hist_act.shape[0]):
        hist_act_gi = hist_act[gene_pos,:]
        gene_len_i = gene_lens[gene_pos]
        gene_end_i = gene_ends[gene_pos] #end is bp relative to TSS
        #loop through each pos
        gs_weight_gene_i = []
        for pos_i in range(hist_act_gi.shape[0]):
            #centred on TSS so TSS = 100_000
            tss_pos = 100_000//bin_size
            #get gene body end relative to the TSS based on bin size
            gene_end_bin = int(np.ceil(gene_end_i//bin_size))
            if (pos_i<(tss_pos-(upsteam_tss//bin_size))): 
                #upstream TSS
                #weight at each distance amount
                #e^(-abs(distance)/5000) + e^-1
                tss_pos_bin = (200_000//bin_size)//2 #tss is centre of window, based on bin size
                dist_i = tss_pos_bin-pos_i  #distance to gene body but upstream so TSS
                weight = np.exp(-1*(np.abs(dist_i)/5000))+np.exp(-1)
            elif(pos_i>tss_pos+gene_end_bin):
                #downstream gene body so use distance calc
                #weight at each distance amount
                #e^(-abs(distance)/5000) + e^-1
                tss_pos_bin = (200_000//bin_size)//2 #tss is centre of window, based on bin size
                gb_pos_bin = tss_pos_bin+gene_end_bin
                dist_i = pos_i - gb_pos_bin  #distance to gene body but downstream so gene end
                weight = np.exp(-1*(np.abs(dist_i)/5000))+np.exp(-1)
            else:
                #in TSS/gene body, use TSS/gene body calc
                #weight for upstr TSS, TSS and gene body
                #1+e^-1
                weight = 1+np.exp(-1)
            #add weight to be mutlipled later
            gs_weight_gene_i.append(weight)     
        #gene_size_weight
        gene_scale_factor = gene_size_weight/gene_len_i
        scaled_gs_weight_gene_i = np.array(gs_weight_gene_i)*gene_scale_factor
        gs_weight_all.append(scaled_gs_weight_gene_i)
    #now multiply by all weights
    return(np.sum(hist_act*np.array(gs_weight_all),axis=1))

# 1. --- SETUP PARAMETERS ------------------------------------------------
#paths
SAVE_PATH = pathlib.Path("./model_results")
MOD_SAVE_PATH = pathlib.Path("./model_results/models")
PRED_PATH = pathlib.Path("./model_results/predictions")
PRED_PATH.mkdir(parents=True, exist_ok=True)

#params
pred_resolution = 100# choice of 100, 500, 2000
# 1 Mb of the assay will be considered for the prediction of gene expression
window_size = 200_000 #100k up and down
#number of k-fold cross validation
k_fold = 4
#seed
seed = 123
#regression problem
y_type = 'log2RPKM'

# Model specifics - similar to https://www.nature.com/articles/s42256-022-00570-9
batch_size = 256
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

#for gene scores
bin_size = 500

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
        
        #load gene length info
        gene_strt_end = pd.read_csv("./metadata/mart_export_gene_lens.txt.gz", sep="\t")
        gene_strt_end = gene_strt_end[["Gene stable ID","Gene start (bp)","Gene end (bp)"]].drop_duplicates()
        #restrict to genes of interest
        gene_strt_end = gene_strt_end[gene_strt_end['Gene stable ID'].isin(meta['gene_id'].tolist())]
        #get avg gene length if any fail
        gene_strt_end['gene_len'] = gene_strt_end['Gene end (bp)']-gene_strt_end['Gene start (bp)']
        avg_gene_len = int(np.average(gene_strt_end['gene_len']))
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
            test_generator = Roadmap3D_tf(cell_i, test_genes, batch_size=batch_size,
                                          w_prom=window_size, w_max=window_size,
                                          marks = [assay_i],y_type=y_type,
                                          pred_res = pred_resolution,
                                          return_pcres=False,return_gene=True)
            
            
            all_y = []
            all_output = []
            genes = []
            for ind in range(len(test_generator)):
                #get data
                X,y = test_generator[ind]
                genes.extend(X['gene_ids'])
                #get gene info - some genes won't be found need to deal with these (right join)
                mtch_gene_dat = pd.DataFrame({'Gene stable ID':X['gene_ids']}).merge(gene_strt_end,how='left')
                #need to add on TSS
                mtch_gene_dat = mtch_gene_dat.merge(meta[["gene_id","start"]],
                                                    left_on="Gene stable ID",right_on="gene_id")
                #substitue averages for where genes aren't found
                mtch_gene_dat.loc[mtch_gene_dat['gene_len'].isna(),'gene_len'] = avg_gene_len
                mtch_gene_dat.loc[mtch_gene_dat['Gene end (bp)'].isna(),'Gene end (bp)'] = 0
                #get end relative to TSS
                mtch_gene_dat['end_rel'] = mtch_gene_dat['Gene end (bp)']-mtch_gene_dat['start']#TSS
                #if error where end is < TSS, just add on the average gene length as end
                mtch_gene_dat.loc[mtch_gene_dat['end_rel']<0,'end_rel']=(avg_gene_len//bin_size)
                gene_ends = mtch_gene_dat['end_rel'].to_list()
                gene_lens = mtch_gene_dat['gene_len'].to_list()
                hist_act = np.array(X['x_p_pred_res'])[:,:,0]
                #get gene score
                output = calc_gene_score(hist_act=hist_act,gene_lens=gene_lens,
                                         gene_ends=gene_ends,bin_size=bin_size)
                #don't eval - return true and pred instead so can agg later with pearson R
                output = output.tolist()
                y = [item for sublist in y.numpy().tolist() for item in sublist]
                all_output.extend(output)
                all_y.extend(y)
            
            losses.append(pd.DataFrame({"fold":[fold]*len(all_y),
                                        "assay":[assay_i]*len(all_y),
                                        "cell":[cell_i]*len(all_y),
                                        "gene":genes,
                                        "act":all_y,
                                        "pred":all_output}))

#concat to single dataframe
losses = pd.concat(losses)
#save res
losses.to_csv(f"{PRED_PATH}/gene_score_perf.csv", sep='\t',index=False)            ##blind_test