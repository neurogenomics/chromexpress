## In silico perturbation of histone mark levels and their effect on expression

#* I think this will need to be done on the single mark chromoformer models since we can't account for the effect of a histone mark change on another histone mark.
#* promoter marks (only alter promoter signals) - H3K27ac (active), H3K27me3 (repressive).
#* distal marks (only alter distal signals) - H3K27ac (active), H3K27me3 (repressive).
#* Use same mark and alter at diff locations for distal and promoter. More consistent.
#* Average the predictions where multiple, k fold models makes for more consistent preds (seen in lit for SNP eff preds)
#* Want a plot of proportion decrease in histone mark activity effect on proportion of expression
#* Question is it a linear relationship that's predicted? Does this change for promoter vs distal marks? Does this change for distance from TSS?
#* Question is there any distal histone mark signal changes causing a drastic change in expression? Could rank and then check for enrichment for top 10% in SNPs for a disease relating to the cell type.

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
#data loading imports
from chromoformer.chromoformer.data import Roadmap3D
#model arch
from chromoformer.chromoformer.net import Chromoformer
from chromoformer.chromoformer.util import seed_everything
import argparse
#pass inputs
# argv
def get_args():
    parser = argparse.ArgumentParser(description="train")
    parser.add_argument('-c', '--CELL', default='', type=str, help='Cell to train in')
    args = parser.parse_args()
    return args

args=get_args()

CELL=args.CELL
#leading and trailing whitespace
CELL=CELL.strip()
#assert it's a valid choice
assert CELL in SAMPLE_IDS, f"{CELL} not valid. Must choose valid cell: {SAMPLE_IDS}"
cell_i = CELL

#params
pred_resolution = 100
#number of k-fold cross validation
k_fold = 4
#seed
seed = 123
#regression problem
y_type = 'log2RPKM'
#pred in batches
batch_size = 1 #one at a time to store individual values
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

#we want to run h3k27ac at prom and distal for expressed genes
#and run h3k27me3 at prom and distal for non-expressed genes
runs = [['h3k27me3','non-expressed','promoter'],['h3k27me3','non-expressed','distal'], 
        ['h3k27ac','expressed','promoter'],['h3k27ac','expressed','distal']]

#We want to decrease histone mark activity across different proportions
props = np.round(np.arange(0, 1.1, .1),2)[::-1] 

losses = []
ind_test=0
with torch.no_grad():
    for assay_i, gene_type_i,location_i in runs:
        print(assay_i)
        print(gene_type_i)
        print(location_i)
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

            #interested in all genes not just test/valid - average the predictions where multiple, k fold models
            all_genes = qs[(fold + 0) % 4] + qs[(fold + 1) % 4] + qs[(fold + 2) % 4] + qs[(fold + 3) % 4]
            all_chrom = list(set(meta[meta.gene_id.isin(all_genes)]['chrom']))
            all_genes = meta[meta.gene_id.isin(all_genes) & meta.chrom.isin(all_chrom)]['gene_id'].tolist()
            #----
            #data loaders ----
            all_dataset = Roadmap3D(cell_i, all_genes,w_prom=window_size, w_max=window_size,
                                    marks = [assay_i],train_dir=train_dir,train_meta=train_meta,
                                    return_gene_ids = True)
            all_loader = torch.utils.data.DataLoader(all_dataset, batch_size=batch_size, 
                                                     num_workers=8, shuffle=False, drop_last=False)
            #get model
            mod_pth = f"{MOD_SAVE_PATH}/chromoformer_{cell_i}_{'-'.join([assay_i])}_kfold{fold}"
            model = Chromoformer(
                n_feats=len([assay_i]), embed_n_layers=embed_n_layers, #1 feature input
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
            #loop for different proportions of reducing histone mark levels
            print("Proportion: ")
            for prop_i in props:
                print(prop_i)
                pred_scores = []
                pred_genes = []
                labels = []
                mut_pos_all = []
                for item in all_loader:
                    for k, v in item.items():
                        #note gene ids aren't tensors
                        if k=='gene':
                            item[k] = v
                        else:
                            item[k] = v.cuda()

                    #only expressed/non-expressed genes returned
                    if (item['label'].cpu().numpy()[0] == 1 and gene_type_i=='expressed') or (
                        item['label'].cpu().numpy()[0] == 0 and gene_type_i=='non-expressed'): 
                        #Now modify histone mark activity
                        if location_i =='promoter': 
                            #x_p_{bin_size} is the promoter regions, need to mutate all resolutions
                            #these contain 40k bp so only mutate centre 6k (same receptive field as prom model)
                            #start of gene is centred: start_p, end_p = start_p - 20000, start_p + 20000
                            recep_field = 6_000
                            for res_i in [100,500,2_000]:
                                num_pos = recep_field//res_i
                                all_pos_num = item[f'x_p_{res_i}'].shape[2]
                                dist_num = (all_pos_num-num_pos)//2
                                #get modulus to adjust - necessary for 2k res
                                mod = (all_pos_num-num_pos)%2
                                item[f'x_p_{res_i}'][0:1,0:1,
                                                     dist_num:(all_pos_num-dist_num-mod),
                                                     0:1] = item[f'x_p_{res_i}'][0:1,0:1,
                                                                                 dist_num:(all_pos_num-dist_num-mod),
                                                                                 0:1]*prop_i
                            #position important for getting mutation location from gene with distal
                            mut_pos = [0]
                        else: #'distal'
                            #x_p_{bin_size} also contains the distal regions, need to mutate all resolutions
                            #these contain 40k bp so mutate all but centre 6k (same receptive field as prom model)
                            #start of gene is centred: start_p, end_p = start_p - 20000, start_p + 20000
                            recep_field = 6_000
                            #Mutate chunks of distal histone mark activity in highest res (2k) bins
                            #note this will increase batch size
                            #position important for getting mutation location from gene with distal
                            mut_pos = []
                            item_org = item.copy()
                            for pos_i in range(0,40_000//2_000):
                                item_pos_i = item_org.copy()
                                #if centre 6k, skip
                                if pos_i not in [8,9,10]:
                                    mut_pos.append(pos_i)
                                    for res_i in [100,500,2_000]:
                                        #update positions to be mutated by resolution
                                        #pos_i at 2k res
                                        pos_i_res = 2_000
                                        pos_i_res_i_strt = pos_i*(pos_i_res//res_i)
                                        pos_i_res_i_end = (pos_i+1)*(pos_i_res//res_i)
                                        num_pos = recep_field//res_i
                                        all_pos_num = item_pos_i[f'x_p_{res_i}'].shape[2]
                                        dist_num = (all_pos_num-num_pos)//2
                                        #get modulus to adjust - necessary for 2k res
                                        mod = (all_pos_num-num_pos)%2
                                        item_pos_i[f'x_p_{res_i}'][0:1,0:1,
                                                                   pos_i_res_i_strt:pos_i_res_i_end,
                                                                   0:1] = item_pos_i[f'x_p_{res_i}'][0:1,0:1,
                                                                                                     pos_i_res_i_strt:pos_i_res_i_end,
                                                                                                     0:1]*prop_i
                                        if pos_i >0:
                                            item[f'x_p_{res_i}'] = torch.cat((item[f'x_p_{res_i}'],item_pos_i[f'x_p_{res_i}']),0)
                                            if pos_i ==0:
                                                item = item_pos_i
                            #multiple by number of separate mutations
                            item['gene'] = item['gene']*((40_000//2_000)-(recep_field//2_000))
                            item['label'] = item['label'].repeat(((40_000//2_000)-(recep_field//2_000)),1)
                            #do the same for the model inputs
                            for item_i in ['pad_mask_p_2000','x_pcre_2000','pad_mask_pcre_2000','interaction_mask_2000',
                                           'pad_mask_p_500','x_pcre_500','pad_mask_pcre_500',
                                           'pad_mask_p_100','x_pcre_100','pad_mask_pcre_100',
                                           'interaction_freq']:
                                #diff ones have diff dims
                                if ('x_pcre' in item_i) or ('interaction_mask' in item_i):
                                    item[item_i] = item[item_i].repeat(((40_000//2_000)-(recep_field//2_000)),1,1,1)
                                elif item_i =='interaction_freq':
                                    item[item_i] = item[item_i].repeat(((40_000//2_000)-(recep_field//2_000)),1,1)
                                else:
                                    item[item_i] = item[item_i].repeat(((40_000//2_000)-(recep_field//2_000)),1,1,1,1)  

                        #predict
                        out = model(item['x_p_2000'], item['pad_mask_p_2000'], item['x_pcre_2000'], 
                                    item['pad_mask_pcre_2000'], item['interaction_mask_2000'],
                                    item['x_p_500'], item['pad_mask_p_500'], item['x_pcre_500'],
                                    item['pad_mask_pcre_500'], item['interaction_mask_2000'],
                                    item['x_p_100'], item['pad_mask_p_100'], item['x_pcre_100'], 
                                    item['pad_mask_pcre_100'], item['interaction_mask_2000'],
                                    item['interaction_freq'])

                        pred_scores.extend(out[:,0].cpu().numpy().tolist())
                        pred_genes.extend(item['gene'])
                        labels.extend(item['label'].cpu().numpy().tolist())
                        mut_pos_all.extend(mut_pos)

                #keep all res in a list index is assay-cell
                losses.append(pd.DataFrame({"fold":[fold]*len(pred_scores),
                                            "assay":[assay_i]*len(pred_scores),
                                            "cell":[cell_i]*len(pred_scores),
                                            "gene_type":[gene_type_i]*len(pred_scores),
                                            "mut":[location_i]*len(pred_scores),
                                            "prop":[prop_i]*len(pred_scores),
                                            "true_gene_exp_label":labels,
                                            "dist_pos_tss":mut_pos_all,
                                            "gene":pred_genes,
                                            "pred_expression":pred_scores}))
            #save losses for assay gene type location cell combo
            #concat to single dataframe
            losses = pd.concat(losses)
            #save res
            losses.to_csv(f"{PRED_PATH}/{cell_i}_{str(ind_test)}_fold{str(ind)}_chromoformer_in_silico_mut.csv",
                          sep='\t',index=False)
            print(f"Saved {cell_i}_fold{str(ind)}_{str(ind_test)}")
            #update - not just doing it at the end of the runs because of the big mem usage of storing all
            losses = []
        ind_test+=1