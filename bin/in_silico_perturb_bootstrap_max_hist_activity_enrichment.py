#Check for overlap with fine-mapped QTL data for Hi-C
#To look at if the model predicts regions with known casual genetic variants for phenotypes, 
#check for overlap with fine-mapped QTLs. Got UK Biobank fine-mapped QTLs from:
#
#Wang, Q. S. et al. Leveraging supervised learning for functionally informed fine-mapping of 
#cis-eQTLs identifies an additional 20,913 putative causal eQTLs. Nat. Commun. 12, 3394 (2021).
#
#Dataset is supp Supplementary Data 7 'List of tissue-specific putative causal eQTLs' from this paper
#
#Used FINEMAP v1.3.1 (Benner et al. 2016 Bioinformatics, 2018 bioRxiv) and SuSiE v0.8.1.0521 (Wang et al. 2018 bioRxiv).
#
#These fine-mapped snps to where SuSiE causal probability > 0.9 in the tissue (posterior inclusion probability (PIP) in
#a credible causal set >0.9) and is >.1 in other tissues to get just high confidence, tissue-specific fine-mapped SNPs

#Test is conducted by matching gtex tissue to cells tested and for each cell, sorting our in silico perturbation 
#predictions in deciles for each cell based on the the models expected change in expression
#Then check the overlap with the fine-mapped snps for the regions for each decile. To get a sense of how likely
#this is to happen by chance, bootstrap sample 10k times the same size number of regions from all possible regions
#and get a p-value by checking how many of the bootstrap lists find more overlaps with the fine-mapped SNPs than
#the decile (same approach as [EWCE](https://www.frontiersin.org/articles/10.3389/fnins.2016.00016/full) 
#has for genes).

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
import pandas
import random

#import constants
from chromexpress.constants import (
    CHROM_LEN, 
    CHROMOSOMES, 
    SAMPLES,
    SAMPLE_IDS,
    CHROMOSOME_DATA,
    SRC_PATH,
    ASSAYS,
    CELLS_NAMES,
    SAMPLE_NAMES,
    PROJECT_PATH)

# ---- functions ----
def bootstrap_neg(df_test,df_eqtl,num_regs,num_poss_regs,num_tests=10_000):
    """
    Bootstrap sample a background set to compare number of regions
    with fine-mapped snps in them
    
    Arguments:
        df_test - dataframe with predicted effect regions to be tested 
        df_eqtl - dataframe with eqtl fine-mapped SNPs
        num_regs - total number of regions to sample
        num_poss_regs - total possible number of regions
        num_tests - number of bootstrap tests to do, default 1000.
        
    """
    #store proportional results
    res = []
    #get num_tests random state numbers
    rand_nums = random.sample(range(1, num_tests*1_000), num_tests)
    for rand_i in rand_nums:
        #randomly sample all data
        df_test_i = df_test.sample(n=num_regs, random_state=rand_i)
        #join on gene, then filter to where region matches
        df_test_i = pd.merge(df_test_i,df_eqtl,
                             left_on='gene',right_on='gene_id')
        #filt reg match
        num_matches = df_test_i[(df_test_i['variant_pos'] >= df_test_i['mut_reg_start'])&(
            df_test_i['variant_pos'] <=df_test_i['mut_reg_end'] )].shape[0]
        #save res
        res.append(num_matches/num_poss_regs)
    
    return(res)
# --------

#need to also get hist activity and check if max activity just captures the enrichment..
hist_act = pd.read_csv('./model_results/checkpoints/h3k27ac_40k_hist_act.csv', sep='\t')
#we want just 20k upstream - 100 bp averaging
#forward and reverse strand already dealt with in data loader
meta_hist_act_dat = hist_act.iloc[:,400:]
hist_act_upstr = pd.concat([hist_act.iloc[:,:(20_000//100)],meta_hist_act_dat], axis=1)
#also average activity to 2k to match hi-C and in silico mut
arr = np.array(hist_act_upstr.iloc[:,:200])
tmp = []
for x in range(0, arr.shape[1], 20):
    averaged = arr[:, x:x+20].mean(axis=1)
    tmp.append(averaged) 
arr_avg = np.column_stack(tmp)
hist_act_upstr = pd.concat([pd.DataFrame(arr_avg),meta_hist_act_dat], axis=1)
#go long instead of wide to get a row per pos
hist_act_upstr = pd.melt(hist_act_upstr, id_vars=['log2RPKM','label','gene','cell'], 
                         value_vars=[i for i in range(0,10)])
#get top 2k bin of activity per gene/cell
hist_act_upstr = hist_act_upstr.sort_values(['cell','gene','value'],ascending=False)
hist_act_upstr_top = hist_act_upstr.groupby(['cell','gene']).head(1).reset_index(drop=True)

#we also want top downstream - 100 bp averaging
#forward and reverse strand already dealt with in data loader
meta_hist_act_dat = hist_act.iloc[:,400:]
hist_act_dwnstr = pd.concat([hist_act.iloc[:,(20_000//100):],meta_hist_act_dat], axis=1)
#also average activity to 2k to match hi-C and in silico mut
arr = np.array(hist_act_dwnstr.iloc[:,:200])
tmp = []
for x in range(0, arr.shape[1], 20):
    averaged = arr[:, x:x+20].mean(axis=1)
    tmp.append(averaged) 
arr_avg = np.column_stack(tmp)
hist_act_dwnstr = pd.concat([pd.DataFrame(arr_avg),meta_hist_act_dat], axis=1)
#go long instead of wide to get a row per pos
hist_act_dwnstr = pd.melt(hist_act_dwnstr, id_vars=['log2RPKM','label','gene','cell'], 
                         value_vars=[i for i in range(0,10)])
#get top 2k bin of activity per gene/cell
hist_act_dwnstr = hist_act_dwnstr.sort_values(['cell','gene','value'],ascending=False)
hist_act_dwnstr_top = hist_act_dwnstr.groupby(['cell','gene']).head(1).reset_index(drop=True)

#join up and downstream
hist_act_upstr_top = pd.concat([hist_act_upstr_top,hist_act_dwnstr_top])

#get fine-mapped eQTLs
PRED_PATH = pathlib.Path("./model_results/predictions")
xl_file = pd.ExcelFile("./qtl_ovrlp/41467_2021_23134_MOESM10_ESM.xlsx")
dfs = {sheet_name: xl_file.parse(sheet_name) 
          for sheet_name in xl_file.sheet_names}
fm_eqtl_tiss = dfs['put_causal_eqtls_tissue_unique']
#match to cells
gtex_map = pd.read_csv("./metadata/gtex_roadmap_mapping.csv")
gtex_map = gtex_map[-pd.isnull(gtex_map['cell_id'])]
fm_eqtl_tiss_filt = pd.merge(fm_eqtl_tiss,gtex_map,left_on='tissue',right_on='gtex_tissue')
fm_eqtl_tiss_filt = fm_eqtl_tiss_filt[-pd.isnull(fm_eqtl_tiss_filt['cell_id'])]
#get gene ids without number extension
fm_eqtl_tiss_filt['gene_id'] = fm_eqtl_tiss_filt.gene.str.split('.',expand=True)[0]
#get variant info
fm_eqtl_tiss_filt[['variant_chr','variant_pos',
                   'variant_a1','variant_a2']]=fm_eqtl_tiss_filt.variant.str.split(":",expand=True)
fm_eqtl_tiss_filt["variant_pos"] = pd.to_numeric(fm_eqtl_tiss_filt["variant_pos"])

#need to match to the same genes and cell types as used in pred bootstrap tests
#----------------------------------------
mut_dat_file = f"{PRED_PATH}/chromoformer_in_silico_mut.csv"
mut_dat = pd.read_csv(mut_dat_file)
#average the effects from multiple models
mut_dat_avg = mut_dat.groupby(['assay','cell','gene_type','mut','prop','true_gene_exp_label',
                               'dist_pos_tss','gene','cell_name','dist_tss','Epigenome ID (EID)',
                               'ANATOMY','cell_anatomy_grp'])['pred_expression'].mean().reset_index(name='pred_expression')
#interested in just where hsit mark was completely removed
mut_dist_dat = mut_dat_avg[mut_dat_avg['prop']==0]
#join on dat with no mutation to get orig exp prediction
no_mut_dist_dat = mut_dat_avg[mut_dat_avg['prop']==1.0]
#join
no_mut_dist_dat = no_mut_dist_dat[['assay','cell','gene_type','mut','true_gene_exp_label',
                                   'gene','pred_expression']]
mut_dist_dat = pd.merge(mut_dist_dat,no_mut_dist_dat,on=['assay','cell','gene_type','mut','true_gene_exp_label','gene'],
                  suffixes=('','_orig'),how='left')
# make sure col order is consistent
mut_dist_dat['cell_anatomy_grp'] = pd.Categorical(mut_dist_dat['cell_anatomy_grp'], 
                                                  ["Cancer line","ESC derived", "ESC", "Primary tissue"])

#delete unneeded
del mut_dat,mut_dat_avg,no_mut_dist_dat

#separate out unexp and exp models - going with just exp - active model (h3k27ac) as in silico perturb seems more sensible
mut_dist_dat_exp = mut_dist_dat[mut_dist_dat['true_gene_exp_label']==1]
del mut_dist_dat
mut_dist_dat_exp['pred_expression_delta']=(mut_dist_dat_exp['pred_expression_orig'] - mut_dist_dat_exp['pred_expression'])/mut_dist_dat_exp['pred_expression']
mut_dist_dat_exp['abs_pred_expression_delta'] = mut_dist_dat_exp['pred_expression_delta'].abs()

#concentrate on upstream since downstream likely gene body
#only chr 1-22 not sex chr
upstr_mut_dist_dat_exp = mut_dist_dat_exp[(mut_dist_dat_exp['dist_tss']<0)]#, #<=-6k i.e. distal reg not prom
dwnstr_mut_dist_dat_exp = mut_dist_dat_exp[(mut_dist_dat_exp['dist_tss']>0)]#, #>=4k+ i.e. distal reg not prom
#unneeded
del mut_dist_dat_exp

xl_file = pd.ExcelFile("./qtl_ovrlp/41467_2021_23134_MOESM10_ESM.xlsx")
dfs = {sheet_name: xl_file.parse(sheet_name) 
          for sheet_name in xl_file.sheet_names}
fm_eqtl_tiss = dfs['put_causal_eqtls_tissue_unique']
#match to cells
gtex_map = pd.read_csv("./metadata/gtex_roadmap_mapping.csv")
gtex_map = gtex_map[-pd.isnull(gtex_map['cell_id'])]
fm_eqtl_tiss_filt = pd.merge(fm_eqtl_tiss,gtex_map,left_on='tissue',right_on='gtex_tissue')
fm_eqtl_tiss_filt = fm_eqtl_tiss_filt[-pd.isnull(fm_eqtl_tiss_filt['cell_id'])]
#get gene ids without number extension
fm_eqtl_tiss_filt['gene_id'] = fm_eqtl_tiss_filt.gene.str.split('.',expand=True)[0]
#get variant info
fm_eqtl_tiss_filt[['variant_chr','variant_pos',
                   'variant_a1','variant_a2']]=fm_eqtl_tiss_filt.variant.str.split(":",expand=True)
fm_eqtl_tiss_filt["variant_pos"] = pd.to_numeric(fm_eqtl_tiss_filt["variant_pos"])
#get start bp (TSS) and chrom from metadata
meta = pd.read_csv('./chromoformer/preprocessing/train.csv')[['gene_id','eid','chrom','start','end','strand']]
upstr_mut_dist_dat_exp = pd.merge(upstr_mut_dist_dat_exp,meta,left_on=['cell','gene'], right_on=['eid','gene_id'])
dwnstr_mut_dist_dat_exp = pd.merge(dwnstr_mut_dist_dat_exp,meta,left_on=['cell','gene'], right_on=['eid','gene_id'])

#calc mut reg - note this is strand specific!!!!
upstr_mut_dist_dat_exp['mut_reg_chrom']=upstr_mut_dist_dat_exp['chrom']
dwnstr_mut_dist_dat_exp['mut_reg_chrom']=dwnstr_mut_dist_dat_exp['chrom']

#deal with forward strand
#Rember dist_tss is a minus number
upstr_mut_dist_dat_exp.loc[upstr_mut_dist_dat_exp.strand=='+', 
                           'mut_reg_start']=upstr_mut_dist_dat_exp['start']+upstr_mut_dist_dat_exp['dist_tss']
upstr_mut_dist_dat_exp.loc[upstr_mut_dist_dat_exp.strand=='+',
                           'mut_reg_end']=upstr_mut_dist_dat_exp['start']+upstr_mut_dist_dat_exp['dist_tss']+2_000
dwnstr_mut_dist_dat_exp.loc[dwnstr_mut_dist_dat_exp.strand=='+', 
                           'mut_reg_start']=dwnstr_mut_dist_dat_exp['start']+dwnstr_mut_dist_dat_exp['dist_tss']
dwnstr_mut_dist_dat_exp.loc[dwnstr_mut_dist_dat_exp.strand=='+',
                           'mut_reg_end']=dwnstr_mut_dist_dat_exp['start']+dwnstr_mut_dist_dat_exp['dist_tss']+2_000

#now deal with reverse strand
#mut_reg_start is act upper boundary just revrsed since rev strand
upstr_mut_dist_dat_exp.loc[upstr_mut_dist_dat_exp.strand=='-', 
                           'mut_reg_start']=upstr_mut_dist_dat_exp['start']+(
    upstr_mut_dist_dat_exp['dist_tss']*-1)-2_000
upstr_mut_dist_dat_exp.loc[upstr_mut_dist_dat_exp.strand=='-',
                           'mut_reg_end']=upstr_mut_dist_dat_exp['start']+(
    upstr_mut_dist_dat_exp['dist_tss']*-1)
dwnstr_mut_dist_dat_exp.loc[dwnstr_mut_dist_dat_exp.strand=='-', 
                           'mut_reg_start']=dwnstr_mut_dist_dat_exp['start']+(
    dwnstr_mut_dist_dat_exp['dist_tss']*-1)-2_000
dwnstr_mut_dist_dat_exp.loc[dwnstr_mut_dist_dat_exp.strand=='-',
                           'mut_reg_end']=dwnstr_mut_dist_dat_exp['start']+(
    dwnstr_mut_dist_dat_exp['dist_tss']*-1)

#
upstr_mut_dist_dat_exp = upstr_mut_dist_dat_exp[upstr_mut_dist_dat_exp['mut_reg_chrom'].isin(
    ['chr'+str(s) for s in list(range(1,23))])]
upstr_mut_dist_dat_exp_gtex = upstr_mut_dist_dat_exp[upstr_mut_dist_dat_exp['cell'].isin(set(gtex_map.cell_id))]
#join on tissue from gtex
upstr_mut_dist_dat_exp_gtex = pd.merge(upstr_mut_dist_dat_exp,gtex_map,left_on='cell',right_on='cell_id')
#get deciles for each tissue
upstr_mut_dist_dat_exp_gtex['quantiles'] = upstr_mut_dist_dat_exp_gtex.groupby(['gtex_tissue'])['pred_expression_delta'].transform(
    lambda x: pd.qcut(x, 10, labels=range(1,11)))
dwnstr_mut_dist_dat_exp = dwnstr_mut_dist_dat_exp[dwnstr_mut_dist_dat_exp['mut_reg_chrom'].isin(
    ['chr'+str(s) for s in list(range(1,23))])]
dwnstr_mut_dist_dat_exp_gtex = dwnstr_mut_dist_dat_exp[dwnstr_mut_dist_dat_exp['cell'].isin(set(gtex_map.cell_id))]
#join on tissue from gtex
dwnstr_mut_dist_dat_exp_gtex = pd.merge(dwnstr_mut_dist_dat_exp,gtex_map,left_on='cell',right_on='cell_id')
#get deciles for each tissue
dwnstr_mut_dist_dat_exp_gtex['quantiles'] = dwnstr_mut_dist_dat_exp_gtex.groupby(['gtex_tissue'])['pred_expression_delta'].transform(
    lambda x: pd.qcut(x, 10, labels=range(1,11)))
#
#join up and dwn and tss
upstr_mut_dist_dat_exp_gtex = pd.concat([upstr_mut_dist_dat_exp_gtex,
                                         dwnstr_mut_dist_dat_exp_gtex,])
#----------------------------------------

#also use this set to get neg controls - i.e. all possible regions upstream in 2k bins
#filt the hi-c
hist_act_upstr_top['cell_gene']=hist_act_upstr_top['cell']+hist_act_upstr_top['gene']
upstr_mut_dist_dat_exp_gtex['cell_gene']=upstr_mut_dist_dat_exp_gtex['cell']+upstr_mut_dist_dat_exp_gtex['gene']
#hist_act_upstr_top = hist_act_upstr_top[hist_act_upstr_top['cell_gene'].isin(upstr_mut_dist_dat_exp_gtex['cell_gene'])]
#get strand from metadata
strnd_df = pd.read_csv('./chromoformer/preprocessing/train.csv')[['gene_id','strand']].drop_duplicates()
strnd_df.rename(columns={'gene_id':'gene'}, inplace=True)
hist_act_upstr_top = hist_act_upstr_top.merge(strnd_df,on='gene')
#54907 rows of 1 2k bin per gene remaining for all testing cell types
#now check for enrichment of the eQTL sets in these and compare to the model's top decile

#loop through each cell and compare to all peaks with bootstrapping
num_tests = 10_000
p_vals = []
tiss_tested = []
all_num_tested = []
cell_tested = set(gtex_map.cell_id)
#remove E118 as two cells link to liver, will add for this cell test
cell_tested.remove('E118')
random.seed(101)
for cell_i in cell_tested:
    print("Cell: ",cell_i)
    #get tiss
    tiss_i = gtex_map[gtex_map['cell_id']==cell_i].gtex_tissue.tolist()
    tiss_i = ', '.join(tiss_i)
    tiss_tested.append(tiss_i)
    #get cell dat
    exp_gtex_cell_i = upstr_mut_dist_dat_exp_gtex.copy()
    #get hic
    hist_act_upstr_top_cell_i = hist_act_upstr_top.copy()
    if cell_i=='E066':
        #add in other cell
        exp_gtex_cell_i = exp_gtex_cell_i[(exp_gtex_cell_i['cell']==cell_i)|(
            exp_gtex_cell_i['cell']=='E118')]
        fm_eqtl_tiss_filt_cell_i = fm_eqtl_tiss_filt[(fm_eqtl_tiss_filt['cell_id']==cell_i)|(
            fm_eqtl_tiss_filt['cell_id']=='E118')]
        hist_act_upstr_top_cell_i = hist_act_upstr_top_cell_i[(hist_act_upstr_top_cell_i['cell']==cell_i)|(
            hist_act_upstr_top_cell_i['cell']=='E118')]
    else:
        exp_gtex_cell_i = exp_gtex_cell_i[exp_gtex_cell_i['cell']==cell_i]
        fm_eqtl_tiss_filt_cell_i = fm_eqtl_tiss_filt[fm_eqtl_tiss_filt['cell_id']==cell_i]
        hist_act_upstr_top_cell_i = hist_act_upstr_top_cell_i[hist_act_upstr_top_cell_i['cell']==cell_i]
    #get total num of possible
    #join on gene, then filter to where region matches
    exp_gtex_cell_i_poss = pd.merge(exp_gtex_cell_i,fm_eqtl_tiss_filt_cell_i,
                                    left_on='gene',right_on='gene_id')
    #filt reg match
    num_poss_eqtl = exp_gtex_cell_i_poss[(exp_gtex_cell_i_poss['variant_pos'] >= exp_gtex_cell_i_poss['mut_reg_start'])&(
        exp_gtex_cell_i_poss['variant_pos'] <=exp_gtex_cell_i_poss['mut_reg_end'] )].shape[0]
    #get top - just all hic
    cell_i_top = hist_act_upstr_top_cell_i
    num_tested = cell_i_top.shape[0]
    all_num_tested.append(num_tested)
    #join on gene, then filter to where region matches
    cell_i_top = pd.merge(cell_i_top,fm_eqtl_tiss_filt_cell_i,left_on='gene',right_on='gene_id')
    #get gene start pos to work out pos of activity
    gene_pos = exp_gtex_cell_i[['gene','start','end']].drop_duplicates().reset_index(False)
    #join to data
    cell_i_top = pd.merge(cell_i_top,gene_pos,left_on='gene_x',right_on='gene',how='left')
    #get act pos of hist activity (2k bp bins)
    #need to do so based on strand
    cell_i_top.loc[cell_i_top['strand']=='+','hist_act_strt'] = cell_i_top['start']-((cell_i_top['variable']+1)*2_000)
    cell_i_top.loc[cell_i_top['strand']=='+','hist_act_end'] = cell_i_top['start']-(cell_i_top['variable']*2_000)
    cell_i_top.loc[cell_i_top['strand']=='-','hist_act_end'] = cell_i_top['start']+((cell_i_top['variable']+1)*2_000)
    cell_i_top.loc[cell_i_top['strand']=='-','hist_act_strt'] = cell_i_top['start']+(cell_i_top['variable']*2_000) 
    #filt reg match
    cell_i_top = cell_i_top[(cell_i_top['variant_pos'] >= cell_i_top['hist_act_strt']) & (
        cell_i_top['variant_pos'] <=cell_i_top['hist_act_end'] )]
    num_top_eqtl = cell_i_top.shape[0]
    top_prop = num_top_eqtl/num_poss_eqtl
    #compare against all randomly sampled ones (bootstraping)
    neg_bs_res = bootstrap_neg(df_test=exp_gtex_cell_i,df_eqtl=fm_eqtl_tiss_filt_cell_i,
                               num_regs=num_tested,num_poss_regs=num_poss_eqtl,
                               num_tests=num_tests)
    #To get a p like value(called probability): 
    #count all background with greater proportion than top list. 
    p_val = np.sum(top_prop<np.array(neg_bs_res))/num_tests
    p_vals.append(p_val)

#nice naming 
tiss_tested_n = ['Cerebellum & Hippocampus' if x=='Brain_Cerebellum, Brain_Hippocampus' else x for x in tiss_tested]
tiss_tested_n = ['Lymphocytes' if x=='Cells_EBV-transformed_lymphocytes' else x for x in tiss_tested_n]

def p_adjust_bh(p):
    """Benjamini-Hochberg p-value correction for multiple hypothesis testing."""
    p = np.asfarray(p)
    by_descend = p.argsort()[::-1]
    by_orig = by_descend.argsort()
    steps = float(len(p)) / np.arange(len(p), 0, -1)
    q = np.minimum(1, np.minimum.accumulate(steps * p[by_descend]))
    return q[by_orig]

quant_i = 1

p_adjust = list(p_adjust_bh(p_vals))    

bst_act_res = pd.DataFrame({"quantile":[quant_i] * len(p_vals),
                            "cell":list(cell_tested),
                            "tissue":tiss_tested,
                            "tissue_nice":tiss_tested_n,
                            "num_positions_tested":all_num_tested,
                            "p_value":p_vals,
                            "FDR_p":p_adjust})
bst_act_res.to_csv("./qtl_ovrlp/all_quant_bootstrap_hist_act_res.csv",index=False)