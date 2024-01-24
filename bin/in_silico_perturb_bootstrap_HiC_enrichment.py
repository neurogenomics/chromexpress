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
from epi_to_express.constants import (
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

#need to also get Hi-C data and check if this just captures the enrichment..

hic_relationships = pd.read_csv("./chromoformer/preprocessing/train.csv")
#not using scores - I guess this is stength of connection
hic_relationships = hic_relationships[["gene_id","eid","chrom","start","end","neighbors"]]

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

#split so one row per connection
hic_relationships
hic_relationships["neighbors"] = hic_relationships["neighbors"].str.split(";")
hic_relationships = hic_relationships.explode("neighbors").reset_index(drop=True)
hic_relationships["neighbor_chrom"] = hic_relationships["neighbors"].str.split(":").str[0]
hic_relationships["neighbor_start"] = hic_relationships["neighbors"].str.split(":").str[1]
hic_relationships[["neighbor_start","neighbor_end"]] = hic_relationships["neighbor_start"].str.split("-", 
                                                                                                     expand=True)
#make sure hi-c on same chrom, in chrom 1-22 and upstream
hic_relationships = hic_relationships[hic_relationships['chrom']==hic_relationships['neighbor_chrom']]
hic_relationships = hic_relationships[hic_relationships['chrom'].isin(["chr"+str(i) for i in range(1,23)])]
hic_relationships['neighbor_start'] = hic_relationships['neighbor_start'].astype(int)
hic_relationships['neighbor_end'] = hic_relationships['neighbor_end'].astype(int)
hic_relationships = hic_relationships[hic_relationships['neighbor_start']<hic_relationships['start']]
#filter to just the cells to be tested
hic_relationships = hic_relationships[hic_relationships['eid'].isin(set(fm_eqtl_tiss_filt['cell_id']))]
#split into 2k bins to match
hic_relationships['neighbor_length'] = hic_relationships['neighbor_end'] - hic_relationships['neighbor_start']
hic_relationships['neighbor_num_2k'] = hic_relationships['neighbor_length']//2_000
hic_relationships['neighbor_num_2k_count'] = 1
hic_relationships = hic_relationships.reset_index(drop=True)
#repeat rows
hic_relationships = hic_relationships.loc[hic_relationships.index.repeat(hic_relationships['neighbor_num_2k'])]
#group by index with transform for date ranges
hic_relationships['neighbor_num_2k_inst'] =(hic_relationships.groupby(level=0)['neighbor_num_2k_count']
                                            .transform(lambda x: range(len(x))))
#unique default index
hic_relationships = hic_relationships.reset_index(drop=True)
#correct start and end
hic_relationships['corr_neighbor_start'] = hic_relationships['neighbor_start']+2_000*hic_relationships['neighbor_num_2k_inst']
hic_relationships['corr_neighbor_end'] = hic_relationships['neighbor_start']+2_000*(hic_relationships['neighbor_num_2k_inst']+1)
hic_relationships
#keep regions between 6k and 20k upstream to match analysis
hic_relationships = hic_relationships[(hic_relationships['corr_neighbor_end']+6_000<=hic_relationships['start']) & (
    hic_relationships['corr_neighbor_start']>=hic_relationships['start']-20_000)]
#need to match to the same genes and cell types as used in pred bootstrap tests
#----------------------------------------
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
mut_dist_dat_exp['pred_expression_delta']=(mut_dist_dat_exp['pred_expression_orig'] - mut_dist_dat_exp['pred_expression'])/mut_dist_dat_exp['pred_expression']
mut_dist_dat_exp['abs_pred_expression_delta'] = mut_dist_dat_exp['pred_expression_delta'].abs()
#concentrate on upstream since downstream likely gene body
#only chr 1-22 not sex chr
upstr_mut_dist_dat_exp = mut_dist_dat_exp[mut_dist_dat_exp['dist_tss']<0]#, #-6k and back i.e. distal reg not prom or gene body
#unneeded
del mut_dist_dat,mut_dist_dat_exp
#get start bp (TSS) and chrom from metadata
meta = pd.read_csv('./chromoformer/preprocessing/train.csv')[['gene_id','eid','chrom','start','end']]
upstr_mut_dist_dat_exp = pd.merge(upstr_mut_dist_dat_exp,meta,left_on=['cell','gene'], right_on=['eid','gene_id'])
#calc mut reg
upstr_mut_dist_dat_exp['mut_reg_chrom']=upstr_mut_dist_dat_exp['chrom']
upstr_mut_dist_dat_exp['mut_reg_start']=upstr_mut_dist_dat_exp['start']+upstr_mut_dist_dat_exp['dist_tss']
upstr_mut_dist_dat_exp['mut_reg_end']=upstr_mut_dist_dat_exp['start']+upstr_mut_dist_dat_exp['dist_tss']+2_000
upstr_mut_dist_dat_exp = upstr_mut_dist_dat_exp[upstr_mut_dist_dat_exp['mut_reg_chrom'].isin(
    ['chr'+str(s) for s in list(range(1,23))])]
upstr_mut_dist_dat_exp_gtex = upstr_mut_dist_dat_exp[upstr_mut_dist_dat_exp['cell'].isin(set(gtex_map.cell_id))]
#join on tissue from gtex
upstr_mut_dist_dat_exp_gtex = pd.merge(upstr_mut_dist_dat_exp,gtex_map,left_on='cell',right_on='cell_id')
#get deciles for each tissue
upstr_mut_dist_dat_exp_gtex['quantiles'] = upstr_mut_dist_dat_exp_gtex.groupby(['gtex_tissue'])['pred_expression_delta'].transform(
    lambda x: pd.qcut(x, 10, labels=range(1,11)))
#----------------------------------------
#also use this set to get neg controls - i.e. all possible regions upstream in 2k bins
#filt the hi-c
hic_relationships['cell_gene']=hic_relationships['eid']+hic_relationships['gene_id']
upstr_mut_dist_dat_exp_gtex['cell_gene']=upstr_mut_dist_dat_exp_gtex['cell']+upstr_mut_dist_dat_exp_gtex['gene']
hic_relationships = hic_relationships[hic_relationships['cell_gene'].isin(upstr_mut_dist_dat_exp_gtex['cell_gene'])]
#rename cols to make it align 
hic_relationships.rename(columns={'eid':'cell'}, inplace=True)
hic_relationships.rename(columns={'gene_id':'gene'}, inplace=True)
#8616 Hi-C 2k bin interactions remaining for all testing cell types
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
    hic_relationships_cell_i = hic_relationships.copy()
    if cell_i=='E066':
        #add in other cell
        exp_gtex_cell_i = exp_gtex_cell_i[(exp_gtex_cell_i['cell']==cell_i)|(
            exp_gtex_cell_i['cell']=='E118')]
        fm_eqtl_tiss_filt_cell_i = fm_eqtl_tiss_filt[(fm_eqtl_tiss_filt['cell_id']==cell_i)|(
            fm_eqtl_tiss_filt['cell_id']=='E118')]
        hic_relationships_cell_i = hic_relationships_cell_i[(hic_relationships_cell_i['cell']==cell_i)|(
            hic_relationships_cell_i['cell']=='E118')]
    else:
        exp_gtex_cell_i = exp_gtex_cell_i[exp_gtex_cell_i['cell']==cell_i]
        fm_eqtl_tiss_filt_cell_i = fm_eqtl_tiss_filt[fm_eqtl_tiss_filt['cell_id']==cell_i]
        hic_relationships_cell_i = hic_relationships_cell_i[hic_relationships_cell_i['cell']==cell_i]
    #get total num of possible
    #join on gene, then filter to where region matches
    exp_gtex_cell_i_poss = pd.merge(exp_gtex_cell_i,fm_eqtl_tiss_filt_cell_i,
                                    left_on='gene',right_on='gene_id')
    #filt reg match
    num_poss_eqtl = exp_gtex_cell_i_poss[(exp_gtex_cell_i_poss['variant_pos'] >= exp_gtex_cell_i_poss['mut_reg_start'])&(
        exp_gtex_cell_i_poss['variant_pos'] <=exp_gtex_cell_i_poss['mut_reg_end'] )].shape[0]
    #get top - just all hic
    cell_i_top = hic_relationships_cell_i
    num_tested = cell_i_top.shape[0]
    all_num_tested.append(num_tested)
    #join on gene, then filter to where region matches
    cell_i_top = pd.merge(cell_i_top,fm_eqtl_tiss_filt_cell_i,left_on='gene',right_on='gene_id')
    #filt reg match
    cell_i_top = cell_i_top[(cell_i_top['variant_pos'] >= cell_i_top['neighbor_start']) & (
        cell_i_top['variant_pos'] <=cell_i_top['neighbor_end'] )]
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

from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import FloatVector

stats = importr('stats')

p_adjust = list(stats.p_adjust(FloatVector(p_vals), method = 'BH'))    
bst_hic_res = pd.DataFrame({"quantile":[quant_i] * len(p_vals),
                            "cell":list(cell_tested),
                            "tissue":tiss_tested,
                            "tissue_nice":tiss_tested_n,
                            "num_positions_tested":all_num_tested,
                            "p_value":p_vals,
                            "FDR_p":p_adjust})
bst_hic_res.to_csv("./qtl_ovrlp/all_quant_bootstrap_hi_c_res.csv",index=False)