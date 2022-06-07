# utils.py

#Data loading, training, test split functions ----------------------------------------------------------

import os
import pathlib
from typing import Sequence, Union

import math
import numpy as np
import pandas as pd
import random
import pyBigWig
import pyensembl

import itertools
import tensorflow as tf

from epi_to_express.constants import (
    CELLS,
    ALLOWED_CELLS,
    ALLOWED_FEATURES,
    CHROMOSOME_DATA,
    CHROMOSOMES,
    CHROM_LEN,
    DNASE_DATA,
    H3K4ME1_DATA,
    H3K27AC_DATA,
    H3K4ME3_DATA,
    H3K9ME3_DATA,
    H3K27ME3_DATA,
    H3K36ME3_DATA,
    H3K9AC_DATA,
    H3K4ME2_DATA,
    H2A_DATA,
    H3K79ME2_DATA,
    H4K20ME1_DATA,
    EXP_PTH
)


def train_valid_split(chromosomes, chrom_len, samples, valid_frac, split):
    """
    Function to create a train validatin split between chromosomes.
    Takes list of chromosomes, list of their lengths, the list of
    sample types and the fraction dedicated to validation data.
    Split type is either CHROM, SAMPLE or BOTH and
    determines if specific chromosomes are used for
    train/test or if its split by samples or if both are used
    """

    def full_index_dist(n,chrom_len=None):
        index = np.asarray([x for x in range(0, n)])
        dist = np.ones(n)/len(index) #proportions
        if chrom_len is not None:
            dist = chrom_len/sum(chrom_len)
        return index, dist

    def sample_index_dist(n, frac, chrom_len=None):
        """
        If chr (chromosome is true) then use the len of each 
        chromosome in the selection dist
        """
        tr_count = int(n*(1-frac))
        tr_index = random.sample(range(0, n), tr_count)
        ts_index = [x for x in range(0, n) if x not in tr_index]
        tr_dist = np.ones(tr_count)/tr_count
        ts_dist = np.ones(n-tr_count)/(n-tr_count)
        if chrom_len is not None:
            tr_dist = chrom_len[tr_index]/sum(chrom_len[tr_index])
            ts_dist = chrom_len[ts_index]/sum(chrom_len[ts_index])
        return tr_index, ts_index, tr_dist, ts_dist

    # RUN PIECE OF CODE BELOW BUT CHANGE CHROMOSOMES TO THE RELEVANT
    # TYPE OF DATA. ALSO REPROGRAM dist selection depending on
    # split type.
    if split == 'CHROM':
        (c_train_index, c_valid_index,
         c_train_dist, c_valid_dist) = sample_index_dist(len(chromosomes),
                                                        valid_frac, chrom_len)
        s_train_index, s_train_dist = full_index_dist(len(samples))
        s_valid_index, s_valid_dist = full_index_dist(len(samples))
    if split == 'SAMPLE':
        (s_train_index, s_valid_index,
         s_train_dist, s_valid_dist) = sample_index_dist(len(samples),
                                                        valid_frac)
        c_train_index, c_train_dist = full_index_dist(len(chromosomes),chrom_len)
        c_valid_index, c_valid_dist = full_index_dist(len(chromosomes),chrom_len)
    if split == 'BOTH':
        (s_train_index, s_valid_index,
         s_train_dist, s_valid_dist) = sample_index_dist(len(samples),
                                                        valid_frac)
        (c_train_index, c_valid_index,
         c_train_dist, c_valid_dist) = sample_index_dist(len(chromosomes),
                                                        valid_frac, chrom_len)
    return (s_train_index, s_valid_index, c_train_index, c_valid_index,
            s_train_dist, s_valid_dist, c_train_dist, c_valid_dist)


def get_path(cell: str, feature: str, pred_res: int) -> pathlib.Path:
    """Looks up path for given feature of a cell in the data paths"""
    # Get full length cell name from synonym
    #get key from value
    cell=list(CELLS.keys())[list(CELLS.values()).index(cell)]
    # Load feature path
    # datasets have been averaged to a certain bp, included in name
    if feature == "dnase":
        return DNASE_DATA[cell+'_'+str(pred_res)]
    elif feature == "h3k27ac":
        return H3K27AC_DATA[cell+'_'+str(pred_res)]
    elif feature == "h3k4me1":
        return H3K4ME1_DATA[cell+'_'+str(pred_res)]
    elif feature == "h3k4me3":
        return H3K4ME3_DATA[cell+'_'+str(pred_res)]
    elif feature == "h3k9me3":
        return H3K9ME3_DATA[cell+'_'+str(pred_res)]
    elif feature == "h3k27me3":
        return H3K27ME3_DATA[cell+'_'+str(pred_res)]
    elif feature == "h3k36me3":
        return H3K36ME3_DATA[cell+'_'+str(pred_res)]
    elif feature == "h3k9ac":
        return H3K9AC_DATA[cell+'_'+str(pred_res)]
    elif feature == "h3k4me2":
        return H3K4ME2_DATA[cell+'_'+str(pred_res)]
    elif feature == "h2a":
        return H2A_DATA[cell+'_'+str(pred_res)]
    elif feature == "h3k79me2":
        return H3K79ME2_DATA[cell+'_'+str(pred_res)]
    elif feature == "h4k20me1":
        return H4K20ME1_DATA[cell+'_'+str(pred_res)]
    else:
        raise ValueError(
            f"Feature {feature} not allowed. Allowed features are {ALLOWED_FEATURES}"
        )

def load_bigwig(path: Union[os.PathLike, str], decode: bool = False):
    """Loads bigwig from a pathlike object"""
    path = str(path)
    if decode:
        try:
            path = path.decode("utf-8")
        except:
            pass
    return pyBigWig.open(path)

class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, cell, chromosomes, features, window_size, 
                 pred_res, batch_size=32, shuffle=True, debug=False,
                 test_subset=False):
        'Initialization'
        self.cell = cell
        self.cell_ids = [list(CELLS.keys())[list(CELLS.values()).index(i)] for i in cell]
        self.chromosomes = chromosomes
        self.features = features
        self.window_size = window_size
        self.pred_res = pred_res
        assert self.window_size % self.pred_res ==0,"pred_res must be divis by window_size"
        assert self.window_size % 2 ==0,"window_size must be divis by 2"
        self.batch_size = batch_size
        self.n_classes = len(cell)
        self.shuffle = shuffle
        self.debug = debug
        self.test_subset = test_subset
        #load exp data
        self.rm_exp = pd.read_csv(str(EXP_PTH / '57epigenomes.RPKM.pc.gz'),
                                  delimiter="\t")[self.cell_ids]
        #call func to get get info and save values to self.genes_info
        self.get_genes()
        #call func to create connection to bigwigs
        self.initiate_bigwigs()
        #shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.genes_info) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        genes_info_temp = [self.genes_info[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(genes_info_temp)

        return X, y
    
    def get_genes(self):
        'Given the chromosomes return the genes to pred'
        gene_info = []
        excl_inds = []
        for ind,ensembl_id in enumerate(list(self.rm_exp.index)):
            try:
                gene = pyensembl.ensembl_grch37.gene_by_id(ensembl_id)
                #excl sex chrome
                if(gene.contig=='X' or gene.contig=='Y'):
                    excl_inds.append(ind)
                else:
                    #save chr, start, end as tuple
                    gene_info.append((ensembl_id,'chr'+gene.contig, 
                                      gene.start, gene.end))
            except:
                #save ind where gene not found to remove
                excl_inds.append(ind)
        #return valid genes on specified chromosomes with pos info
        if(self.test_subset):
            self.genes_info = [t for t in gene_info if t[1] in self.chromosomes][0:256]
        else:
            self.genes_info = [t for t in gene_info if t[1] in self.chromosomes]
    
    def initiate_bigwigs(self):
        """
        Initiate the connection to the bigwigs
        """
        # Input verification
        if not all(np.isin(self.cell, ALLOWED_CELLS)):
            raise ValueError(
                "Cell types contain values which are not allowed. "
                f"Allowed cell types are {ALLOWED_CELLS}."
            )
        if not all(np.isin(self.features, ALLOWED_FEATURES)):
            raise ValueError(
                "Features contain values which are not allowed. "
                f"Allowed features are {ALLOWED_FEATURES}."
            )
        assert len(self.features) > 0, "Must provide at least one feature"

        # Load file handles for all cells and features
        self.data_conn = {
            cell_i: {
                feature: load_bigwig(get_path(cell_i, feature, self.pred_res))
                for feature in self.features
            }
            for cell_i in self.cell
        }
    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.genes_info))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, genes_info_temp):
        'Generates data containing batch_size samples'
        # Initialization
        X = np.empty((len(genes_info_temp),#same as self.batch_size, 
                      self.window_size//self.pred_res,
                     len(self.features)))
        y = np.empty((len(genes_info_temp),#self.batch_size
                     ), dtype=float)
        
        # Generate data
        for i, gene in enumerate(genes_info_temp):
            gene_start = gene[2]
            chrom = gene[1]
            #work out window strt & end
            #make sure within chromesome bounds
            window_strt = max(gene_start - self.window_size//2,0)
            window_end = min(gene_start + self.window_size//2,
                             int(CHROM_LEN[np.where(CHROMOSOMES==chrom)[0][0]]))
            for j, feature in enumerate(self.features):
                # load data
                if (self.debug):
                    print(f"Cell: {self.cell[0]}")
                    print(f"Feature: {feature}")
                    print(f"Selected chromosome: {chrom}")
                    print(f"Window start: {window_strt}")
                    print(f"Window end: {window_end}")
                dat=np.nan_to_num(
                        #self.cell[0] - since again only 1 cell
                        self.data_conn[self.cell[0]][feature].values(
                            chrom,
                            window_strt,
                            window_end
                        ))
                #pad with zero's if hit end chro or strt
                org_strt=gene_start - self.window_size//2
                org_end=gene_start + self.window_size//2
                chr_end=int(CHROM_LEN[np.where(CHROMOSOMES==chrom)[0][0]])
                if(org_strt<0):
                    strt_pad = np.zeros(org_strt*-1)
                    dat = np.concatenate([strt_pad,dat])
                if(chr_end-org_end<0):
                    end_pad = np.zeros(org_end-chr_end)
                    dat = np.concatenate([dat,end_pad])
                X[i,:,j] = np.arcsinh(np.mean( 
                #X[i,:] = np.arcsinh(np.mean( 
                    dat.reshape(-1, self.pred_res),#averaging at desired pred_res  
                    axis=1))

            # Store exp value - only one value so can use [0]
            #mean normalised
            normalized_df=(self.rm_exp-self.rm_exp.mean())/self.rm_exp.std()
            #max min normalised
            #normalized_df=(self.rm_exp-self.rm_exp.min())/(self.rm_exp.max()-self.rm_exp.min())
            y[i] = normalized_df.loc[gene[0]][0]
        return X, y
    
    
    
    