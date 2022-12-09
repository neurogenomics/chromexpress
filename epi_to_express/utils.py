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
    CHROM_LEN
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
                 test_subset=False,hist_mark_pos='around'):
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
        self.hist_mark_pos = hist_mark_pos
        #load exp data - remove cells that we aren't studying
        self.rm_exp = pd.read_csv(str(EXP_PTH / '57epigenomes.RPKM.pc.gz'),
                                  delimiter="\t")[CELLS]#[self.cell_ids]
        #call func to get get info and save values to self.genes_info
        self.get_genes()
        #call func to create connection to bigwigs
        self.initiate_bigwigs()
        #shuffle
        self.indexes = np.arange(len(self.genes_info))
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
            if(self.hist_mark_pos=='around'):
                window_strt = max(gene_start - self.window_size//2,0)
                window_end = min(gene_start + self.window_size//2,
                                 int(CHROM_LEN[np.where(CHROMOSOMES==chrom)[0][0]]))
            else:#'upstream'
                window_strt = max(gene_start - self.window_size,0)
                window_end = gene_start
                
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
            #normalized_df=(self.rm_exp-self.rm_exp.mean())/self.rm_exp.std()
            #max min normalised
            normalized_df=(self.rm_exp-self.rm_exp.min())/(self.rm_exp.max()-self.rm_exp.min())
            #filter to cells of interest- NB are using all expression data here including blind test for norm step
            normalized_df = normalized_df[self.cell_ids]
            y[i] = normalized_df.loc[gene[0]][0]
        return X, y
    

import tensorflow as tf
import pandas as pd
import numpy as np
import math
from pathlib import Path
from epi_to_express.constants import PROJECT_PATH, ASSAYS
train_dir = PROJECT_PATH/'chromoformer'/'preprocessing'
train_meta = train_dir / 'train.csv'

import tensorflow as tf
import pandas as pd
import numpy as np
import math
from pathlib import Path
from epi_to_express.constants import PROJECT_PATH, ASSAYS
train_dir = PROJECT_PATH/'chromoformer'/'preprocessing'
train_meta = train_dir / 'train.csv'

class Roadmap3D_tf(tf.keras.utils.Sequence):
    """
    Adaption of chromoformer's pytorch dataloader
    to work with tensorflow
    """
    def __init__(self, cell, target_genes, batch_size, shuffle = True, 
                 i_max=8, w_prom=40_000, w_max=40_000, marks = ASSAYS,
                 pred_res = 100,y_type='log2RPKM',return_pcres=False):
        self.eid = cell
        self.target_genes = target_genes # List of ENSGs.
        meta = pd.read_csv(train_meta)
        #expression in meta is raw expression (RPKM) taken from here:
        #https://egg2.wustl.edu/roadmap/data/byDataType/rna/expression/57epigenomes.RPKM.pc.gz
        #need to convert to log2, add tiny value so not getting log2 of zero
        meta.expression = np.log2(meta.expression+5e-324)
        self.meta = meta[meta.eid == self.eid].reset_index(drop=True)
        self.ensg2label = {r.gene_id:r.label for r in self.meta.to_records()}
        #log2 RPKM
        self.ensg2exp = {r.gene_id:r.expression for r in self.meta.to_records()}
        self.ensg2tss, self.ensg2pcres, self.ensg2scores = {}, {}, {}
        for r in self.meta.to_records():
            self.ensg2tss[r.gene_id] = (r.chrom, r.start, r.end, r.strand) 
            if not pd.isnull(r.neighbors):
                self.ensg2pcres[r.gene_id] = r.neighbors.split(';')
                self.ensg2scores[r.gene_id] = [float(s) for s in r.scores.split(';')]
            else:
                self.ensg2pcres[r.gene_id] = []
                self.ensg2scores[r.gene_id] = []
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.i_max = i_max  # Maximum number of cis-interacting pCREs.
        self.w_prom = w_prom  # Promoter window size.
        self.w_max = w_max  # Maximum size of pCRE to consider.
        self.marks = marks
        self.pred_res = pred_res
        assert type(self.pred_res)!=list, "Can only handle one pred_res currently not a list"
        self.y_type = y_type
        self.return_pcres = return_pcres
        self.all_ys = ['label','log2RPKM']
        self.indices = np.arange(len(self.target_genes))
        self.on_epoch_end()
        
    def _bin_and_pad(self, x, bin_size, max_n_bins):
        """Given a 2D tensor x, make binned tensor by taking average values of 
        `bin_size` consecutive values.
        Appropriately pad by 
        left_pad = ceil((max_n_bins - n_bins) / 2)
        right_pad = floor((max_n_bins - n_bins) / 2)
        """
        l = tf.shape(x)[1]
        n_bins = math.ceil(l / bin_size)

        # Binning.
        x_binned = []
        for i in range(n_bins):
            b = tf.math.reduce_mean(x[:, i * bin_size: (i + 1) * bin_size],
                                    axis=1,keepdims=True)
            b = tf.math.log(b + 1)
            x_binned.append(b)
        x_binned = tf.concat(x_binned, axis=1)

        # Padding.
        left_pad = math.ceil( (max_n_bins - n_bins) / 2 )
        right_pad = math.floor( (max_n_bins - n_bins) / 2 )
        x_binned = tf.concat([
            tf.zeros([tf.shape(x)[0], left_pad]),
            x_binned,
            tf.zeros([tf.shape(x)[0], right_pad]),
        ], axis=1)
        #x_binned = np.concatenate([
        #    np.zeros([x.shape[0], left_pad]),
        #    x_binned,
        #    np.zeros([x.shape[0], right_pad]),
        #], axis=1)

        return x_binned, left_pad, n_bins, right_pad
    
    def _get_region_representation(self, chrom, start, end, bin_size, max_n_bins, 
                                   strand='+', window=None):
        #filter to assays of interest
        marks_ind = [ASSAYS.index(i) for i in self.marks]
        
        x = tf.cast(tf.convert_to_tensor(np.load(str(train_dir / 
                                             f'data/{self.eid}/{chrom}_{start}-{end}.npy'))[marks_ind,:]),
                    tf.float32)
        #x = np.load(str(train_dir / f'data/{self.eid}/{chrom}_{start}-{end}.npy'))[marks_ind,:]
        if window is not None:
            x = x[:, 20000 - window // 2:20000 + window // 2]
        x, left_pad, n_bins, right_pad = self._bin_and_pad(x, bin_size, max_n_bins)

        if strand == '+':
            return x, left_pad, n_bins, right_pad
        else:
            return x[:,::-1], right_pad, n_bins, left_pad

    def _split_interval_string(self, interval_string):
        """
        chrom:start-end -> (chrom, start, end)
        """
        chrom, tmp = interval_string.split(':')
        start, end = map(int, tmp.split('-'))

        return chrom, start, end

    def __getitem__(self, index):
        """
        Get data for all batch
        """
        # Generate indexes of the batch
        btch_ind = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        # Get data
        X, y = self.__generate_dat(btch_ind)
        
        return X,y    
        
    def __generate_dat(self, btch_ind):
        """
        Get next batch
        """
        y = []
        X_x_p_bin_size = []
        X_x_pcre_bin_size = []
        X_pad_mask_p_bin_size = []
        X_pad_mask_pcre_bin_size = []
        X_interaction_freq = []
        X_interaction_mask_bin_size = []
        for index in btch_ind:
            target_gene = self.target_genes[index]
            assert target_gene in self.ensg2tss, f"Unknown gene - {target_gene}"

            pcres, scores = self.ensg2pcres[target_gene], self.ensg2scores[target_gene]
            n_partners, n_dummies = len(pcres), self.i_max - len(pcres)

            if self.y_type=='label':
                #y.append(tf.constant(self.ensg2label[target_gene],shape=(1,1),dtype=tf.int32))
                y.append(np.asarray([[self.ensg2label[target_gene]]],dtype=np.int32))
            elif self.y_type=='log2RPKM':
                #y.append(tf.constant(self.ensg2exp[target_gene],shape=(1,1),dtype=tf.float32))
                y.append(np.asarray([[self.ensg2exp[target_gene]]],dtype=np.float32))
            else:
                assert y_type in self.all_ys, "Y choice not supported (y_type). Should be one of log2RPKM or label."

            chrom_p, start_p, end_p, strand_p = self.ensg2tss[target_gene]
            start_p, end_p = start_p - 20000, start_p + 20000
            
            #can only handle 1 bins size
            bin_size = self.pred_res
            max_n_bins = self.w_max // bin_size
            x_pcres, mask_pcres = [], []
            #this gets histone mark data
            #base-pair resolution data available in 'chromoformer/preprocessing/hist/E003-H3K36me3.npz'
            #these are referred to as read_depths
            x_p, left_pad_p, n_bins_p, right_pad_p = self._get_region_representation(chrom_p, 
                                                                                     start_p, 
                                                                                     end_p, 
                                                                                     bin_size, 
                                                                                     max_n_bins, 
                                                                                     strand_p, 
                                                                                     window=self.w_prom)
            #x_p = tf.expand_dims(tf.transpose(x_p,[1, 0]),axis=0) # 1 x max_n_bins x 7
            x_p = np.expand_dims(np.transpose(x_p,[1, 0]),axis=0) # 1 x max_n_bins x 7
            mask_p = np.ones([1, max_n_bins, max_n_bins], dtype=bool)
            mask_p[0, left_pad_p:left_pad_p + n_bins_p, left_pad_p:left_pad_p + n_bins_p] = 0
            mask_p = tf.convert_to_tensor(mask_p)
            mask_p = tf.expand_dims(mask_p,axis=0)

            interaction_freq = tf.zeros([self.i_max + 1, self.i_max + 1])

            if self.return_pcres:
                for i, (score, pcre) in enumerate(zip(scores, pcres)):

                    chrom_pcre, start_pcre, end_pcre = self._split_interval_string(pcre)
                    if end_pcre - start_pcre > 40000:
                        print(target_gene, chrom_pcre, start_pcre, end_pcre)

                    x_pcre, left_pad_pcre, n_bins_pcre, right_pad_pcre = self._get_region_representation(chrom_pcre, 
                                                                                                         start_pcre, 
                                                                                                         end_pcre, 
                                                                                                         bin_size, 
                                                                                                         max_n_bins)

                    mask_pcre = tf.ones([1, max_n_bins, max_n_bins], dtype=tf.bool)
                    mask_pcre[0, left_pad_p:left_pad_p + n_bins_p, 
                              left_pad_pcre:left_pad_pcre + n_bins_pcre] = 0

                    x_pcres.append(x_pcre)
                    mask_pcres.append(mask_pcre)

                    interaction_freq[0, i + 1] = score

                x_pcres.append(tf.zeros([7, n_dummies * max_n_bins]))
                x_pcres = tf.reshape(tf.concat(x_pcres, axis=1),[-1, self.i_max, max_n_bins])
                x_pcres = tf.transpose(x_pcres,[1, 2, 0]) # i_max x max_n_bins x 7
                for _ in range(n_dummies):
                    m = tf.ones([1, max_n_bins, max_n_bins], dtype=tf.bool)
                    mask_pcres.append(m)

            interaction_mask = np.ones([self.i_max + 1, self.i_max + 1], dtype=bool)
            interaction_mask[:n_partners + 1, :n_partners + 1] = 0
            interaction_mask = tf.convert_to_tensor(interaction_mask)

            X_x_p_bin_size.append(x_p) # 1 x max_n_bins x 7
            if self.return_pcres:
                X_x_pcre_bin_size.append(x_pcres) # i_max x max_n_bins x 7
                X_pad_mask_p_bin_size.append(mask_p) # 1 x max_n_bins x max_n_bins
                X_pad_mask_pcre_bin_size.append(tf.stack(mask_pcres)) # i_max x max_n_bins x max_n_bins
                X_interaction_mask_bin_size.append(tf.expand_dims(interaction_mask, axis=0))
                X_interaction_freq.append(tf.expand_dims(interaction_freq, 0))
        
        #combine all
        if self.return_pcres:
            X = ({'x_p_pred_res':tf.concat(X_x_p_bin_size,axis=0),
                  'x_pcre_pred_res':tf.concat(X_x_pcre_bin_size,axis=0),
                  'pad_mask_p_pred_res':tf.concat(X_pad_mask_p_bin_size,axis=0),
                  'pad_mask_pcre_pred_res':tf.concat(X_pad_mask_pcre_bin_size,axis=0),
                  'interaction_freq':tf.concat(X_interaction_freq,axis=0),
                  'interaction_mask_pred_res':tf.concat(X_interaction_mask_bin_size,axis=0)
                  })
        else:
            X = ({'x_p_pred_res':tf.concat(X_x_p_bin_size,axis=0)})
            #X = ({'x_p_pred_res':np.concatenate(X_x_p_bin_size,axis=0)})
        y = tf.concat(y,axis=0)
        #y = np.concatenate(y,axis=0)
        
        return X,y

    def __len__(self):
        return len(self.target_genes)//self.batch_size
    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle == True:
            np.random.shuffle(self.indices)

    
    
