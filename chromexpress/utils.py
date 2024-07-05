# utils.py

#Data loading, training, test split functions ----------------------------------------------------------

import os
import pathlib
from typing import Sequence, Union

import math
import numpy as np
import pandas as pd
import random
import pyensembl

import itertools
import tensorflow as tf

from chromexpress.constants import (
    CELLS,
    ALLOWED_CELLS,
    ALLOWED_FEATURES,
    CHROMOSOME_DATA,
    CHROMOSOMES,
    CHROM_LEN
)

#Pearson R Implementation ----------------------------------------------------
class pearsonR(tf.keras.metrics.Metric): 
    def __init__(self, name="correlation", **kwargs): 
        super(pearsonR, self).__init__(name=name, **kwargs)
        self.correlation = self.add_weight(name="correlation", initializer="zeros")
        self.n = self.add_weight(name="n", initializer="zeros")
        self.x = self.add_weight(name="x", initializer="zeros")
        self.x_squared = self.add_weight(name="x_squared", initializer="zeros")
        self.y = self.add_weight(name="y", initializer="zeros")
        self.y_squared = self.add_weight(name="y_squared", initializer="zeros")
        self.xy = self.add_weight(name="xy", initializer="zeros")
        
    def update_state(self, y_true, y_pred, sample_weight=None): 
        self.n.assign_add(tf.reduce_sum(tf.cast((y_pred == y_true), "float32")))
        self.n.assign_add(tf.reduce_sum(tf.cast((y_pred != y_true), "float32")))
        self.xy.assign_add(tf.reduce_sum(tf.multiply(y_pred, y_true)))
        self.x.assign_add(tf.reduce_sum(y_pred))
        self.y.assign_add(tf.reduce_sum(y_true))
        self.x_squared.assign_add(tf.reduce_sum(tf.math.square(y_pred)))
        self.y_squared.assign_add(tf.reduce_sum(tf.math.square(y_true)))
        
    def result(self): 
        return (self.n*self.xy - self.x*self.y)/tf.math.sqrt((self.n*self.x_squared - tf.math.square(self.x))*(self.n*self.y_squared - tf.math.square(self.y)))
        
    def reset_state(self): 
        self.n.assign(0.0)
        self.x.assign(0.0)
        self.x_squared.assign(0.0)
        self.y.assign(0.0)
        self.y_squared.assign(0.0)
        self.xy.assign(0.0)
        self.correlation.assign(0.0)

import tensorflow as tf
import pandas as pd
import numpy as np
import math
from pathlib import Path
from chromexpress.constants import PROJECT_PATH, ASSAYS
train_dir = PROJECT_PATH/'chromoformer'/'preprocessing'
train_meta = train_dir / 'train.csv'

import tensorflow as tf
import pandas as pd
import numpy as np
import math
from pathlib import Path
from chromexpress.constants import PROJECT_PATH, ASSAYS
train_dir = PROJECT_PATH/'chromoformer'/'preprocessing'
train_meta = train_dir / 'train.csv'

class Roadmap3D_tf(tf.keras.utils.Sequence):
    """
    Adaption of chromoformer's pytorch dataloader
    to work with tensorflow
    """
    def __init__(self, cell, target_genes, batch_size, shuffle = True, 
                 i_max=8, w_prom=40_000, w_max=40_000, marks = ASSAYS,
                 pred_res = 100,y_type='log2RPKM',return_pcres=False,
                 return_gene=False,off_centre=None):
        self.eid = cell
        self.target_genes = target_genes # List of ENSGs.
        self.return_gene = return_gene
        meta = pd.read_csv(train_meta)
        #expression in meta is raw expression (RPKM) taken from here:
        #https://egg2.wustl.edu/roadmap/data/byDataType/rna/expression/57epigenomes.RPKM.pc.gz
        #need to convert to log2, add tiny value so not getting log2 of zero
        meta.expression = np.log2(meta.expression+1)#5e-324)
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
        self.off_centre = off_centre #proportion of window upstream => 5/6 would be 2.5k of 3k window
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

        # Binning
        x_binned = []
        for i in range(n_bins):
            b = tf.math.reduce_mean(x[:, i * bin_size: (i + 1) * bin_size],
                                    axis=1,keepdims=True)
            b = tf.math.log(b + 1)
            x_binned.append(b)
        x_binned = tf.concat(x_binned, axis=1)

        # Padding
        left_pad = math.ceil( (max_n_bins - n_bins) / 2 )
        right_pad = math.floor( (max_n_bins - n_bins) / 2 )
        x_binned = tf.concat([
            tf.zeros([tf.shape(x)[0], left_pad]),
            x_binned,
            tf.zeros([tf.shape(x)[0], right_pad]),
        ], axis=1)

        return x_binned, left_pad, n_bins, right_pad
    
    def _get_region_representation(self, chrom, start, end, bin_size, max_n_bins, 
                                   strand='+', window=None,off_centre=None):
        #filter to assays of interest
        marks_ind = [ASSAYS.index(i) for i in self.marks]
        
        x = tf.cast(tf.convert_to_tensor(np.load(str(train_dir / 
                                             f'data/{self.eid}/{chrom}_{start}-{end}.npy'))[marks_ind,:]),
                    tf.float32)
        #x = np.load(str(train_dir / f'data/{self.eid}/{chrom}_{start}-{end}.npy'))[marks_ind,:]
        if window is not None:
            if off_centre is not None:
                x = x[:, 20000 - int((window*off_centre)) // 2:20000 + int((window*(1-off_centre))) // 2]
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
        y_lab = []
        X_x_p_bin_size = []
        X_x_pcre_bin_size = []
        X_pad_mask_p_bin_size = []
        X_pad_mask_pcre_bin_size = []
        X_interaction_freq = []
        X_interaction_mask_bin_size = []
        X_genes = []
        for index in btch_ind:
            target_gene = self.target_genes[index]
            X_genes.append(target_gene)
            assert target_gene in self.ensg2tss, f"Unknown gene - {target_gene}"

            pcres, scores = self.ensg2pcres[target_gene], self.ensg2scores[target_gene]
            n_partners, n_dummies = len(pcres), self.i_max - len(pcres)

            if self.y_type=='label':
                #y.append(tf.constant(self.ensg2label[target_gene],shape=(1,1),dtype=tf.int32))
                y.append(np.asarray([[self.ensg2label[target_gene]]],dtype=np.int32))
            elif self.y_type=='log2RPKM':
                #y.append(tf.constant(self.ensg2exp[target_gene],shape=(1,1),dtype=tf.float32))
                y.append(np.asarray([[self.ensg2exp[target_gene]]],dtype=np.float32))
            elif self.y_type=='both':
                y.append(np.asarray([[self.ensg2exp[target_gene]]],dtype=np.float32))
                y_lab.append(np.asarray([[self.ensg2label[target_gene]]],dtype=np.int32))
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
                                                                                     window=self.w_prom,
                                                                                     off_centre=self.off_centre
                                                                                    )
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
        elif self.return_gene:
            X = ({'x_p_pred_res':tf.concat(X_x_p_bin_size,axis=0),
                  'gene_ids':X_genes})
        else:
            X = ({'x_p_pred_res':tf.concat(X_x_p_bin_size,axis=0)})
            #X = ({'x_p_pred_res':np.concatenate(X_x_p_bin_size,axis=0)})
        y = tf.concat(y,axis=0)

        if self.y_type=='both':
            y = {'log2RPKM':y,'label':tf.concat(y_lab,axis=0)}
        
        return X,y

    def __len__(self):
        return len(self.target_genes)//self.batch_size
    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle == True:
            np.random.shuffle(self.indices)

    
    
