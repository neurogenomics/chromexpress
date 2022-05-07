# utils.py

#mulit-channel MSE, split by channel----------------------------------------------------------
from tensorflow.keras.metrics import mean_squared_error

#@tf.function
def multi_mse(y_true,y_pred):
    return mean_squared_error(y_true,y_pred)

#@tf.function
def multi_mse_for_class(index,num_s,num_l):
    def multi_mse_inner(true,pred):
        #get indexs of hist mark
        indexs = list(np.arange(index,num_s*num_l,num_l))
        #get only the desired class
        true = tf.gather(true,indexs,axis=2)
        pred = tf.gather(pred,indexs,axis=2)
        #return dice per class
        return multi_mse(true,pred)
    #have to give each a unique name or metrics call will give out
    multi_mse_inner.__name__='multi_mse_inner'+str(index)
    return multi_mse_inner

#Data loading, training, test split functions ----------------------------------------------------------

import os
import pathlib
from typing import Sequence, Union

import numpy as np
import random
import pyBigWig

import itertools
import tensorflow as tf

from dna_hist_mark_pred.constants import (
    CELLS,
    ALLOWED_CELLS,
    ALLOWED_FEATURES,
    CHROMOSOME_DATA,
    DNA_DATA,
    #DNASE_AVG_PATH,
    BLACKLIST_PATH,
    #DNASE_DATA,
    ATAC_DATA,
    H3K4ME1_DATA,
    H3K27AC_DATA,
    H3K4ME3_DATA,
    H3K9ME3_DATA,
    H3K27ME3_DATA,
    H3K36ME3_DATA,
    ENHANCER_MAP_DATA
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


def get_path(cell: str, feature: str, pred_res: int, 
                 labels_for_all_cells: bool) -> pathlib.Path:
    """Looks up path for given feature of a cell in the data paths"""
    # Get full length cell name from synonym
    #get key from value
    cell=list(CELLS.keys())[list(CELLS.values()).index(cell)]
    # Load feature path
    if feature in ["A", "C", "G", "T"]:
        return DNA_DATA[feature]
    elif feature == "dnase_avg":
        return DNASE_AVG_PATH
    #datasets have been averaged to a certain bp, included in name
    elif feature == "dnase":
        #if we are pred a track for each cell, also pred dnase
        #if(labels_for_all_cells):
        #    return DNASE_DATA[cell+'_'+str(pred_res)]
        #if we pred new cell types quant norm dnase data used to train
        #else:
        #    return DNASE_QN_DATA[cell+'_'+str(pred_res)]
        #Not QN anymore....
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
    elif feature == "enhancer_map": #pred_res is window_size in this case
        return ENHANCER_MAP_DATA[cell+'_enhancer_map_'+str(pred_res)]
    elif feature == "chrom_access_embed":
        #return DNASE_DATA[cell+'_'+str(pred_res)]
        return ATAC_DATA[cell+'_'+str(pred_res)]
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


def random_exclude(exclude: set,chromosome_len: int, window_size: int):
    """Random sampling excluding specific values"""
    randInt = int(
            np.random.randint(low=0, high=chromosome_len - window_size, size=1)
        )
    # if any of the vision of the model in a blacklist region, ignore it
    # so large range - max range of the model, even though nodes won't cover it all
    in_blacklist = any(max(i.start,
                           randInt-window_size) < min(i.stop,randInt+window_size) for i in exclude)
    return random_exclude(exclude, chromosome_len, 
                              window_size) if in_blacklist==True else randInt 

def create_buffer(window_size: int, pred_res: int, pred_prop: float = 0.3):
    """
    Define the size (bp) of the buffer. 
    The buffer is the bp's of the input for which an output isn't predicted as they
    fall on the edge of the input window. These input bp's instead just inform the
    predictions for the other bp's.
    """
    buffer_bp = (window_size*(1-pred_prop))/2
    #buffer length needs to also be divisible by pred res so get closest value:
    buffer_bp = int(round(buffer_bp/pred_res)*pred_res)
    target_bp = window_size-(2*(buffer_bp))
    target_length = int(target_bp/pred_res)
    #return number of base-pairs for the buffer and 
    #the number of positions to predict across (bp's/predicted resolution)
    #and the number of bp's to predict across
    return buffer_bp, target_length, target_bp

def load_y(data,labels_for_all_cells,target_length,labels,cells,
           selected_chromosome,selected_cell,window_start,
           buffer_bp,window_size,pred_res,debug):
    #ensure window size divis by pred res
    window_size_calc = (window_size//pred_res)*pred_res
    #and centre it
    diff = window_size - window_size_calc
    if(diff>1):
        buff_res = (window_size - window_size_calc)//2
    else:
        buff_res = 0
    """Function to load y labels from bigwigs"""                                
    if labels_for_all_cells:
            # Output labels for each of the possible cells -track for each cell-epi mark
            all_y = np.zeros(shape=(target_length, len(labels) * len(cells)))
            if (debug):
                print(f"y labels order:{cells}, {labels}")
            for j, cell in enumerate(cells):
                for i, label in enumerate(labels):
                    #data at pred_res bp lvl already but loaded in at 1bp lvl
                    #need to avg back up!
                    #also data is arcsinh transformed to deal help with different seq depths
                    all_y[:, (j * len(labels)) + i] = np.arcsinh(np.mean(
                        np.nan_to_num(
                            data[cell][label].values(
                                selected_chromosome,
                                window_start+buffer_bp+buff_res,
                                window_start + window_size_calc - buffer_bp+buff_res,
                                numpy=True
                            )
                        ).reshape(-1, pred_res),#averaging at desired pred_res 
                        axis=1))
    else:
        # Output labels only for selected cells
        all_y = np.zeros(shape=(target_length, len(labels)))
        for i, label in enumerate(labels):
            print(selected_cell)
            print(label)
            #data at pred_res bp lvl already but loaded in at 1bp lvl
            #need to avg back up!
            #also data is arcsinh transformed to deal help with different seq depths
            all_y[:, i] = np.arcsinh(np.mean( 
                np.nan_to_num(
                    data[selected_cell][label].values(
                        selected_chromosome,
                        window_start+buffer_bp+buff_res,
                        window_start + window_size_calc - buffer_bp+buff_res,
                        numpy=True
                    )
                ).reshape(-1, pred_res),#averaging at desired pred_res  
                axis=1))
            print("----")
    return all_y

def initiate_bigwigs(
    cells: Sequence[str],
    cell_probs: Sequence[float],
    chromosomes: Sequence[str],
    chromosome_probs: Sequence[float],
    features: Sequence[str],
    labels: Sequence[str],
    pred_res: int = 25,
    labels_for_all_cells=False
):
    """
    Initiate the connection to the bigwigs
    """

    # Input verification
    if not all(np.isin(cells, ALLOWED_CELLS)):
        raise ValueError(
            "Cell types contain values which are not allowed. "
            f"Allowed cell types are {ALLOWED_CELLS}."
        )
    if not all(np.isin(features, ALLOWED_FEATURES)):
        raise ValueError(
            "Features contain values which are not allowed. "
            f"Allowed features are {ALLOWED_FEATURES}."
        )
    if not all(np.isin(labels, ALLOWED_FEATURES)):
        raise ValueError(
            "Labels contain values which are not allowed. "
            f"Allowed labels are {ALLOWED_FEATURES}."
        )
    # make sure dnase second last track and dnase_avg last track
    # location necessary for operations
    if all(np.isin(['dnase','dnase_avg'],features)):
        features.append(features.pop(features.index('dnase')))
        features.append(features.pop(features.index('dnase_avg')))
    # make sure dnase_embed last track
    # location necessary for operations
    if (np.isin(['chrom_access_embed'],features)):
        features.append(features.pop(features.index('chrom_access_embed')))      

    assert len(cells) == len(cell_probs), "Must provide probabilities for all cells"
    assert len(chromosomes) == len(
        chromosome_probs
    ), "Must provide probabilities for all chromosomes"
    assert len(features) > 0, "Must provide at least one feature"
    assert len(labels) > 0, "Must provide at least one label"
    
    # Load file handles for all cells and features
    data = {
        cell: {
            feature: load_bigwig(get_path(cell, feature, pred_res, labels_for_all_cells))
            for feature in features + labels
        }
        for cell in cells
    }
    return data


def generate_data(
    cells: Sequence[str],
    cell_probs: Sequence[float],
    chromosomes: Sequence[str],
    chromosome_probs: Sequence[float],
    features: Sequence[str],
    labels: Sequence[str],
    data: dict,#pybigwig object 
    num_samps: int = 1,
    window_size: int = 1000000, #1MB in either direction receptive field model
    pred_res: int = 25,
    debug: bool = False,
    as_tensor: bool = True,
    labels_for_all_cells=False,
    reverse_complement=False,
    rand_seq_shift=False,
    pred_prop=0.3,
    rand_pos=True,
    chro: int =0,
    pos: int =0,
    cell: str='None',
    n_genomic_positions=1125000,#avocado-1126469
    training: bool = True
):
    """
    Generate the tensor graph with connections for the training data.
    The positions is randomly sampled.
    """
    # Input verification
    #only check if training - want to apply to new cell types
    if training:
        if not all(np.isin(cells, ALLOWED_CELLS)):
            raise ValueError(
                "Cell types contain values which are not allowed. "
                f"Allowed cell types are {ALLOWED_CELLS}."
            )
    if not all(np.isin(features, ALLOWED_FEATURES)):
        raise ValueError(
            "Features contain values which are not allowed. "
            f"Allowed features are {ALLOWED_FEATURES}."
        )
    if not all(np.isin(labels, ALLOWED_FEATURES)):
        raise ValueError(
            "Lables contain values which are not allowed. "
            f"Allowed labels are {ALLOWED_FEATURES}."
        )
    # make sure dnase second last track and dnase_avg last track
    # location necessary for operations
    if all(np.isin(['dnase','dnase_avg'],features)):
        features.append(features.pop(features.index('dnase')))
        features.append(features.pop(features.index('dnase_avg')))
    # make sure dnase_embed last track
    # location necessary for operations
    if (np.isin(['chrom_access_embed'],features)):
        features.append(features.pop(features.index('chrom_access_embed')))       
        
    if training:
        assert len(cells) == len(cell_probs), "Must provide probabilities for all cells"
    assert len(chromosomes) == len(
        chromosome_probs
    ), "Must provide probabilities for all chromosomes"
    assert len(features) > 0, "Must provide at least one feature"
    assert len(labels) > 0, "Must provide at least one label"
    #assert window_size%pred_res==0, "Window size must be divisible by prediciton resolution"
    
    # At each generator iteration:
    while True:
        samps_X =[]
        samps_embed_250 =[]
        samps_embed_5k =[]
        samps_y =[]
        for samp in range(num_samps):
            #if adding random perm, load once slightly large then subset
            if rand_seq_shift:
                #up to 3 nucleotides either side
                #allowed_values = list(range(-3, 3+1)) #up to not incl second number
                #allowed_values.remove(0)
                #only need positive values either way since this is dist from act to rand shift
                rand_shift_amt = random.choice(range(1, 3+1))#up to not incl second number
            else:
                rand_shift_amt = 0

            if debug:
                print(f"Random shift: {rand_shift_amt} nucleotides")    

            #randomly select the chromosome and position
            if rand_pos:
                # Select cell
                selected_cell = np.random.choice(cells, replace=True, p=cell_probs)
                if debug:
                    print(f"Selected cell: {selected_cell}")

                # Select chromosome
                selected_chromosome = np.random.choice(
                    chromosomes, replace=True, p=chromosome_probs
                )
                chromosome_len = CHROMOSOME_DATA[selected_chromosome]
                if debug:
                    print(f"Selected chromosome: {selected_chromosome}")

                # Exclude blacklist regions
                blacklist_regions = load_bigwig(BLACKLIST_PATH)
                blacklist_chr = blacklist_regions.entries(selected_chromosome,0,chromosome_len)
                # Need to get all blacklist regions in the chormosome of interest
                blacklist_chr_positions = [range(i[0],i[1]) for i in blacklist_chr]
                # Select window to read - exclude blacklist positions
                # if embedding length bigger than window size, that's the window for sampling
                if(not(labels_for_all_cells) and window_size<n_genomic_positions):
                    window_start = random_exclude(blacklist_chr_positions, chromosome_len, 
                                                  n_genomic_positions+rand_shift_amt)
                else:    
                    window_start = random_exclude(blacklist_chr_positions, chromosome_len, 
                                                  window_size+rand_shift_amt)
            else:
                selected_chromosome=chro
                window_start=pos
                if cell=='None' and labels_for_all_cells:
                    print("cell == None")
                    #pick any, not using
                    selected_cell =cells[0]
                else:
                    #for predicting in 1 new cell type
                    selected_cell=cells

            if debug:
                print(f"Selected window: {window_start}")

            # Load the data
            # X data (features)
            if labels_for_all_cells:
                #only passes DNA as input
                all_X = np.zeros(shape=(window_size + rand_shift_amt, len(features)))
                for i, feature in enumerate(features):
                    #load full window including rand_shift_amt so don't have to load twice
                    all_X[:, i] = data[selected_cell][feature].values(
                        selected_chromosome, window_start, 
                        window_start + window_size + rand_shift_amt,
                        numpy=True
                    )

            else:
                #passes dna and data to represent cell types
                #trying dnase signal minus avg dnase, so no separate track for the avg
                all_X = np.zeros(shape=(window_size + rand_shift_amt, len(features)-1))
                for i, feature in enumerate(features):
                    #load full window including rand_shift_amt so don't have to load twice
                    if(feature!='dnase_avg' and feature!='chrom_access_embed'):
                        #also data is arcsinh transformed to deal help with different seq depths
                        all_X[:, i] = data[selected_cell][feature].values(
                            selected_chromosome, window_start, 
                            window_start + window_size + rand_shift_amt,
                            numpy=True
                        )
                    #if using cell type embedding
                    elif(feature=='chrom_access_embed'):
                        # Make 3 tracks for diff resolutions
                        # also data is arcsinh transformed to deal help with different seq depths
                        #load data then augment
                        chrom_access_raw = data[selected_cell][feature].values(
                            selected_chromosome, window_start, 
                            window_start + n_genomic_positions,numpy=True)
                        #250bp
                        cell_embed_250 = np.arcsinh(np.float32(np.mean(#enformer works with float32 not 64
                        np.nan_to_num(chrom_access_raw).reshape(-1, 250),#bp res
                        axis=1)))
                        #5kbp
                        cell_embed_5k = np.arcsinh(np.float32(np.mean(#enformer works with float32 not 64
                        np.nan_to_num(chrom_access_raw).reshape(-1, 5000),#bp res
                        axis=1)))

            #if random shifting of pos happening make sep inputs
            #create list of the different datasets
            X = []
            #create list for embedding if using
            if (np.isin(['chrom_access_embed'],features)):        
                #embed_X_25 = []
                embed_X_250 = [] 
                embed_X_5k = []
            if rand_seq_shift:
                #Now subset this full load as necessary
                #append org
                X.append(all_X[:-rand_shift_amt])
                #append random shift
                X.append(all_X[rand_shift_amt:])
                #embedding - do it. twice since the same for rand seq diff
                if (np.isin(['chrom_access_embed'],features)):
                    embed_X_250.append(cell_embed_250)
                    embed_X_250.append(cell_embed_250)
                    embed_X_5k.append(cell_embed_5k)
                    embed_X_5k.append(cell_embed_5k)
            else:
                #only passing org input without rand shift input
                X.append(all_X)
                #embedding - only pass once
                if (np.isin(['chrom_access_embed'],features)):
                    embed_X_250.append(cell_embed_250)
                    embed_X_5k.append(cell_embed_5k)

            # y data (labels)
            #work out buffer where preds won't be made
            buffer_bp,target_length,target_bp = create_buffer(window_size=window_size,
                                                              pred_res=pred_res,
                                                              pred_prop= pred_prop)

            all_y = load_y(data,labels_for_all_cells,target_length,labels,cells,
                           selected_chromosome,selected_cell,window_start,
                           buffer_bp,window_size,pred_res,debug)

            #if random shifting of pos happening make sep inputs
            #can't use same trick as with x as averaging with buffer so just 
            #load random shift separate and match with corresponding entry
            y = []
            if rand_seq_shift:
                all_y2 = load_y(data,labels_for_all_cells,target_length,labels,cells,
                                selected_chromosome,selected_cell,
                                window_start+rand_shift_amt,#add random shift 
                                buffer_bp,window_size,pred_res,debug)
                #append org
                y.append(all_y)
                #append random shift
                y.append(all_y2)
            else:
                y.append(all_y)

            # if training on reverse complement, calculate and add it in
            # we will get the reverse complement of the actual and randomly shifted input 
            if reverse_complement:
                #in case random perm added:
                org_len_X = len(X)
                for x_i in range(org_len_X):
                    X.append(X[x_i][::-1])    
                #same for y
                org_len_y = len(y)
                for y_i in range(org_len_y):
                    y.append(y[y_i][::-1])
                #same for embedding 
                if (np.isin(['chrom_access_embed'],features)):
                    for x_i in range(org_len_X):
                        embed_X_250.append(embed_X_250[x_i][::-1])
                        embed_X_5k.append(embed_X_5k[x_i][::-1])    
        
            # return features and labels as tensors
            if (not(np.isin(['chrom_access_embed'],features))):
                if as_tensor:
                    samps_X.append([np.float32(tf.convert_to_tensor(x_i.copy())) for x_i in X])
                    samps_y.append([np.float32(tf.convert_to_tensor(y_i.copy())) for y_i in y])
                else:
                    samps_X.append(X)
                    samps_y.append(y)

            else: #return embedding too
                if as_tensor:
                    #enformer works with float32 not 64
                    samps_X.append(tf.stack([np.float32(tf.convert_to_tensor(x_i.copy())) for x_i in X]))
                    samps_embed_250.append(tf.stack([tf.convert_to_tensor(x_i.copy()) for x_i in embed_X_250]))
                    samps_embed_5k.append(tf.stack([tf.convert_to_tensor(x_i.copy()) for x_i in embed_X_5k]))
                    samps_y.append(tf.stack([np.float32(tf.convert_to_tensor(y_i.copy())) for y_i in y]))
                else:
                    samps_X.append({"dna": X,"chrom_access_250": embed_X_250,"chrom_access_5k": embed_X_5k})
                    samps_y.append(y)
        
        # finally return with multiple samples
        if (not(np.isin(['chrom_access_embed'],features))):
            if as_tensor:
                yield(tf.concat(samps_X,axis=0),tf.concat(samps_y,axis=0))
            else:
                yield(samps_X,samps_y)
        else:
            if as_tensor:
                    yield({"dna":tf.concat(samps_X,axis=0),
                           "chrom_access_250":tf.concat(samps_embed_250,axis=0),
                           "chrom_access_5k":tf.concat(samps_embed_5k,axis=0)},
                          tf.concat(samps_y,axis=0))
            else:
                    yield(samps_X,samps_y)