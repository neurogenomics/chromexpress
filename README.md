# epi_to_express
Which epigenetic factors are the best predictors of gene expression? An analysis using ENCODE data.


## Aim
The aim of the project is to try and infer which histone mark or chromatin accessibility assay
does the best job of predicting expression profiles in the same cell type.

* The same neural network architecture will be applied
* A variation of the CNN model from [Evaluating deep learning for predicting epigenomic profiles](https://www.biorxiv.org/content/10.1101/2022.04.29.490059v1.full) 
will be used, trained for 100 epochs using ADAM(default params) and early stopping with patience of 10 epochs.
The initial learning rate was set to 0.001 and decayed by a factor of 0.2 when the loss function did not 
improve after a patience of 3 epochs. Batch size 64.
* The model will predict the continuous RPKM expression values per cell.
* Assay data 1Mbp around the gene of interest will be considered.
* Mulitiple cells will be tested and scored aggregated
* Mulitiple histone marks/chro access assays will be tested and benchmarked.
* Whole chromosomes of genes will be held out for the blind test.
* Combinations won't be considered

## Data
ROADMAP data is used for this project since it is less sparse and more uniformly processed (e.g.
sequencing depth). See `ROADMAP_training_assays_and_cells.ipynb` but in short, the **11** cells/tissues 
being used are:
* E003 : H1 Cell Line
* E114 : A549
* E116 : GM12878
* E117 : HELA
* E118 : HEPG2
* E119 : HMEC
* E120 : HSMM
* E122 : HUVEC
* E123 : K562
* E127 : NHEK
* E128 : NHLF

The **12** assays to be investigated are:
* H3K36me3
* H3K4me1
* H3K4me3
* H3K9me3
* H3K27me3
* H3K27ac
* DNase
* H3K9ac
* H3K4me2
* H2A
* H3K79me2
* H4K20me1

These were used to predict expression. RPKM expression matrix for **protein coding** genes were modelled.

All data was aligned to hg19.

ENCODE blacklist regions were also excluded from training.

## Steps

Use the conda environments (yaml files in ./environments) for the steps.

### 1. Download Data

To download all necessary files (in parallel) run:

```
python bin/download_data_parallel.py
```

with the epi_to_express conda env.

### 2. Convert all bigwigs to 25bp resolution files

The model averages predicitons to 25bp resolution. So the data tracks need to be
adjusted to this resolution. Use the r_bioc environment to do this and run the script:

```
bash bin/avg_bigwig_tracks.sh
```

Note you might need to change permissions on all bigwigs before running this:

```
chmod 777 ./data/*
```

### train the models


## Next Steps
* Test combinations
* Test TF binding as well as hist marks and chrom access 
