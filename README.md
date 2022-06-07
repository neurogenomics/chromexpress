# epi_to_express
Which epigenetic factors are the best predictors of gene expression? An analysis using ROADMAP data.

## Background
This question was studied quite some time ago: [
Histone modification levels are predictive for gene expression](https://www.pnas.org/doi/10.1073/pnas.0909344107#:~:text=Using%20the%20trained%20model%20parameters,0.82%2C%20respectively)
However:
* Only considered 2 T-cells
* Used a linear model
* Only considered promoter histone mark information (i.e. very short range)

Our work here expands on this by considering more cell types to see if there are universal histone marks
which are predictive, considers far wider sequences of histone mark information which should encourage
enhancer and other long range marks and finally, uses neural networks which is more in line with current
approaches so would be more applicable.

Note on RPKM: RPKM (reads per kilobase of transcript per million reads mapped) is a gene expression unit 
that measures the expression levels (mRNA abundance) of genes or transcripts. RPKM is a gene length 
normalized expression unit that is used for identifying the differentially expressed genes by comparing 
the RPKM values between different experimental conditions. Generally, the higher the RPKM of a gene, 
the higher the expression of that gene. 

Here we normalised the RPKM per cell to have mean 0 std 1.

## Aim
The aim of the project is to try and infer which histone mark or chromatin accessibility assay
does the best job of predicting expression profiles in the same cell type.

* The same neural network architecture will be applied
* A variation of the CNN model from [Evaluating deep learning for predicting epigenomic profiles](https://www.biorxiv.org/content/10.1101/2022.04.29.490059v1.full) 
will be used: 
    * trained for 100 epochs 
    * ADAM(default params)
    * Early stopping with patience of 12 epochs.
    * The initial learning rate was set to 0.001 
    * Learning rate decay by a factor of 0.2 with a patience of 3 epochs. 
    * Batch size 128.
* The model will predict the continuous RPKM expression values per cell.
* Assay data 100k bp around the gene of interest will be considered.
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

## Steps

Use the conda environments (yaml files in ./environments) for the steps.

### 1. Download Data

To download all necessary files (in parallel) run:

```
python bin/download_data_parallel.py
```

with the epi_to_express conda env.

**Also** run the below command with the same env to get the reference genome data for grch37:

```
pyensembl install --release 75 --species homo_sapiens
```

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
