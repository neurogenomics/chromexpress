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

Another approach worth mentioning is 
[Prediction of histone post-translational modification patterns based on nascent transcription data](https://www.nature.com/articles/s41588-022-01026-x):
* Investigated relationship hist mods and transcription - predicted histone marks from transcription
* Used an SVR (Support Vector Regression)
* Separate model per histone mark
* 10 histone marks, only in k562 cells
* Used Pro/GRO-seq(transcription) - labels RNA as being transcribed so avoids issues with RNA degredation
* not easy to find histone mark window
* data is interesting though

Note on RPKM: RPKM (reads per kilobase of transcript per million reads mapped) is a gene expression unit 
that measures the expression levels (mRNA abundance) of genes or transcripts. RPKM is a gene length 
normalized expression unit that is used for identifying the differentially expressed genes by comparing 
the RPKM values between different experimental conditions. Generally, the higher the RPKM of a gene, 
the higher the expression of that gene. 

Here we normalised the RPKM per **cell** to be in the 0 - 1 range.

## Aim
The aim of the project is to try and infer which histone mark or chromatin accessibility assay
does the best job of predicting expression profiles in the same cell type.

* The same neural network architecture will be applied
* A variation of the CNN model from [Evaluating deep learning for predicting epigenomic profiles](https://www.biorxiv.org/content/10.1101/2022.04.29.490059v1.full) 
will be used: 
    * trained for up to 100 epochs 
    * ADAM(default params)
    * Early stopping with patience of 12 epochs.
    * The initial learning rate was set to 0.001 
    * Learning rate decay by a factor of 0.2 with a patience of 3 epochs. 
    * Batch size 128.
* The model will predict the normalised continuous RPKM expression values per cell.
* Assay data 10k bp around the gene of interest will be considered (same window size as org study).
* 5 fold cross-validation approach with validation dataset proportion of 30% (split by chromosome)
* Mulitiple cells will be tested and scored aggregated
* Mulitiple histone mark/chromsome accessibility assays will be tested and benchmarked.
* Whole chromosomes of genes will be held out for the blind test.
* Combinations won't be considered yet)

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
* Increase window from 10k
* Test combinations
* Test TF binding as well as hist marks and chrom access


## Second model and data approach

We will use [chromoformer](https://www.nature.com/articles/s41467-022-34152-5) model:
* Predict gene expression from histone marks. Identified distal important regions from pcHi-C data.
* Trained on and predicted in the same cell type (not in previously unseen cell types - just on held out chromosomes)
* Three independent modules were used to produce multi-scale representation of gene expression regulation. Each of the modules is fed with input HM features at different resolutions (100/500/2k bp) to produce an embedding vector reflecting the regulatory state of the core promoter. Each module passed through a transformer.
* Each module was passed local histone mark information at TSS and pCREs (see below point). Again the difference in these modules was the resolution.
* The idea is that using this approach, they looked at gene regulation in a three-layered hierarchy: (1) _cis_-regulation by core promoters, (2) 3D pairwise interaction between a core promoter and a putative _cis_-regulatory regions (pCREs) and (3) a collective regulatory effect imposed by the set of 3D pairwise interactions.
* pcHi-C was used to derive the pCREs for the cell type. Taken from [here](https://www.nature.com/articles/s41588-019-0494-8), the model wasn't trained on this data directly.
* Considered up to 40 kbp distance.
* Trained on ROADMAP data (11 cell types). To benchmark they split genes in 4-fold CV and trained 4 times to get performance. 
* The model was also used to do a small amount of cross-species and cross cell-type predictions (in discussion section).
* They did some validation of what the attention layers are picking up, might be worth exploring.

### Model training specifics
* mean squared error (MSE) loss function
* log2-transformed RPKM values were used for Chromoformer-reg training - using this approach too 
* 10 training epochs - AdamW optimiser
* The initial learning rate was chosen as 3×10−5 and was decreased by 13% after each epoch so that
it can approximately shrink to half of its value after each of the five epochs.
* Batch size was fixed to 64


### Approach

* Use the chromoformer model and another, simple CNN model (from above to compare)
* The receptive window of the CNN model will be less so you might see interesting things 
like promoter specific histone marks might be enough/do better than enhancer ones with this model.
* Train on the 11 cell types from chromoformer and their data but create test dataset 
not just train/val. Also do 4 fold CV.
* Separate model created for each cell type and each fold.
* Try combinations of histone marks and single histone marks
* Inspect effect of altering histone marks at certain positions (maybe)

