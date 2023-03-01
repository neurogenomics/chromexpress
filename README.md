# epi_to_express
Which epigenetic factors are the best predictors of gene expression? An analysis using ROADMAP data.

## Background
This question was studied quite some time ago: [
Histone modification levels are predictive for gene expression](https://www.pnas.org/doi/10.1073/pnas.0909344107#:~:text=Using%20the%20trained%20model%20parameters,0.82%2C%20respectively)
However:
* Only considered 2 T-cells
* Used a linear model
* Only considered promoter histone mark information (i.e. very short range)
* Used linear regression for this

Another approach [Differential contribution to gene expression prediction of histone modifications at enhancers or promoters](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1009368). Found predictive marks at enhancers are also (as well as at promoters) predictive of expression. Defined states in the genome of regions that are active/repressive based own histone marks present and split into enhancers (by known enhancer mark) and promoters (by overlap with TSS). Then linked enhancers to genes using Hi-C data. Created two models, one built on the enhancer positions and one on the promoters and then looked at variable importance of the marks in these. However:

* Only looked at mouse embryonic stem cells (ESC)
* Only 4 histone marks: H3K4me3, H3K27me3, H3K27ac, and H3K4me1 to define enhancer and promoter regions
* Used 10 hist marks to predict expression.
* Used a linear model local regression (LOESS) which weights contribution by distance from the TSS
* Included promoters and enhancers
* Found H3K27me3 — a histone modification associated with transcriptional gene repression — was the prevalent mark in both enhancer and promoter model. Found H3K27ac importance in the enhancer model was masked by H3K27me3. Removing H3K27me3 from enhancer model only slightly dropped performance and then H3K27ac was the most important mark.
* It seems circular that they pick the training regions using 4 marks and then use these regions as inputs to the model measuring 10 marks

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

Here we normalised the log2RPKM (specifcally log2(RPKM+1) to avoid numerical errors)

## Aim
The aim of the project is to try and infer which histone mark or chromatin accessibility assay
does the best job of predicting expression profiles in the same cell type.

* The same neural network architectures will be applied to all marks - one large and one narrow input window model.
* A variation of the CNN model, similar to that proposed in [DeepChrome: deep-learning for predicting gene expression from histone modifications](https://academic.oup.com/bioinformatics/article/32/17/i639/2450757) was developed for the short range model: 
    * Input window 6k bp around TSS
    * ADAM(default params) optimiser
* [chromoformer](https://www.nature.com/articles/s41467-022-34152-5) will be used as the large window input model:
    * 40k window around TSS - uses attention
    * AdamW optimiser
it can approximately shrink to half of its value after each of the five epochs.
* Both models will use:
    * mean squared error (MSE) loss function.
    * The models will predict the continuous log2RPKM expression values per cell.
    * 4 fold cross-validation approach with validation dataset proportion of 12.5% and test of 12.5%. Thus 75% for training on each fold.
    * Trained for up to 100 epochs
    * Early stopping with patience of 12 epochs.
    * The initial learning rate was set to 0.001
    * Learning rate decay by a factor of 0.2 with a patience of 3 epochs.
    * Batch size 64.
    * Mulitiple cells will be tested and scored aggregated
    * Mulitiple histone mark/chromsome accessibility assays will be tested and benchmarked.
    * Combinations won't be considered **yet**.

## Data
ROADMAP data is used for this project since it is less sparse and more uniformly processed (e.g.
sequencing depth). See `ROADMAP_training_assays_and_cells.ipynb` but in short, the **11** cells/tissues 
being used are:
* E003 : H1 cells
* E004 : H1 BMP4 derived mesendoderm
* E005 : H1 BMP4 derived trophoblast
* E006 : H1 derived mesenchymal stem cells
* E007 : H1 derived neuronal progenitor cultured cells
* E016 : HUES64 cells
* E066 : Liver
* E087 : Pancreatic islets
* E114 : A549 EtOH 0.02pct lung carcinoma
* E116 : GM12878 lymphoblastoid
* E118 : HepG2 hepatocellular carcinoma

The **7** histone marks to be investigated are:
* H3K36me3
* H3K4me1
* H3K4me3
* H3K9me3
* H3K27me3
* H3K27ac
* H3K9ac

These were used to predict expression. log2RPKM expression matrix for **protein coding** genes were modelled. 

Performance was measured by Pearson R.

All data was aligned to hg19.

## Approach

* Use the chromoformer model and another, simple CNN model (from above to compare)
* The receptive window of the CNN model will be less so you might see interesting things 
like promoter specific histone marks might be enough/do better than enhancer ones with this model.
* Train on the 11 cell types from chromoformer and their data but create test dataset
not just train/val. Also do 4 fold CV.
* Separate model created for each cell type and each fold.
* Try combinations of histone marks and single histone marks
* Inspect effect of altering histone marks at certain positions (maybe)

## Steps

Use the conda environments (yaml files in ./environments) for the steps.

### 1. Download Data

Follow steps in chromoformer repo embedded in this repo to download all data.


### 2. train the models
TO DO Add scripts to run....

### 3. Measure performance
TO DO add jupyternotebooks to inspect

## Background on chromoformer original study
[chromoformer](https://www.nature.com/articles/s41467-022-34152-5) model:
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
* Model training specifics
    * mean squared error (MSE) loss function
    * log2-transformed RPKM values were used for Chromoformer-reg training - using this approach too 
    * 10 training epochs - AdamW optimiser
    * The initial learning rate was chosen as 3×10−5 and was decreased by 13% after each epoch so that
it can approximately shrink to half of its value after each of the five epochs.
