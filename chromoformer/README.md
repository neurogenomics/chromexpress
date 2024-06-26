# Cloned from [Chromoformer repository](https://github.com/dohlee/chromoformer) 

# Chromoformer

[![DOI](https://zenodo.org/badge/432363545.svg)](https://zenodo.org/badge/latestdoi/432363545)

This repository provides the official code implementations for Chromoformer.

We also provide our pipelines for preprocessing input data and training Chromoformer model to help researchers reproduce the results and extend the study with their own data.
The repository includes two directories: `preprocessing` and `chromoformer`.


## Download & process training data 
Refer to the directory named [`preprocessing`](preprocessing) to explore how we preprocessed the ChIP-seq signals and gene expression data for 11 cell lines from [Roadmap Epigenomics Project](http://www.roadmapepigenomics.org). We provide the whole preprocessing workflow from raw ChIP-seq reads to processed read depth signals for training as a one-shot `snakemake` pipeline. One can easily extend the pipeline for other cell types or other histone marks by slightly tweaking the parameters.

[`chromoformer`](chromoformer) directory provides the PyTorch implementation of the Chromoformer model.

## Model description

The full model architecture is shown below. It consists of three independent modules each of which accepts input features at different resolutions and in turn produces an embedding vector of the regulatory state at the core promoter. The resulting three regulatory embeddings are concatenated to form a multi-scale regulatory embedding which is subsequently fed into fully-connected layers to predict the expression level of a gene.

![model1](img/model1.png)

There are three transformer-based submodules: Embedding, Pairwise Interaction and Regulation transformer. To fully utilize the transformer architecture to model the complex dynamics of *cis*-regulations involving multiple layers, we conceptually decomposed the gene regulation into a three-layered hierarchy: (1) *cis*-regulation by core promoters, (2) 3D pairwise interaction between a core promoter and a putative *cis*-regulatory regions (pCREs) and (3) a collective regulatory effect imposed by the set of 3D pairwise interactions.

![model2](img/model2.png)


## See also

We acknowledge the following deep learning model architectures since they were used as benchmark models for the evaluation of Chromoformer.
We highly recommend going through those works thoroughly to understand the concepts of deep learning-based gene expression prediction using histone modifications.

- `DeepChrome`: https://github.com/QData/DeepChrome
- `AttentiveChrome`: https://github.com/QData/AttentiveChrome
- `HM-CRNN`: https://github.com/pptnz/deeply-learning-regulatory-latent-space
- `GC-MERGE`: https://github.com/rsinghlab/GC-MERGE
- `DeepDiff`: https://github.com/QData/DeepDiffChrome

## Citation

Lee, D., Yang, J., & Kim, S. Learning the histone codes with large genomic windows and three-dimensional chromatin interactions using transformer. Nature Communications (2022)

```
@article{lee2022chromoformer,
  author={Lee, Dohoon and Yang, Jeewon and Kim, Sun},
  title={Learning the histone codes with large genomic windows and three-dimensional chromatin interactions using transformer},
  journal={Nature Communications},
  doi={10.1038/s41467-022-34152-5},
  year={2022},
  publisher={Nature Publishing Group}
}
```
