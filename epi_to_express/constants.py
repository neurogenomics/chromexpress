"""All project wide constants are saved in this module."""
import os
import pathlib
import numpy as np

# Note: constants should be UPPER_CASE
constants_path = pathlib.Path(os.path.realpath(__file__))
SRC_PATH = pathlib.Path(os.path.dirname(constants_path))
PROJECT_PATH = pathlib.Path(os.path.dirname(SRC_PATH))

# Data paths
METADATA_PATH = PROJECT_PATH / "metadata"

# Load metadata
#  for cell type samples
# dictionary of cell id and descriptive name
CELLS = {'E003' : 'H1 cells',
         'E004' : 'H1 BMP4 derived mesendoderm',
         'E005' : 'H1 BMP4 derived trophoblast',
         'E006' : 'H1 derived mesenchymal stem cells',
         'E007' : 'H1 derived neuronal progenitor cultured cells',
         'E016' : 'HUES64 cells',
         'E066' : 'Liver',
         'E087' : 'Pancreatic islets',
         'E114' : 'A549 EtOH 0.02pct lung carcinoma',
         'E116' : 'GM12878 lymphoblastoid',
         'E118' : 'HepG2 hepatocellular carcinoma',
         }

SAMPLES=list(CELLS.values())
SAMPLE_IDS=list(CELLS.keys())

#hg19 chrom lengths
CHROM_LEN =np.array([249250621, 243199373, 198022430, 191154276, 180915260, 171115067,
            159138663, 146364022, 141213431, 135534747, 135006516, 133851895,
            115169878, 107349540, 102531392,  90354753,  81195210,  78077248,
            59128983,  63025520,  48129895,  51304566])
#Model will predict on chromsomes 1-22 (not sex chromosomes)
CHROMOSOMES =np.array(['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8',
              'chr9', 'chr10', 'chr11', 'chr12', 'chr13', 'chr14', 'chr15',
              'chr16', 'chr17', 'chr18', 'chr19', 'chr20', 'chr21', 'chr22'])

#Assays - epi
#order same as chromoformer output
ASSAYS = ['h3k4me1', 'h3k4me3', 'h3k9me3', 'h3k27me3', 'h3k36me3', 'h3k27ac', 'h3k9ac']
ALLOWED_FEATURES = ASSAYS
ALLOWED_CELLS = SAMPLES
CHROMOSOME_DATA = {
    chromosome: length for chromosome, length in zip(CHROMOSOMES, CHROM_LEN)
}
