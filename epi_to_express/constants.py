"""All project wide constants are saved in this module."""
import os
import pathlib
import numpy as np

# Note: constants should be UPPER_CASE
constants_path = pathlib.Path(os.path.realpath(__file__))
SRC_PATH = pathlib.Path(os.path.dirname(constants_path))
PROJECT_PATH = pathlib.Path(os.path.dirname(SRC_PATH))

# Data paths
DATA_PATH = PROJECT_PATH / "data"
EXP_PTH = DATA_PATH / "expression"
METADATA_PATH = PROJECT_PATH / "metadata"
MODEL_REFERENCE_PATH = DATA_PATH / "model_ref/"

# Load metadata
#  for cell type samples
# dictionary of cell id and descriptive name
CELLS = {'E003' : 'H1 Cell Line',
         'E114' : 'A549',
         'E116' : 'GM12878',
         'E117' : 'HELA',
         'E118' : 'HEPG2',
         'E119' : 'HMEC',
         'E120' : 'HSMM',
         'E122' : 'HUVEC',
         'E123' : 'K562',
         'E127' : 'NHEK',
         #'E128' : 'NHLF' #all values are nan in exp mat so don't use
         }

SAMPLES=list(CELLS.values())
SAMPLE_NAMES=list(CELLS.keys())

#hg19 chrom lengths
CHROM_LEN =np.array([249250621, 243199373, 198022430, 191154276, 180915260, 171115067,
            159138663, 146364022, 141213431, 135534747, 135006516, 133851895,
            115169878, 107349540, 102531392,  90354753,  81195210,  78077248,
            59128983,  63025520,  48129895,  51304566])
#Model will predict on chromsomes 1-22 (not sex chromosomes)
CHROMOSOMES =np.array(['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8',
              'chr9', 'chr10', 'chr11', 'chr12', 'chr13', 'chr14', 'chr15',
              'chr16', 'chr17', 'chr18', 'chr19', 'chr20', 'chr21', 'chr22'])

#Assays - epi & chrom access
ASSAYS = ['h3k36me3','h3k4me1','h3k4me3','h3k9me3','h3k27me3','h3k27ac','dnase',
          'h3k9ac','h3k4me2','h2a','h3k79me2','h4k20me1']

# Extract all possible data file paths
H3K36ME3_DATA = {path.stem: path for path in DATA_PATH.glob("h3k36me3/*")}
H3K4ME1_DATA = {path.stem: path for path in DATA_PATH.glob("h3k4me1/*")}
H3K4ME3_DATA = {path.stem: path for path in DATA_PATH.glob("h3k4me3/*")}
H3K9ME3_DATA = {path.stem: path for path in DATA_PATH.glob("h3k9me3/*")}
H3K27ME3_DATA = {path.stem: path for path in DATA_PATH.glob("h3k27me3/*")}
H3K27AC_DATA = {path.stem: path for path in DATA_PATH.glob("h3k27ac/*")}
DNASE_DATA = {path.stem: path for path in DATA_PATH.glob("dnase/*")}
H3K9AC_DATA = {path.stem: path for path in DATA_PATH.glob("h3k9ac/*")}
H3K4ME2_DATA = {path.stem: path for path in DATA_PATH.glob("h3k4me2/*")}
H2A_DATA = {path.stem: path for path in DATA_PATH.glob("h2a/*")}
H3K79ME2_DATA = {path.stem: path for path in DATA_PATH.glob("h3k79me2/*")}
H4K20ME1_DATA = {path.stem: path for path in DATA_PATH.glob("h4k20me1/*")}


ALLOWED_FEATURES = ["expression", "dnase","h3k27ac", "h3k4me1","h3k9ac","h3k4me2",
                    "h3k4me3","h3k9me3","h3k27me3","h3k36me3","h2a","h3k79me2","h4k20me1"]
ALLOWED_CELLS = SAMPLES
CHROMOSOME_DATA = {
    chromosome: length for chromosome, length in zip(CHROMOSOMES, CHROM_LEN)
}
