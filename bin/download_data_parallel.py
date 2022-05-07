"""
Selection of functions to download data from sources like Roadmap.
"""

from datetime import datetime
import requests
import pandas as pd
import os
import errno
from typing import List
#from pyarrow import csv
from functools import partial
from multiprocessing import Pool, cpu_count
from epi_to_express.constants import ASSAYS, METADATA_PATH, DATA_PATH

def create_path(filename: str) -> None:
    """
    This function creates the path to the
    specified folder.

    Parameters
    ----------
    filename : str
        the path to be created
    """
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise


def download_file(url_name: tuple,folder: str, extension : bool = True) -> str:
    """
    This function downloads a file from the specified url
    and writes it to a folder.

    Parameters
    ----------
    url_name: tuple
        the name for and url from which to download the file.
    folder: str
        the folder in which to write the file
    """
    name = url_name[0]
    url = url_name[1]
    print('Downloading: {name}'.format(name=name))
    #don't add bigwig extension for QTL files or if specified
    if extension == False:
        local_filename = str(folder) + '/' + name
    else:
        local_filename = str(folder) + '/' + name + ".bigWig"
    try:
        create_path(local_filename)
        with requests.get(url, stream=True) as r:
            #check if link didn't work
            if r.status_code == 404:
                print("Something went wrong when attempting to download from {}".format(name))
            else:
                r.raise_for_status()
                with open(local_filename, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)         
    except:
        print("Something went wrong when attempting to download from {}".format(name))
    return local_filename


def download_bigwigs(exp_type: str) -> None:
    """
    Download the turing files to their folders
    in the data_download folder. File urls are read from
    the files in metadata

    Parameters
    ----------
    exp_type: string
        the test type, h3k4me1, h3k27ac, dnase, atac or dna
    """
    #don't bother going in parallel for expression data
    if(exp_type=='expression'):
        exp_link = ('57epigenomes.RPKM.pc.gz',
                    'https://egg2.wustl.edu/roadmap/data/byDataType/rna/expression/57epigenomes.RPKM.pc.gz')
        folder_name = DATA_PATH / '{exp_type}'.format(exp_type=exp_type)
        download_file(exp_link,folder_name,extension=False)
    #bigwigs
    else:
        df = pd.read_csv(METADATA_PATH / '{exp_type}.csv'.format(exp_type=exp_type),
                sep=',')
        name_urls = df.values.tolist()
        folder_name = DATA_PATH / '{exp_type}'.format(exp_type=exp_type)
        #create parallel use all cpus
        pool = Pool(cpu_count())
        #same folder_name for all
        download_func = partial(download_file, folder = folder_name)
        results = pool.map(download_func, name_urls)
        #close parallel
        pool.close()
        pool.join()


def print_names(file_list: List[str]) -> str:
    return '\n'.join(['Saved file at: {}'.format(file) for file in file_list])


def download_blacklist_regions() -> None:
    """
    Downloads ENCODE's blacklist regions in hg19
    """
    folder_name = DATA_PATH / 'model_ref'
    blck_list = ("encode_blacklist.bigBed",
            "https://www.encodeproject.org/files/ENCFF000KJP/@@download/ENCFF000KJP.bigBed")
    download_file(blck_list,folder_name,extension=False)

if __name__ == '__main__':
    print(datetime.now())
    print("There are {} CPUs on this machine ".format(cpu_count()))
    for exp_type in ['model_ref','expression']+ASSAYS:
        print("Downloading {exp_type}".format(exp_type=exp_type))
        download_bigwigs(exp_type)
    print("download training data completed")
    print(datetime.now())
    #download encode blacklist regions
    download_blacklist_regions()
    print("All downloads complete")
    print(datetime.now())
