# load.py
import pandas as pd

def load_data(path):
    # Load a gzipped TSV file and return it as a pandas DataFrame
    return pd.read_csv(path, compression='gzip', header=0, sep='\t')

# Function to load specific data files
def load_gene_expression():
    return load_data('archs4_gene_expression_norm_transposed.tsv.gz')

def load_small_gene_expression():
    return load_data('gtex_gene_expression_norm_transposed.tsv.gz')

def load_isoform_expression():
    return load_data('gtex_isoform_expression_norm_transposed.tsv.gz')

def load_gene_isoform_annotation():
    return load_data('gtex_gene_isoform_annoation.tsv.gz')

def load_annotations():
    return load_data('gtex_annot.tsv.gz')
