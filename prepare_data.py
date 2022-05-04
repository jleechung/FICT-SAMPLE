#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 07:17:18 2020

@author: haotian teng
"""
import os
import argparse
import numpy as np
import pandas as pd

from fict.fict_input import RealDataLoader
from fict.utils import data_op as dop


def get_args():
    parser = argparse.ArgumentParser(
        prog = 'FICT prepare',
        description = 'Prepare data for running FICT from a (cell X feature) .csv file'
    )
    parser.add_argument('-i', '--input', required = True, help = 'Path to input data', type = str)
    parser.add_argument('-o', '--output', required = True, help = 'Prefix for output files', type = str)
    parser.add_argument('--gene_cols', required = True, help = 'Start and end columns for genes (comma-separated values)', type = str)
    parser.add_argument('--coord_cols', required = True, help = 'Start and end columns for coordinates (comma-separated values)', type = str)
    parser.add_argument('--header', default = 0, help = 'Row number of header (0-index)', type = int)
    parser.add_argument('--n_type', default = 10, help = 'Number of cell types', type = int)
    parser.add_argument('--thres_dist', default = 100, help = 'Threshold distance', type = float)
    parser.add_argument('--cell_anno', default = None, help = 'Column name for cell annotation', type = str)
    parser.add_argument('--fov', default = None, help = 'Column name for FOV', type = str)
    args = parser.parse_args()
    return (args)

def process_range(input):
    '''Convert csv to range
    Args:
        input: (str) comma-separated values
    '''
    lst = [int(i) for i in input.split(',')]
    lst.sort()
    return np.arange(lst[0],lst[1])

def check_key(data, colname):
    '''Fetch data from pandas dataframe
    Args:
        data: pandas df
        colname: (str) column to fetch
    '''
    if colname is not None:
        if colname not in data.columns:
            raise KeyError('%s not found in data columns' % colname)
        return data[colname]
    return None

def main(args):

    # Hyper parameter setting
    print("Setting hyper parameters")
    ## Parse args
    n_c = args.n_typee
    threshold_distance = args.thres_dist
    gene_col = process_range(args.gene_cols)
    coord_col = process_range(args.coord_cols)
    header = args.header
    data_f = args.input

    if not os.path.exists(data_f):
        raise FileNotFoundError("%s not found" % args.input)

    os.makedirs(args.output,exist_ok=True)

    # Data preprocessing
    print("Reading data from %s" % data_f)
    data = pd.read_csv(data_f, header = header)

    ## Read gene expression, remove nan, count normalize
    gene_expression = data.iloc[:, gene_col]
    nan_cols = np.unique(np.where(np.isnan(gene_expression))[1])
    for nan_col in nan_cols:
        gene_col = np.delete(gene_col, nan_col)
    gene_name = data.columns[gene_col]
    gene_expression = np.asarray(data.iloc[:, gene_col])
    gene_expression = gene_expression / np.sum(gene_expression, axis = 1, keepdims = True)

    ## Read coordinates
    coordinates = np.asarray(data.iloc[:, coord_col])

    ## Read annotations and FOV (if applicable)
    cell_types = check_key(data, args.cell_anno)
    bregma = check_key(data, args.fov)

    # Create data loader
    real_df = RealDataLoader(gene_expression,
                             coordinates,
                             threshold_distance = threshold_distance,
                             gene_list = gene_name,
                             num_class = n_c,
                             cell_labels = cell_types,
                             field = bregma,
                             for_eval = False)

    dop.save_loader(real_df, args.output)
    dop.save_smfish(real_df, args.output)

if __name__ == "__main__":
    args = get_args()
    main(args)
