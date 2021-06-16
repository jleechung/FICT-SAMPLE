"""
Created on Fri Jun 11 20:26:04 2021
Mix two simulator to generate a mixed cell type assignment
@author: Haotian Teng
"""
import pickle
import argparse
import os
import sys
import numpy as np
from typing import List
from matplotlib import pyplot as plt

def mix_cell_type(sims:List, ref_sim:int = 0, rank_axis:int = 0):
    n_sim = len(sims)
    prime_sim = sims.pop(ref_sim)
    for i,sim in enumerate(sims):
        try:
            assert (sim.all_types == prime_sim.all_types).all()
        except AssertionError:
            raise AssertionError("Simulators need to have same cell types.")
        try:
            assert sim.sample_n == prime_sim.sample_n
            assert sim.gene_n == prime_sim.gene_n
        except AssertionError:
            raise AssertionError("Simulators need to have same number of samples and genes.")
    sample_n = prime_sim.sample_n
    prime_argsort = np.argsort(prime_sim.coor[:,rank_axis])
    for i,sim in enumerate(sims):
        sim.coor = sim.coor - np.mean(sim.coor,axis = 0) + np.mean(prime_sim.coor,axis = 0)
        rank_coor = sim.coor[:,rank_axis]
        argsort = np.argsort(rank_coor)
        start_idx = int(i/n_sim*sample_n)
        end_idx = int((i+1)/n_sim*sample_n)
        idxs = argsort[start_idx:end_idx]
        prime_idxs = prime_argsort[start_idx:end_idx]
        prime_sim.cell_type_assignment[prime_idxs] = sim.cell_type_assignment[idxs]
        prime_sim.coor[prime_idxs] = sim.coor[idxs]
    prime_sim.calculate_adjacency()
    prime_sim._get_neighbourhood_frequency(recount_neighbourhood = True)
    return prime_sim
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='FICT-SAMPLE',
                                     description='Train on simuulation data.')
    parser.add_argument('-s', '--sims', required = True,
                        help="The path of simulator.bin files, seperated by comma.")
    parser.add_argument('-o','--output',required = True,
                        help="The output folder of the merged simulator.")
    args = parser.parse_args(sys.argv[1:])
    sim_fs = args.sims.split(',')
    sims = []
    for sim_f in sim_fs:
        with open(sim_f,'rb') as f:
            sims.append(pickle.load(f))
    for sim in sims:
        plt.figure()
        plt.scatter(sim.coor[:,0],
                    sim.coor[:,1],
                    c = sim.cell_type_assignment)
    prime_sim = mix_cell_type(sims,rank_axis = 1)
    if not os.path.isdir(args.output):
        os.mkdir(args.output)
    plt.figure()
    plt.scatter(prime_sim.coor[:,0],
                prime_sim.coor[:,1],
                c = prime_sim.cell_type_assignment)
    with open(os.path.join(args.output,'simulator.bin'),'wb+') as f:
        pickle.dump(prime_sim,f)
