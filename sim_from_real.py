#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 15:41:29 2020

@author: heavens
"""
from fict.utils.joint_simulator import Simulator
from fict.utils.joint_simulator import SimDataLoader
from fict.utils.joint_simulator import get_gene_prior
from fict.utils.joint_simulator import get_nf_prior
from fict.utils.opt import valid_neighbourhood_frequency
from fict.utils.scsim import Scsim
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn import manifold
from fict.fict_input import RealDataLoader
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import pickle
import argparse,sys,os
import numpy as np
from matplotlib import pyplot as plt
import warnings
from os.path import join
warnings.filterwarnings("ignore")
def plot_freq(neighbour,axes,color,cell_tag):
    sample_n = neighbour.shape[1]
    neighbour = neighbour/np.sum(neighbour,axis = 1,keepdims = True)
    std = np.std(neighbour, axis = 0)/np.sqrt(sample_n)
    mean = np.mean(neighbour, axis = 0)
    x = np.arange(sample_n)
    yerror = np.asarray([-std,std])
#    make_error_boxes(axes, x, mean, yerror = yerror)
    patches = axes.boxplot(neighbour,
                        vert=True,  # vertical box alignment
                        patch_artist=True,
                        notch=True,
                        usermedians = mean) # fill with color
    for patch in patches['boxes']:
        patch.set_facecolor(color)
        patch.set_color(color)
        patch.set_alpha(0.5)
    for patch in patches['fliers']:
        patch.set_markeredgecolor(color)
        patch.set_color(color)
    for patch in patches['whiskers']:
        patch.set_color(color)
    for patch in patches['caps']:
        patch.set_color(color)
    axes.errorbar(x+1,mean,color = color,label = cell_tag)
    return mean,yerror

def read_prior(data_f,n_c,header = 1,gene_col = np.arange(9,164),coor_col = [5,6]):
    ### Data preprocessing
    print("Reading data from %s"%(data_f))
    data = pd.read_excel(data_f,header = header,engine = 'openpyxl')
    cell_types = data['Cell_class']
    data = data[cell_types!= 'Ambiguous']
    cell_types = data['Cell_class']
    X = data['Centroid_X'].astype(np.float)
    Y = data['Centroid_Y'].astype(np.float)
    gene_expression = data.iloc[:,gene_col]
    type_tags = np.unique(cell_types)
    coordinates = data.iloc[:,coor_col]
    ### Choose only the n_c type cells
    print("Choose the subdataset of %d cell types"%(n_c))
    if len(type_tags)<n_c:
        raise ValueError("Only %d cell types presented in the dataset, but require %d, reduce the number of cell type assigned."%(len(type_tags),n_c))
    mask = np.asarray([False]*len(cell_types))
    for tag in type_tags[:n_c]:
        mask = np.logical_or(mask,cell_types==tag)
    gene_expression = gene_expression[mask]
    cell_types = np.asarray(cell_types[mask])
    coordinates = np.asarray(coordinates[mask])
    
    ### Generate prior from the given dataset.
    gene_mean,gene_std = get_gene_prior(gene_expression,cell_types)
    neighbour_freq_prior,tags,type_count = get_nf_prior(coordinates,cell_types)
    return gene_mean,gene_std,neighbour_freq_prior,tags,type_count,np.asarray([X,Y]).transpose()

def simulation(sim_folder,
               sample_n = 2000,
               n_g = 1000,
               n_c = 3,
               density = 20,
               threshold_distance = 1,
               using_splatter = False,
               method = 0,
               data_f = "datasets/aau5324_Moffitt_Table-S7.xlsx",
               use_refrence_coordinate = False,
               *args,
               **kwargs):
    if not os.path.isdir(sim_folder):
        os.mkdir(sim_folder)
    methods = ['addictive','exclusive','stripe','real']
    o_f = join(sim_folder,"%s"%(methods[method]))
    if not os.path.isdir(o_f):
        os.mkdir(o_f)
    if not os.path.isdir(join(o_f,"figures")):
        os.mkdir(join(o_f,"figures"))
    if (method == 3) | use_refrence_coordinate:
        gene_mean,gene_std,neighbour_freq_prior,tags,type_count,coor = read_prior(data_f = data_f,n_c = n_c,*args,**kwargs)
    print("######## Begin simulation with %s configuration ########"%(methods[method]))
    def addictive_freq(n_c):
        target_freq = np.ones((n_c,n_c))
        for i in np.arange(n_c):
            target_freq[i,i] = 4*(n_c-1)
        target_freq/np.sum(target_freq,axis=1,keepdims=True)
        return valid_neighbourhood_frequency(target_freq)[0]
    
    def exclusive_freq(n_c):
        target_freq = np.ones((n_c,n_c))
        for i in np.arange(n_c):
            target_freq[i,i] = 3*(n_c-1)
            if i%2 == 1:
                target_freq[i-1,i] = 3*(n_c-1)
                target_freq[i,i-1] = 3*(n_c-1)
        target_freq/np.sum(target_freq,axis=1,keepdims=True)
        return valid_neighbourhood_frequency(target_freq)[0]
    
    def stripe_freq(n_c):
        target_freq = np.ones((n_c,n_c))
        for i in np.arange(n_c):
            target_freq[i,i] = 3*(n_c-1)
            if i>0:
                target_freq[i-1,i] = 3*(n_c-1)
        target_freq/np.sum(target_freq,axis=1,keepdims=True)
        return valid_neighbourhood_frequency(target_freq)[0]
    def real_freq(n_c):
        assert len(neighbour_freq_prior) == n_c
        target_freq = np.asarray(neighbour_freq_prior)
        target_freq/np.sum(target_freq,axis=1,keepdims=True)
        return valid_neighbourhood_frequency(target_freq)[0]
    
    freq_map = {0:addictive_freq,1:exclusive_freq,2:stripe_freq,3:real_freq}
    target_freq = freq_map[method](n_c)
    
    sim = Simulator(sample_n,n_g,n_c,density)
    sim.gen_parameters(gene_mean_prior = None)
    if use_refrence_coordinate:
        reference_coordinate = coor
    else:
        reference_coordinate = None
    sim.gen_coordinate(density = density,
                       ref_coor = reference_coordinate)
    
    ### Assign cell types by Gibbs sampling and load
    print("Assign cell type using Gibbs sampling.")
    sim.assign_cell_type(target_neighbourhood_frequency=target_freq, 
                         method = "Gibbs-sampling",
                         max_iter = 500,
                         use_exist_assignment = False)
    print("Refine cell type using Metropolis–Hastings algorithm.")
    sim.assign_cell_type(target_neighbourhood_frequency=target_freq, 
                         method = "Metropolis-swap",
                         max_iter = 30000,
                         use_exist_assignment = True,
                         annealing = False)
    fig,axs = plt.subplots()
    axs.scatter(sim.coor[:,0],sim.coor[:,1], c = sim.cell_type_assignment,s = 20)
    axs.set_title("Cell type assignment after assign_neighbour")
    axs.set_xlabel("X")
    axs.set_ylabel("Y")
    fig.savefig(join(o_f,"figures/Cell_location.png"))
    
    sim._get_neighbourhood_frequency()
    
    if using_splatter:
        print("Generate gene expression using Splatter.")
        sim_gene_expression,sim_cell_type,sim_cell_neighbour = sim.gen_expression_splatter()
    else:
        print("Generate gene expression.")
        sim_gene_expression,sim_cell_type,sim_cell_neighbour = sim.gen_expression(drop_rate = None)
        
    ### Save the simulator to the file
    print("Saving...")
    with open(join(o_f,"simulator.bin"),'wb+') as f:
        pickle.dump(sim,f)
    np.savetxt(join(o_f,"simulator.csv"),sim_gene_expression,delimiter = ",")
    
    ### Show the neighbourhood frequency of generated dataset
    print("Generate neighbourhood frequency plot.")
    mask = np.zeros(sim_cell_type.shape)
    test_cell = np.arange(n_c)
    for cell_idx in test_cell:
        mask = np.logical_or(mask,sim_cell_type == cell_idx)
    partial_cell_type = sim_cell_type[mask]
    partial_neighbour = sim_cell_neighbour[mask]
    fig,axs = plt.subplots()
    colors = ['green', 'blue','red','yellow','purple']
    for i,cell_idx in enumerate(test_cell):
        freq_true,yerror = plot_freq(partial_neighbour[partial_cell_type == cell_idx],
                                     axes = axs,
                                     color = colors[i],
                                     cell_tag = test_cell[i])
    nb_freqs = np.zeros((n_c,n_c))
    for i in np.arange(n_c):
        parital_nb = sim_cell_neighbour[sim_cell_type==i]
        freq = parital_nb/np.sum(parital_nb,axis = 1,keepdims = True)
        nb_freqs[i,:] = np.mean(freq,axis = 0)
    plt.title("Generated neighbourhood frequency of cell %d %d and %d."%(test_cell[0],test_cell[1],test_cell[2]))
    plt.xlabel("Cell type")
    plt.ylabel("Frequency")
    plt.savefig(join(o_f,"figures/Neighbourhood_Frequency.png"))
    print("Target neighbourhood frequency:")
    print(target_freq)
    print("Generated neighbourhood frequency:")
    print(nb_freqs)
    if use_refrence_coordinate:
        fig,axs = plt.subplots()
        axs.scatter(sim.ref_coor[:,0],
                    sim.ref_coor[:,1],
                    color = 'red',
                    label = "All cells.",
                    s = 1)
        axs.scatter(sim.coor[:,0],
                    sim.coor[:,1],
                    color = 'yellow',
                    label = "Selected cells.",
                    s = 1)
        plt.legend()
        fig.savefig(join(o_f,"figures/Coordinate_sampling.png"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='FICT-SAMPLE',
                                     description='Generate simulation dataset.')
    parser.add_argument('-o','--output',required = True,
                        help = "Output folder.")
    parser.add_argument('--n_type', default = 3,
                        help = "The number type of cells generated.")
    parser.add_argument("--splatter",action = "store_true",
                        help = "If we are going to use splatter.")
    parser.add_argument("--reference_coordinate", action = "store_true",
                        help = "If we are going to use the coordinates of\
                        reference dataset.")
    args = parser.parse_args(sys.argv[1:])
    for method in np.arange(4):
        simulation(args.output,
                   method = method,
                   n_c = args.n_type,
                   using_splatter= args.splatter,
                   use_refrence_coordinate=args.reference_coordinate)