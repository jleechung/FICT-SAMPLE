"""
Created on Tue Jun 22 04:09:50 2021

@author: Haotian Teng
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#! To run this deamon completely, these datasets need to be downloaded from here:
#! MERFISH: https://datadryad.org/stash/dataset/doi:10.5061/dryad.8t8s248
#! osmFISH: https://github.com/RubD/spatial-datasets/blob/master/data/2018_osmFISH_SScortex/raw_data/osmFISH_SScortex_mouse_all_cells.loom
#! seqFISH: http://spatial.rc.fas.harvard.edu/install.html
"""
Created on Wed Jul  1 11:16:54 2020

@author: haotian teng
"""
import os 
import pickle
import pandas as pd
import anndata
from matplotlib import pyplot as plt
import numpy as np
import string
from itertools import permutations
from matplotlib import cm
from fict.utils.data_op import tsne_reduce
from matplotlib.patches import Rectangle
from sklearn.metrics.cluster import adjusted_rand_score
from mpl_toolkits.axes_grid1 import make_axes_locatable
from fict.utils import data_op as dop

plt.rcParams["font.size"] = "25"
# Set default figure size
plt.rcParams["figure.figsize"] = [3.5, 2.625]
plt.rcParams["figure.dpi"] = 300

# Set x axis
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["xtick.major.size"] = 3
plt.rcParams["xtick.major.width"] = 0.5
plt.rcParams["xtick.minor.size"] = 1.5
plt.rcParams["xtick.minor.width"] = 0.5
plt.rcParams["xtick.minor.visible"] = True
plt.rcParams["xtick.top"] = True

# Set y axis
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["ytick.major.size"] = 3
plt.rcParams["ytick.major.width"] = 0.5
plt.rcParams["ytick.minor.size"] = 1.5
plt.rcParams["ytick.minor.width"] = 0.5
plt.rcParams["ytick.minor.visible"] = False
plt.rcParams["ytick.right"] = True

# Set line widths
plt.rcParams["axes.linewidth"] = 0.5
plt.rcParams["grid.linewidth"] = 0.5
plt.rcParams["lines.linewidth"] = 1.

# Remove legend frame
plt.rcParams["legend.frameon"] = False

# Always save as 'tight'
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["savefig.pad_inches"] = 0.05

plt.rcParams["legend.frameon"] = True
gene_name = ['Ace2', 'Adora2a', 'Aldh1l1', 'Amigo2', 'Ano3', 'Aqp4', 'Ar',
       'Arhgap36', 'Avpr1a', 'Avpr2', 'Baiap2', 'Bdnf', 'Blank_1',
       'Blank_2', 'Blank_3', 'Blank_4', 'Blank_5', 'Bmp7', 'Brs3',
       'Calcr', 'Cbln1', 'Cbln2', 'Cckar', 'Cckbr', 'Ccnd2', 'Cd24a',
       'Cdkn1a', 'Cenpe', 'Chat', 'Coch', 'Col25a1', 'Cplx3', 'Cpne5',
       'Creb3l1', 'Crhbp', 'Crhr1', 'Crhr2', 'Cspg5', 'Cxcl14', 'Cyp19a1',
       'Cyp26a1', 'Cyr61', 'Dgkk', 'Ebf3', 'Egr2', 'Ermn', 'Esr1', 'Etv1',
       'Fbxw13', 'Fezf1', 'Fn1', 'Fst', 'Gabra1', 'Gabrg1', 'Gad1',
       'Galr1', 'Galr2', 'Gbx2', 'Gda', 'Gem', 'Gjc3', 'Glra3', 'Gpr165',
       'Greb1', 'Grpr', 'Htr2c', 'Igf1r', 'Igf2r', 'Irs4', 'Isl1',
       'Kiss1r', 'Klf4', 'Krt90', 'Lepr', 'Lmod1', 'Lpar1', 'Man1a',
       'Mc4r', 'Mki67', 'Mlc1', 'Myh11', 'Ndnf', 'Ndrg1', 'Necab1',
       'Nos1', 'Npas1', 'Npy1r', 'Npy2r', 'Ntng1', 'Ntsr1', 'Nup62cl',
       'Omp', 'Onecut2', 'Opalin', 'Oprd1', 'Oprk1', 'Oprl1', 'Oxtr',
       'Pak3', 'Pcdh11x', 'Pdgfra', 'Pgr', 'Plin3', 'Pnoc', 'Pou3f2',
       'Prlr', 'Ramp3', 'Rgs2', 'Rgs5', 'Rnd3', 'Rxfp1', 'Scgn', 'Selplg',
       'Sema3c', 'Sema4d', 'Serpinb1b', 'Serpine1', 'Sgk1', 'Slc15a3',
       'Slc17a6', 'Slc17a7', 'Slc17a8', 'Slc18a2', 'Slco1a4', 'Sox4',
       'Sox6', 'Sox8', 'Sp9', 'Synpr', 'Syt2', 'Syt4', 'Sytl4', 'Tacr1',
       'Tacr3', 'Tiparp', 'Tmem108', 'Traf4', 'Trhr', 'Ttn', 'Ttyh2',
       'Adcyap1', 'Cartpt', 'Cck', 'Crh', 'Gal', 'Gnrh1', 'Mbp', 'Nnat',
       'Nts', 'Oxt', 'Penk', 'Scg2', 'Sln', 'Sst']

def load(f):
    with open(f,'rb') as f:
        obj = pickle.load(f)
    return obj

def heatmap(cv,
            ax,
            xticks= None,
            yticks = None,
            title = '',
            highlight_cells = None,
            dtype = np.int,
            x_rotation = 0,
            cmap = None,
            text_annotate = True,
            cbar_label = '',
            cbar_kw={},
            divider = False,
            **kwargs):
    m,n = cv.shape
    im = ax.imshow(cv,cmap = cmap)
    if divider:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.01,aspect=20)
    if cbar_label != '':
        print("Adding color bar.")
        cbar = plt.colorbar(im, cax=cax, **cbar_kw)
        cbar.ax.set_ylabel(cbar_label, va="center",labelpad=15)
    # We want to show all ticks...
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(m))
    # ... and label them with the respective list entries
    if xticks is not None:
        ax.set_xticklabels(xticks[:n])
    if yticks is not None:
        ax.set_yticklabels(yticks[:m])
    
    # Rotate the tick labels and set their alignment.
    if x_rotation >0:
        plt.setp(ax.get_xticklabels(), rotation=x_rotation, ha="right",
                 rotation_mode="anchor")
    
    # Loop over data dimensions and create text annotations.
    if text_annotate:
        for i in range(m):
            for j in range(n):
                if cv[i,j]==0:
                    text = ax.text(j, i, "*",
                                   ha="center", va="center", color="w")
                elif dtype == np.int:
                    text = ax.text(j, i, "%.0f"%(cv[i,j]),
                                   ha="center", va="center", color="w")
                else:
                    text = ax.text(j, i, "%.2f"%(cv[i,j]),
                                   ha="center", va="center", color="w")                
    if highlight_cells is not None:
        for coor in highlight_cells:
            ax.add_patch(Rectangle((coor[0]-0.5,coor[1]-0.5), 1, 1, fill=False, edgecolor='blue', lw=3))
    ax.set_title(title)

    return ax

def cluster_visualization(posterior,loader,ax,mode = 'gene',mask = None):
    """Visualize the cluster
    Input:
        posterior: The posterior probability .
        loader: The dataloader.
        ax: The axes of the figure that is going to be printed on.
        mode: Can be one of the following mode:
            gene, neighbourhood, coordinate.
    """
    predict = np.argmax(posterior,axis = 0)
    class_n = len(set(predict))
    colors = cm.get_cmap('Set2', class_n)
    print("Reduce the dimension by T-SNE")
    if mode == 'gene':
        locs = tsne_reduce(loader.xs[0],
                                   method = 'barnes_hut')
    elif mode == 'coordinate':
        locs = loader.coordinate
    elif mode == 'neighbourhood':
        locs = tsne_reduce(loader.xs[1],method = 'barnes_hut')
    if mask is not None:
        locs = locs[mask,:]
        predict = predict[mask]
    ax.scatter(locs[:,0],
               locs[:,1],
               c=predict,
               cmap = colors,
               s = 5)
    return ax,locs,predict

def compare_visual(e_gene,e_spatio,loaders,i,j,field = None,mode = 'coordinate'):
    figs,axs = plt.subplots(nrows = 2,ncols = 2)
    figs.set_size_inches(24,h=12)
    loader = loaders[i]
    if field is not None:
        mask = loader.field==field
        mask = mask[:,0]
    else:
        mask = None
        
    cluster_visualization(e_gene[i,j,0],loader,axs[0][0],mode = mode,mask = mask)
    cluster_visualization(e_gene[i,j,1],loader,axs[0][1],mode = mode,mask = mask)
    cluster_visualization(e_spatio[i,j,0],loader,axs[1][0],mode = mode,mask = mask)
    cluster_visualization(e_spatio[i,j,1],loader,axs[1][1],mode = mode,mask = mask)
    axs[0][0].set_title("Gene model %d on dataset %d"%(i,i))
    axs[0][1].set_title("Gene model %d on dataset %d"%(j,i))
    axs[1][0].set_title("Spatio model %d on dataset %d"%(i,i))
    axs[1][1].set_title("Spatio model %d on dataset %d"%(j,i))
    return figs,axs

def confusion_matrix(e1,e2,field_mask = None):
    n1 = e1.shape[0]
    n2 = e2.shape[0]
    predict1 = np.argmax(e1,axis = 0)
    predict2 = np.argmax(e2,axis = 0)
    if field_mask is not None:
        predict1 = predict1[field_mask]
        predict2 = predict2[field_mask]
    cf_matrix = np.zeros((n1,n2))
    for i in np.arange(n1):
        for j in np.arange(n2):
            cf_matrix[i,j] = np.sum(np.logical_and(predict1==i,predict2==j))
    return cf_matrix

def greedy_match(confusion,bijection = True):
    """Match the cluster id of two cluster assignment by greedy search the 
    maximum overlaps.
    Args:
        confusion: A M-by-N confusion matrix, require M>N.
        bijection: If the match is one-to-one, require a square shape confusion
        matrix, default is True.
    Return:
        perm: A length N vector indicate the permutation, the ith value of perm
        vector is the column index of the confusion matrix match the ith row of the
        confusion matrix.
        overlap: The total overlap of the given permutation.
    """
    confusion = np.copy(confusion)
    class_n = confusion.shape[0]
    perm = np.arange(class_n)
    overlap = 0
    if bijection:
        for i in np.arange(class_n):
            ind = np.unravel_index(np.argmax(confusion, axis=None), confusion.shape)
            overlap += confusion[ind]
            perm[ind[0]] = ind[1]
            confusion[ind[0],:] = -1
            confusion[:,ind[1]] = -1
    else:
        perm = np.argmax(confusion, axis = 1)
        overlap = np.sum(np.max(confusion,axis = 1))
    return perm,overlap
    
def cluster_plot(e1,e2,loader,cell1,cell2,field = None,title = ['',''],axs = None,show_legend = False):
    predict1 = np.argmax(e1,axis = 0)
    predict2 = np.argmax(e2,axis = 0)
    if type(cell1) is np.int_ or type(cell1) is int:
        cell1 = np.asarray([cell1])
    if type(cell2) is np.int_ or type(cell2) is int:
        cell2 = np.asarray([cell2])
    mask1 = predict1 == cell1[0]
    mask2 = predict2 == cell2[0]
    for idx,c in enumerate(cell1):
        mask1 = np.logical_or(mask1,predict1 == c)
    for idx,c in enumerate(cell2):
        mask2 = np.logical_or(mask2,predict2 == cell2[idx])
    if field is not None:
        field_mask = loader.field==field
        field_mask = field_mask[:,0]
        mask1 = np.logical_and(mask1,field_mask)
        mask2 = np.logical_and(mask2,field_mask)
    locs = loader.coordinate
    locs1 = locs[mask1,:]
    locs2 = locs[mask2,:]
    if axs is None:
        figs,axs = plt.subplots(nrows = 1,ncols = 2)
        figs.set_size_inches(24,h=8)
    predict1 = predict1[mask1]
    predict2 = predict2[mask2]
    p1 = np.copy(predict1)
    p2 = np.copy(predict2)
    colors1 = cm.get_cmap('Set2', len(cell1))
    colors2 = cm.get_cmap('Set2', len(cell2))
    for idx,c in enumerate(cell1):
        p1[predict1==c] = idx
    for idx,c in enumerate(cell2):
        p2[predict2==c] = idx
    scatter1 = axs[0].scatter(locs1[:,0],
                   locs1[:,1],
                   c=p1,
                   cmap = colors1,
                   label = np.arange(len(cell1)),
                   s=5)
    axs[0].set_title(title[0])
    scatter2 = axs[1].scatter(locs2[:,0],
                   locs2[:,1],
                   c=p2,
                   cmap = colors2,
                   label = np.arange(len(cell2)),
                   s=5)
    axs[1].set_title(title[1])
    if show_legend:
        legend1 = axs[0].legend(*scatter1.legend_elements(),
                            loc="lower left", title="Classes")
        axs[0].add_artist(legend1)
        legend1 = axs[1].legend(*scatter2.legend_elements(),
                            loc="lower left", title="Classes")
        axs[1].add_artist(legend1)
    return axs

def permute_accuracy(predict,y):
    """Find the best accuracy among all the permutated clustering.
    Input args:
        predict: the clustering result.
        y: the true label.
    Return:
        best_accur: return the best accuracy.
        perm: return the permutation given the best accuracy.
    """
    predict = np.asarray(predict,dtype = np.int)
    y = np.asarray(y,dtype = np.int)
    label_tag = np.unique(y)
    predict_tag = np.unique(predict)
    sample_n = len(y)
    if len(label_tag) == len(predict_tag):
        perms = list(permutations(label_tag))
        hits = []
        for perm in perms:
            hit = np.sum([(predict == p) * (y == i) for i,p in enumerate(perm)])
            hits.append(hit)
        return np.max(hits)/sample_n,perms[np.argmax(hits)]
    else:
        hits = np.zeros(len(predict_tag))
        average_hits = np.zeros(len(predict_tag))
        max_hits = np.zeros(len(predict_tag))
        perms = np.zeros(len(predict_tag))
        for predict_i in predict_tag:
            for label_i in label_tag:
                hit = np.sum([(predict == predict_i) * (y == label_i) ])
                if hit>hits[predict_i]:
                    hits[predict_i] = hit
                    perms[predict_i] = label_i
                    max_hits[predict_i] = hit
                average_hits[predict_i]+=hit
            average_hits[predict_i] = (average_hits[predict_i] - hits[predict_i])/(len(predict_tag)-1)
        if len(predict_tag) >= len(label_tag):
            return (np.sum(hits)-np.sum(average_hits))/sample_n,perms
        else:
            return np.sum(max_hits)/sample_n,perms

def select_by_cluster(loader,expectation,ids,field = None):
    predict = np.argmax(expectation,axis = 0)
    locs = loader.coordinate
    mask = predict == ids[0]
    for i in ids:
        mask = np.logical_or(mask,predict == i)
    if field is not None:
        field_mask = loader.field==field
        field_mask = field_mask[:,0]
        mask = np.logical_and(mask,field_mask)
    locs = locs[mask,:]
    predict = predict[mask]
    return locs,predict,mask

def save_gene_expression(save_folder,loader,expectation,cluster_id,gene_name,gene_count):
    """Save the gene expression in Deseq2 readable format.
    """
    count_f = os.path.join(save_folder,"count.csv")
    meta_f = os.path.join(save_folder,"meta.csv")
    _,predict,mask = select_by_cluster(loader,expectation,cluster_id)
    expression = gene_count[mask,:]
    c_fine = np.copy(predict)
    for idx,p in enumerate(cluster_id):
        c_fine[predict==p] = idx
    sample_n = expression.shape[0]
    gene_n = expression.shape[1]
    if not os.path.isdir(save_folder):
        os.mkdir(save_folder)
    if np.sum(expression<0):
        raise ValueError("Negative expression value")
    with open(count_f,'w+') as f:
        f.write(",".join(['gene']+['cell'+str(x) for x in np.arange(sample_n)])+'\n')
        for i in np.arange(gene_n):
            f.write(",".join([gene_name[i]]+[str(x) for x in expression[:,i]])+'\n')
    with open(meta_f,'w+') as f:
        f.write("id,subcluster\n")
        for i in np.arange(sample_n):
            f.write(",".join(['cell'+str(i),'c'+str(c_fine[i])])+"\n")
    colors_fine = cm.get_cmap('cool',len(cluster_id))
    return c_fine

def significant_gene(result_f):
    result_fs = []
    for file in os.listdir(result_f):
        if file.startswith("MASTresult"):
            result_fs.append(file)
    result_fs = sorted(result_fs)
    des = []
    for r in result_fs:
        with open(os.path.join(result_f,r),'r') as f:
            de = pd.read_csv(f,header = 0,index_col = 0)
        de.sort_values(by = ["fdr"])
        de = de.set_index('primerid')
        des.append(de)
    return des



if __name__ == "__main__":
    plt.close('all')
    base7_f = "/home/heavens/twilight_data1/MERFISH_data/animal_multi2"
    base20_f = "/home/heavens/twilight_data1/MERFISH_data/animal_multi4_20class"
    model7_f = os.path.join(base7_f,'trained_models.bn')
    model20_f = os.path.join(base20_f,'trained_models.bn')
    cv7_f = os.path.join(base7_f,'cv_result.bn')
    loader7_f = os.path.join(base7_f,'loaders.bn')
    
    
    cv20_f = os.path.join(base20_f,'cv_result.bn')
    loader20_f = os.path.join(base20_f,'loaders.bn')
    loader7 = load(loader7_f)
    loader20 = load(loader20_f)
    model_7 = load(model7_f)
    model_20 = load(model20_f)
    animal_id = 0
    label,tag = dop.one_hot_vector(np.asarray(loader7[animal_id].cell_labels))
    field = 0.01
    field_mask = loader7[animal_id].field==field
    field_mask = field_mask[:,0]
    e_gene7,e_spatio7,cv_gene7,cv_spatio7 = load(cv7_f)
    e_gene20,e_spatio20,cv_gene20,cv_spatio20 = load(cv20_f)
    e_gene_coarse = e_gene7[animal_id,animal_id,0]
    e_spatio_coarse = e_spatio7[animal_id,animal_id,0]
    e_gene_fine = e_gene20[animal_id,animal_id,0]
    e_spatio_fine = e_spatio20[animal_id,animal_id,0]
    cm_gene = confusion_matrix(e_gene_fine,
                               e_gene_coarse,
                               field_mask)
    cm_spatio = confusion_matrix(e_spatio_fine,
                                 e_gene_coarse,
                                 field_mask)
    fig,axs = plt.subplots(ncols = 2, nrows = 1,figsize = (20,30))
    perm_gene,overlap_gene = greedy_match(cm_gene,bijection = False)
    perm_spatio,overlap_spatio = greedy_match(cm_spatio,bijection = False)
    
    heatmap(cm_gene,
            axs[0],
            title = 'Gene confusion matrix.',
            highlight_cells = list(zip(perm_gene,np.arange(len(perm_gene)))))
    heatmap(cm_spatio,
            axs[1],
            title = "Spatio confusion matrix.",
            highlight_cells = list(zip(perm_spatio,np.arange(len(perm_spatio)))))
    fine_cluster_n = e_spatio_fine.shape[0]
    for cluster_id in np.arange(7):
        fine_cluster = np.arange(fine_cluster_n)
        fine_cluster_ids = fine_cluster[perm_gene == cluster_id]
        cluster_plot(e_gene_coarse,
                     e_gene_fine,
                     loader7[animal_id],
                     np.asarray([cluster_id]),
                     fine_cluster_ids,
                     field = field,
                     title = ['Gene model %d of coarse cluster %d bregma %.2f'%(animal_id,cluster_id,field),'Gene model %d of the fine clusters bregma %.2f'%(animal_id,field)])
    
    for cluster_id in np.arange(7):
        fine_cluster = np.arange(fine_cluster_n)
        fine_cluster_ids = fine_cluster[perm_spatio == cluster_id]
        print(fine_cluster_ids)
        cluster_plot(e_gene_coarse,
                     e_spatio_fine,
                     loader7[animal_id],
                     np.asarray([cluster_id]),
                     fine_cluster_ids,
                     field = field,
                     title = ['Gene model %d of coarse cluster %d bregma %.2f'%(animal_id,cluster_id,field),'Spatio model %d of the fine clusters bregma %.2f'%(animal_id,field)])

