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
import json
import pickle
import requests
import pandas as pd
import anndata
from matplotlib import pyplot as plt
import numpy as np
import string
from itertools import permutations
from subprocess import Popen, PIPE
from fict.utils.data_op import tsne_reduce
from matplotlib.patches import Rectangle
from sklearn.metrics.cluster import adjusted_rand_score
from mpl_toolkits.axes_grid1 import make_axes_locatable
from fict.utils import data_op as dop

def load(f):
    with open(f,'rb') as f:
        obj = pickle.load(f)
    return obj

def confusion_matrix(e1,e2):
    n1 = e1.shape[0]
    n2 = e2.shape[0]
    predict1 = np.argmax(e1,axis = 0)
    predict2 = np.argmax(e2,axis = 0)
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

def select_by_cluster(expectation,ids,field = None):
    predict = np.argmax(expectation,axis = 0)
    mask = predict == ids[0]
    for i in ids:
        mask = np.logical_or(mask,predict == i)
    predict = predict[mask]
    return predict,mask

def save_gene_expression(save_folder,expectation,cluster_id,gene_name,gene_count):
    """Save the gene expression in Deseq2 readable format.
    """
    count_f = os.path.join(save_folder,"count.csv")
    meta_f = os.path.join(save_folder,"meta.csv")
    predict,mask = select_by_cluster(expectation,cluster_id)
    expression = gene_count[mask,:]
    c_fine = np.copy(predict)
    for idx,p in enumerate(cluster_id):
        c_fine[predict==p] = idx
    sample_n = expression.shape[0]
    gene_n = expression.shape[1]
    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)
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

def find_cluster_by_cell_type(expectation:np.ndarray,
                              reference:np.ndarray,
                              cell_type_name:str):
    if cell_type_name not in reference:
        raise ValueError("Given cell type %s is not in the reference cell type"%(cell_type_name))
    prediction = np.argmax(expectation,axis = 0)
    uniq,counts = np.unique(prediction[reference==cell_type_name],return_counts=True)
    return uniq[np.argmax(counts)]

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

def extract_genes(e_coarse,
                  e_fine,
                  gene_name,
                  cluster_id,
                  gene_count_bg,
                  save_f = '.'
                  ):
    fine_cluster_n = e_fine.shape[0]
    cm = confusion_matrix(e_fine,
                          e_coarse)
    perm,overlap = greedy_match(cm,bijection = False)
    fine_clusters = np.arange(fine_cluster_n)
    fine_clusters = fine_clusters[perm == cluster_id]
    save_gene_expression(os.path.join(save_f,'genes/'),e_fine,fine_clusters,gene_name,gene_count_bg)

def go_enrichment_analysis(gene_list,
                           taxid = "9606",
                           test_type = 'FISHER',
                           correction = 'FDR',
                           query_type = 'biological process'):
    """Conduct GO enrichment analysis given a gene name list.
    Args:
        gene_list: A list of genes that conduct GO enrichment analysis on.
        taxid: The taxid of the species doing analysis on, default is homo spanies(9606), can be 10090(Mouse).
        test_type: The statistic type that used, default conduct Fisher test, can be BINOMIAL
        correction: The correction method used, default is FDR, can be BONFERRONI or NONE.
        
    """
    query_ref = {'biological process':"GO%3A0008150",
                  'molecular function':"GO%3A0003674",
                  'cellular component':"GO%3A0005575"}
    query_type = query_ref[query_type]
    base_URL = "http://pantherdb.org/services/oai/pantherdb/enrich/overrep?"
    search_URL = "geneInputList=%s&organism=%s&annotDataSet=%s&enrichmentTestType=%s&correction=%s"%(','.join(gene_list),
                                          taxid,
                                          query_type,
                                          test_type,
                                          correction)
    request_URL = base_URL+search_URL
    r = requests.get(request_URL, headers={ "Accept" : "application/json"})
    if not r.ok:
      r.raise_for_status()
    result = json.loads(r.text)
#    root = ET.fromstring(r.text)
#    name = root[1].text
    return result['results']['result']

def get_Gos(des,n_Gos = 10):
    """
    des is the output from significant_gene function, contains the information
    about the differentially expressed genes by Deseq2.
    """
    cluster_Gos = []
    cluster_fdrs = []
    MIN_P = 1/np.finfo(np.double).max
    for de in des:
        Go_list = go_enrichment_analysis(list(de['symbolid']),taxid = "10090")
        significant_Gos = []
        fdrs = np.asarray([x['fdr'] for x in Go_list])
        Go_list = np.asarray(Go_list)
        argsort = np.argsort(fdrs)
        fdrs = fdrs[argsort]
        Go_list = Go_list[argsort]
        significant_Gos = Go_list[fdrs<0.05]
        significant_Gos = significant_Gos[:n_Gos]
        cluster_Gos.append(significant_Gos)
        cluster_fdrs.append({x['term']['id']+'-'+x['term']['label']:x['fdr'] for x in significant_Gos})
    Gos = set()
    for cluster_Go in cluster_Gos:
        Gos = Gos.union(set([x['term']['id']+'-'+x['term']['label'] for x in cluster_Go]))
    Gos = list(Gos)
    P_Go = np.ones((len(Gos),len(des)),dtype = np.float)
    for i,G in enumerate(Gos):
        for j,de in enumerate(des):
            if G in cluster_fdrs[j].keys():
                P_Go[i,j] = cluster_fdrs[j][G] if cluster_fdrs[j][G]>0 else MIN_P
    return P_Go,Gos


if __name__ == "__main__":
    CONFIG = {"ANIMAL_ID":0,
              "FIELD_DEPTH":None,
              "DEGENE_METHOD":"Deseq2",
              "CHOSEN_CELL_TYPES":["Excitatory","Inhibitory"]#Can be MAST
              }
    ## Debugging code
    class Args:
        pass
    args = Args()
    args.output = "/home/heavens/CMU/FISH_Clustering/FICT-SAMPLE/datasets/GO_enrichment_analysis_result/"
    ##
    
    if not os.path.isdir(args.output):
        os.mkdir(args.output)
    ## Original data file:
    data_f = "./datasets/Moffitt_and_Bambah-Mukku_et_al_merfish_all_cells.csv"
    
    ## INPUT FILES FOR FICT
    base7_f = "/home/heavens/BRIDGE_SCRATCH/data/MERFISH_data/cv_result_multi/animal1-4+grid+embedding+ridge_cov/1-4+SF0.1+R30"
    base20_f = "/home/heavens/BRIDGE_SCRATCH/data/MERFISH_data/cv_result_multi/animal1-4+grid+embedding+ridge_cov/1-4+SF1+R30"
    model7_f = os.path.join(base7_f,'trained_models.bn')
    model20_f = os.path.join(base20_f,'trained_models.bn')
    cv7_f = os.path.join(base7_f,'cv_result.bn')
    cv20_f = os.path.join(base20_f,'cv_result.bn')
    loader7_f = os.path.join(base7_f,'loaders.bn')
    loader20_f = os.path.join(base20_f,'loaders.bn')
    
    ## LOAD INPUT FILES
    ### Load original data
    print("Load MERFISH data.")
    animal_id = CONFIG["ANIMAL_ID"]
    field = CONFIG["FIELD_DEPTH"]
    GENE_COL = np.arange(9,164)
    data_all = pd.read_csv(data_f,header = 0)
    gene_expression_all = data_all.iloc[:,GENE_COL]
    nan_cols = np.unique(np.where(np.isnan(gene_expression_all))[1])
    for nan_col in nan_cols:
        GENE_COL = np.delete(GENE_COL,nan_col)
    gene_name = data_all.columns[GENE_COL]
    data = data_all[data_all['Animal_ID']==animal_id+1]
    cell_types = data['Cell_class'].to_numpy()
    data = data[cell_types!= 'Ambiguous']
    cell_types = data['Cell_class'].to_numpy()
    e_ref = dop.one_hot_vector(cell_types)
    gene_count_full = data.iloc[:,GENE_COL]
    
    ## LOAD MODEL PREDICTION
    print("Load model predictions.")
    ### Load FICT&GMM result
    print("Load FICT&GMM result.")
    loader7 = load(loader7_f)
    loader20 = load(loader20_f)
    model_7 = load(model7_f)
    model_20 = load(model20_f)
    label,tag = dop.one_hot_vector(np.asarray(loader7[animal_id].cell_labels))
    if field:
        field_mask = loader7[animal_id].field==field
        field_mask = field_mask[:,0]
    else:
        field_mask = np.ones(len(cell_types),dtype = bool)
    cell_types = cell_types[field_mask]
    e_gene7,e_spatio7,_,_= load(cv7_f)
    e_gene20,e_spatio20,_,_= load(cv20_f)
    e_gene_coarse = e_gene7[animal_id,animal_id,0][:,field_mask]
    e_spatio_coarse = e_spatio7[animal_id,animal_id,0][:,field_mask]
    e_gene_fine = e_gene20[animal_id,animal_id,0][:,field_mask]
    e_spatio_fine = e_spatio20[animal_id,animal_id,0][:,field_mask]
    
    ### Load scanpy result
    print("Load Scanpy result.")
    e_scanpy_fine = None
    ### Load smfish result
    print("Load SmfishHmrf result.")
    e_smfish_fine = None
    ### Load seurat result
    print("Load Seurat result.")
    e_seurat_fine = None
    
    ec = [e_gene_coarse,e_spatio_coarse,e_ref,e_ref,e_ref]
    ef = [e_gene_fine,e_spatio_fine,e_smfish_fine,e_scanpy_fine,e_seurat_fine]
    method = ["GMM","FICT","SmfishHmrf","Scanpy","Seurat"]
    
    ## EXTRACT THE DIFFERENTIAL EXPRESSED GENES GIVEN COARSE AND FINE CLUSTER
    if CONFIG["DEGENE_METHOD"] == "Deseq2":
        gene_count_full = np.asarray(gene_count_full,dtype = np.int)
        gene_count_full += 1
    else:
        gene_count_full = np.asarray(gene_count_full,dtype = np.double)
        gene_count_full += 1e-5
        
    for c,f,m in zip(ec,ef,method):
     for ct in CONFIG["CHOSEN_CELL_TYPES"]:
        print("GO analysis for %s method on %s cell type"%(m,ct))
        cluster_id = find_cluster_by_cell_type(c,cell_types,ct)
        save_f = os.path.join(args.output,m,ct)
        extract_genes(c,f,gene_name,cluster_id,gene_count_full,save_f = save_f)
        p = Popen(['/usr/bin/Rscript', '--vanilla',"DEgene_analysis_deseq2.R",
                   os.path.join(save_f,"genes"),
                   os.path.join(save_f,"genes")], stdin=PIPE, stdout=PIPE, stderr=PIPE)
        output, err = p.communicate(b"input data that is passed to subprocess' stdin")
        print(p.returncode, output.decode(), err.decode())
        des = significant_gene(save_f)
        P_Go,Gos = get_Gos(des)

