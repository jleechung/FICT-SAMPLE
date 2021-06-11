"""
Created on Fri Apr  9 13:59:42 2021

@author: Haotian Teng
"""

import umap
import numpy as np
from matplotlib import pyplot as plt
fit = umap.UMAP()
f = "/home/heavens/bridge_scratch/data/Benchmark/sim_ref_9/real/FICT_result/0/sim_gene_all.npz"
data = np.load(f)
feature = data['feature']
label = data['labels']
gene_n = feature.shape[1]
# for i in np.arange(2):
#     feature[label==i,:] = np.random.multivariate_normal(mean = np.zeros(gene_n)+i+np.random.rand(gene_n),
#                                                         cov = np.eye(gene_n),
#                                                         size = sum(label==i))
u = fit.fit_transform(feature)
plt.scatter(u[:,0],u[:,1],c = label)
plt.title('UMAP embedding of random colors')

# g_label_f = "/home/heavens/bridge_scratch/data/Benchmark/sim_ref/addictive/FICT_result/0/label_g.csv"
# g_label = np.loadtxt(g_label_f).astype(int)
# plt.scatter(u[:,0],u[:,1],c = g_label)
# plt.title('UMAP embedding of random colors')

# sg_label_f = "/home/heavens/bridge_scratch/data/Benchmark/sim_ref/addictive/FICT_result/0/label_sg.csv"
# sg_label = np.loadtxt(sg_label_f).astype(int)
# plt.scatter(u[:,0],u[:,1],c = sg_label)
# plt.title('UMAP embedding of random colors')