#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 16:51:17 2021

@author: shengkai
"""

import umap
import numpy as np
import pickle
#import numba
import sys
import matplotlib.pyplot as plt

data_file = sys.argv[1]
n_com = int(sys.argv[2])
NN = int(sys.argv[3])
metric = sys.argv[4]

print("loading data file: %s:"%data_file)
data = np.load(data_file)

print("Param: n_components = %i \nn_neighbors = %i\nmetric = %s"%(n_com, NN, metric))

u = umap.UMAP(verbose=True, n_components = n_com,  n_neighbors=NN, metric = metric).fit(data)
np.save("%s_umap_%s_ncom%i_NN%i.npy"%(data_file.split(".")[0], metric, n_com, NN),u.embedding_)
#    pickle.dump(u_voxel_left, open("umap_left_canberra_p%03i"%i, "wb"))

plt.figure()
plt.scatter(*u.embedding_.T, s = 1, color = "grey")
plt.savefig("%s_umap_%s_ncom%i_NN%i.png"%(data_file.split(".")[0]))
