#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 19 17:35:21 2022

@author: shengkai
"""

import numpy as np
import mrcfile
#import pandas as pd
#import seaborn as sns
import networkx as nx
import pickle
from networkx.algorithms.components.connected import connected_components
from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage
import os
import sys
#%% function load all mrc in a dir

threshold = float(sys.argv[1])

def load_mrc(dp, box):
    if dp[-1]!="/": dp = dp + "/"
    name_list = [i for i in os.listdir(dp) if i.split(".")[-1]=="mrc"]
    name_list.sort()
    num = len(name_list)
    temp = np.zeros((num, box**3))
    for i, name in enumerate(name_list):
        temp[i] = mrcfile.open(dp + name).data.reshape(-1)
    return (name_list, temp)

print("loading mrc...")
map_name, mrc  = load_mrc(".", 160)
print("data loaded")
#%%
os.system("mkdir hier_result_tr_%.2f"%threshold)
#%%
"""
G_temp = nx.Graph()
c_temp = np.array(range(160**3))
r_temp = c_temp.reshape((160,160,160))
for i in range(40,120):
    for j in range(40,120):
        for k in range(40,120):
            G_temp.add_edge(r_temp[i,j,k], r_temp[i+1,j,k])
            G_temp.add_edge(r_temp[i,j,k], r_temp[i,j+1,k])
            G_temp.add_edge(r_temp[i,j,k], r_temp[i-1,j,k])
            G_temp.add_edge(r_temp[i,j,k], r_temp[i,j-1,k])
            G_temp.add_edge(r_temp[i,j,k], r_temp[i+1,j+1,k])
            G_temp.add_edge(r_temp[i,j,k], r_temp[i-1,j+1,k])
            G_temp.add_edge(r_temp[i,j,k], r_temp[i-1,j+1,k])
            G_temp.add_edge(r_temp[i,j,k], r_temp[i-1,j-1,k])
            G_temp.add_edge(r_temp[i,j,k], r_temp[i,j,k+1])
            G_temp.add_edge(r_temp[i,j,k], r_temp[i,j+1,k+1])
            G_temp.add_edge(r_temp[i,j,k], r_temp[i+1,j,k+1])
            G_temp.add_edge(r_temp[i,j,k], r_temp[i-1,j,k+1])
            G_temp.add_edge(r_temp[i,j,k], r_temp[i,j-1,k+1])
"""
print("create octet connection")
G_temp = nx.Graph()
c_temp = np.array(range(160**3))
r_temp = c_temp.reshape((160,160,160))
for i in range(40,120):
    for j in range(40,120):
        for k in range(40,120):
            G_temp.add_edge(r_temp[i,j,k], r_temp[i+1,j,k])
            G_temp.add_edge(r_temp[i,j,k], r_temp[i,j+1,k])
            G_temp.add_edge(r_temp[i,j,k], r_temp[i-1,j,k])
            G_temp.add_edge(r_temp[i,j,k], r_temp[i,j-1,k])
            G_temp.add_edge(r_temp[i,j,k], r_temp[i,j,k+1])
            G_temp.add_edge(r_temp[i,j,k], r_temp[i,j,k-1])


#%%
def dust_seg(seg, connected_map, size_threshold=250):
    t_temp = set(np.where(seg==1)[0])
    xx = connected_map.subgraph(t_temp)
    connected_xx = list(connected_components(xx))
    xx_len = np.array(list(map(len, connected_xx)))
    xx_len_t = np.where(xx_len>size_threshold)[0]
    print(xx_len_t)

    divided_seg =  np.zeros((len(xx_len_t), len(seg)))
    for i,t in enumerate(xx_len_t):
        divided_seg[i, list(connected_xx[t])] = 1           
    
    return xx_len, divided_seg

def mass_cal(voxel_num, pix_size = 2.62, density = 1.32):
    return (voxel_num*(pix_size**3)/10*density*6.02/1000)

#%%
def two_map_mass_diff(map1, map2, size_threshold=250, sigma_num = 3, connected_map = G_temp):
    diff_map = map1*1 - map2*1
    diff_map_std = diff_map[diff_map!=0].std()
    print(diff_map_std)
    m1_m2 = diff_map > diff_map_std*sigma_num
    m2_m1 = diff_map < -diff_map_std*sigma_num
    return(mass_cal(dust_seg(m1_m2, size_threshold=size_threshold, connected_map =connected_map)[1].sum()), 
           mass_cal(dust_seg(m2_m1, size_threshold=size_threshold, connected_map =connected_map)[1].sum()))

#%%
def two_map_mass_diff_bi(map1, map2, size_threshold=250, connected_map = G_temp):
    diff_map = map1*1 - map2*1
    m1_m2 = diff_map > 0.5
    m2_m1 = diff_map < -0.5
    return(mass_cal(dust_seg(m1_m2, size_threshold=size_threshold, connected_map =connected_map)[1].sum()), 
           mass_cal(dust_seg(m2_m1, size_threshold=size_threshold, connected_map =connected_map)[1].sum()))

#%%
def mass_diff_matrix(test_maps, test_map_name_list, size_threshold=250, sigma_num = 3):
    map_num = test_maps.shape[0]
    matrix = np.zeros((map_num, map_num))
    print("number of pairs: %i"%(map_num*(map_num+1)/2))
    for i in range(map_num-1):
        for j in range(i+1,map_num):
            print("computing %i %i"%(i,j))
            diff = two_map_mass_diff(test_maps[i], test_maps[j], size_threshold=size_threshold, sigma_num = sigma_num)
            matrix[i,j] = diff[0]
            matrix[j,i] = diff[1]
    return(matrix)
#%%
def mass_diff_matrix_bi(test_maps, test_map_name_list, size_threshold=250):
    map_num = test_maps.shape[0]
    matrix = np.zeros((map_num, map_num))
    print("number of pairs: %i"%(map_num*(map_num+1)/2))
    for i in range(map_num-1):
        for j in range(i+1,map_num):
            print("computing %i %i"%(i,j))
            diff = two_map_mass_diff_bi(test_maps[i], test_maps[j], size_threshold=size_threshold)
            matrix[i,j] = diff[0]
            matrix[j,i] = diff[1]
    return(matrix)

#%%

def mass_diff_matrix_add(matrix):
    temp_matrix = np.zeros(matrix.shape)
    for i in range(matrix.shape[0]-1):
        for j in range(i+1,matrix.shape[0]):
            temp_matrix[i,j] = np.abs(matrix[i,j])+np.abs(matrix[j,i])
            temp_matrix[j,i] = temp_matrix[i,j]
    return temp_matrix
#%%
"""
a = mass_diff_matrix(test_maps, test_map_name_list)
a_label = [i.split(".")[0] for i in test_map_name_list]

dist_mat = mass_diff_matrix_add(a)

np.save("./hier_result/fold_mat.npy", a)
np.save("./hier_result/dist_mat.npy", dist_mat)
plt.figure()
linkage_matrix = linkage(np.array([dist_mat[i,j] for i in range(dist_mat.shape[0]-1) for j in range(i+1,dist_mat.shape[0])]), "average")
dendrogram(linkage_matrix, color_threshold=1, labels=a_label,show_leaf_counts=True)

plt.savefig("./hier_result/fdendrogram.png")
plt.close()

#%%            
l17_F = np.load("/Users/shengkai/Desktop/data/P26/L17_hier/compare_to_Frealign/l17_unmasked_frealign.npy")
l17_F_name =  np.load("/Users/shengkai/Desktop/data/P26/L17_hier/compare_to_Frealign/l17_unmasked_frealign_name.npy")

l17_F_bi = l17_F>(l17_F.mean(1)+3*l17_F.std(1)).reshape(-1,1)
l17_F_bi = l17_F_bi*1

l17_C_name, l17_C = load_mrc('/Users/shengkai/Desktop/data/P26/L17_hier/mrc', 160)
l17_C_label = [i.split("P")[-1].split("_")[1] for i in l17_C_name]

l17_C_bi = l17_C>np.array([i[i!=0].mean()+4.5*i[i!=0].std() for i in l17_C]).reshape(-1,1)


l17_F_bi = l17_F > np.percentile(l17_F, 99, 1).reshape(-1,1)
l17_C_bi = l17_C > np.percentile(l17_C, 99, 1).reshape(-1,1)

"""
print("mrc binarized by threshold %.2f"%threshold)
map_label = [i.split(".")[0] for i in map_name]
mrc_bi = mrc > threshold

print("calculating mass diff")
a = mass_diff_matrix_bi(mrc_bi, map_label)

#%%
dist_mat = mass_diff_matrix_add(a)


np.save("./hier_result_tr_%.2f/fold_mat.npy"%threshold, a)
np.save("./hier_result_tr_%.2f/dist_mat.npy"%threshold, dist_mat)
np.save("./hier_result_tr_%.2f/label.npy"%threshold, map_label)
plt.figure()
linkage_matrix = linkage(np.array([dist_mat[i,j] for i in range(dist_mat.shape[0]-1) for j in range(i+1,dist_mat.shape[0])]), "average")
dendrogram(linkage_matrix, color_threshold=1, labels=map_label,show_leaf_counts=True,leaf_font_size=12)
plt.axhline(y=10, color = "red")
plt.savefig("./hier_result_tr_%.2f/dendrogram.png"%threshold)
plt.close()

#%%
"""
b = mass_diff_matrix_bi(np.vstack((l17_C_bi,l17_F_bi)), l17_C_name + [i for i in map_namelist[:42]])
#%%
dist_mat = mass_diff_matrix_add(b)

np.save("./hier_result/fold_mat_bi.npy", b)
np.save("./hier_result/dist_mat_bi.npy", dist_mat)

plt.figure()
linkage_matrix = linkage(np.array([dist_mat[i,j] for i in range(dist_mat.shape[0]-1) for j in range(i+1,dist_mat.shape[0])]), "average")
dendrogram(linkage_matrix, color_threshold=1,labels=l17_C_label + ["F%02i"%i for i in range(42)],show_leaf_counts=True,leaf_font_size=12)
"""
