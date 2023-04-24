#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 17:14:17 2022

@author: shengkai

kinetic simulation with observations

"""

# Considering there are 7 segments in assembly process
# a length 7 array is a binary array 
# The dependency is:
# seg1 and seg2 -> seg3 -> seg4 -> seg5 --(seg6 bind and leave) - seg7

# there are several state can be found:
# nothing bound: [0,0,0,0,0,0,0]
# seg1 only :    [1,0,0,0,0,0,0]
# seg2 only :    [0,1,0,0,0,0,0]
# seg3 binds:    [1,1,1,0,0,0,0]
# seg4 binds:    [1,1,1,1,0,0,0]
# seg5 binds:    [1,1,1,1,1,0,0]
# seg6 binds:    [1,1,1,1,1,1,0]
# seg6 leaves and seg7 binds:    [1,1,1,1,1,0,1]


# 1st assumption every state has equal amount(20)

state=[[0,0,0,0,0,0,0],
[1,0,0,0,0,0,0],
[0,1,0,0,0,0,0],
[1,1,1,0,0,0,0],
[1,1,1,1,0,0,0],
[1,1,1,1,1,0,0],
[1,1,1,1,1,1,0],
[1,1,1,1,1,0,1]]

#%% load package and function
import numpy as np
import re
import os
import mrcfile
import matplotlib.pyplot as plt
import pandas as pd
import umap
from sklearn.decomposition import PCA
import seaborn as sns
import hdbscan
import pickle
from itertools import compress
import random
import copy
import networkx as nx

def save_density(data, grid_spacing, outfilename, origin=None):
    """
    Save the density to an mrc file. The origin of the grid will be (0,0,0)
    • outfilename: the mrc file name for the output
    """
    print("Saving mrc file ...")
    data = data.astype('float32')
    with mrcfile.new(outfilename, overwrite=True) as mrc:
        mrc.set_data(data)
        mrc.voxel_size = grid_spacing
        if origin is not None:
            mrc.header['origin']['x'] = origin[0]
            mrc.header['origin']['y'] = origin[1]
            mrc.header['origin']['z'] = origin[2]
        mrc.update_header_from_data()
        mrc.update_header_stats()
    print("done")
    
from sklearn import  linear_model
from sklearn.metrics import  r2_score, mean_squared_error

from networkx.algorithms.components.connected import connected_components


#%%

def to_graph(l):
    G = nx.Graph()
    for part in l:
        # each sublist is a bunch of nodes
        G.add_nodes_from(part)
        # it also imlies a number of edges:
        G.add_edges_from(to_edges(part))
    return G

def to_edges(l):
    """ 
        treat `l` as a Graph and returns it's edges 
        to_edges(['a','b','c','d']) -> [(a,b), (b,c),(c,d)]
    """
    it = iter(l)
    last = next(it)

    for current in it:
        yield last, current
        last = current    

def solve_qua(d, i, j):
    a = [[2,3],[4,5]]
    temp = [i,j,0,0,0,0]
    for i, t in enumerate(d[0]):
        temp[a[t[0]][t[1]]] = d[1][i] 
    return temp

def solve_uull(u1, u2, l1, l2, i, j):
    t00 = np.logical_and(l1, l2).sum()
    t01 = np.logical_and(l1, u2).sum()
    t10 = np.logical_and(u1, l2).sum()
    t11 = np.logical_and(u1, u2).sum()
    return((i,j,t00,t01,t10,t11))
    
#%% function load all mrc in a dir

def load_mrc(dp, box):
    if dp[-1]!="/": dp = dp + "/"
    name_list = [i for i in os.listdir(dp) if i.split(".")[-1]=="mrc"]
    name_list.sort()
    num = len(name_list)
    temp = np.zeros((num, box**3))
    for i, name in enumerate(name_list):
        temp[i] = mrcfile.open(dp + name).data.reshape(-1)
    return (name_list, temp)

#%% sns_cluster
def sns_occ_cluster2(test_map, seg, metric = "euclidean", norm = False, norm_col = 0, transpose  = False, give_name = False, test_map_name = 0, seg_name = 0):
    
    occ = np.matmul(test_map, seg.T)/seg.sum(1).reshape(1,-1)
    if norm:
        occ = occ/occ[:, norm_col].reshape(-1,1)
    col_name = seg_name
    row_name = test_map_name
    
    if transpose:
        occ = occ.T
        col_name = test_map_name
        row_name = seg_name
    
    if give_name:
        df = pd.DataFrame(occ, columns= col_name, index = row_name)
        cl = sns.clustermap(df, xticklabels=True, yticklabels=True, cmap = "rocket_r", metric = metric)
    else:    
        cl = sns.clustermap(occ, xticklabels=True, yticklabels=True, cmap = "rocket_r", metric = metric)
    
    return occ, cl

#%%
def sns_occ_cluster(test_map, seg, metric = "euclidean", norm = False, norm_col = 0, transpose  = False):
    if not transpose:
        if norm:
            occ = np.matmul(test_map, seg.T)/seg.sum(1).reshape(1,-1)
            occ = occ/occ[:, norm_col].reshape(-1,1)
            sns.clustermap(occ, xticklabels=True, yticklabels=True, cmap = "rocket_r", metric = metric)
        else:
            occ = np.matmul(test_map, seg.T)/seg.sum(1).reshape(1,-1)
            sns.clustermap(occ, xticklabels=True, yticklabels=True, cmap = "rocket_r", metric = metric)
    else:
        if norm:
            occ = np.matmul(test_map, seg.T)/seg.sum(1).reshape(1,-1)
            occ = occ/occ[:, norm_col].reshape(-1,1)
            sns.clustermap(occ.T, xticklabels=True, yticklabels=True, cmap = "rocket_r", metric = metric)
        else:
            occ = np.matmul(test_map, seg.T)/seg.sum(1).reshape(1,-1)
            sns.clustermap(occ.T, xticklabels=True, yticklabels=True, cmap = "rocket_r", metric = metric)
    
    return occ


#%%
def prune_DG(DG):
    ls_suc = []
    
    for i in list(DG.nodes):
        ls_suc += [list(DG.successors(i))]
    
    for t1, i in enumerate(DG.nodes):
        for t2, j in enumerate(DG.nodes):
            if j == i: continue
            if sum(np.isin(np.array(ls_suc[t1]+[i]), np.array(ls_suc[t2])))==len(ls_suc[t1])+1:
                for k in ls_suc[t1]:
                    ls_suc[t2].remove(k)
            print(ls_suc)
    DG_prune = nx.DiGraph()           
    for i, j in enumerate(ls_suc):
        if len(j)>0:
            for k in j:
                DG_prune.add_edge(list(DG.nodes)[i],k)
    return DG_prune

#%%
def prune_DG2(DG):
    DG_temp = list(DG.edges())
    source = [node for node in DG if DG.in_degree(node)==0][0]
    for i in DG[source].keys():
        if DG.in_degree(i)>1:
            DG_temp.remove((source, i))
    print(DG_temp)
    xx = nx.DiGraph()
    xx.add_edges_from(DG_temp)
    return xx

#%%
def prune_DG3(DG):
    DG_temp = list(DG.edges())
    for i in DG.nodes():
        for j in DG.successors(i):
            for k in set(DG.successors(i)) & set(DG.successors(j)):
                DG_temp.remove((i,k))
    
    print(DG_temp)
    xx = nx.DiGraph()
    xx.add_edges_from(DG_temp)
    return xx

    

#%% import data
seg = np.load("/Users/shengkai/Desktop/data/PDBs_forAlignment_box160/occ_origin_matrix.npy")
unmasked_map = np.load("/Users/shengkai/Desktop/data/Biopaper_unmaskedmaps/unmasked_map_wo70S.npy")

bin_name = np.load("/Users/shengkai/Desktop/data/PDBs_forAlignment_box160/bin_name.npy")
bin_map = np.load("/Users/shengkai/Desktop/data/Biopaper_unmaskedmaps/bin_map_wo70S.npy")

map_namelist = np.load("/Users/shengkai/Desktop/data/Biopaper_unmaskedmaps/map_namelist_wo70S.npy")
seg_name = [i.split("_")[1].split(".")[0] for i in bin_name]
pattern = "[L]"

df = pd.read_csv("/Users/shengkai/Desktop/data/Biopaper_unmaskedmaps/ummasked_name_class.csv")
pd.DataFrame(index = map_namelist).join(df.set_index("name"))["class"]=="C"

cl = pd.DataFrame(index = map_namelist).join(df.set_index("name"))["class"].values

bin_map_DE = bin_map[np.logical_or(cl=="E", cl=="D")]
unmasked_map_DE = unmasked_map[np.logical_or(cl=="E", cl=="D")]
namelist_DE = map_namelist[np.logical_or(cl=="E", cl=="D")]
cl_DE = cl[np.logical_or(cl=="E", cl=="D")]

bin_map_BC = bin_map[np.logical_or(cl=="B", cl=="C")]
unmasked_map_BC = unmasked_map[np.logical_or(cl=="B", cl=="C")]
namelist_BC = map_namelist[np.logical_or(cl=="B", cl=="C")]
cl_BC = cl[np.logical_or(cl=="B", cl=="C")]

#npy saved at '/Users/shengkai/Desktop/data/unmasked_map_by_class'den
r = np.load("/Users/shengkai/Desktop/data/unmasked_map_by_class/radius.npy")
r60 = r.reshape(-1)<60

unmasked_map_DE_r = unmasked_map_DE[:, r60==1]
unmasked_map_BC_r = unmasked_map_BC[:, r60==1]

seg_BC = np.load("/Users/shengkai/Desktop/data/unmasked_map_by_class/seg_BC.npy")

#%% try segBC pathway analysis

seg_BC_33_name, seg_BC_33 = load_mrc("/Users/shengkai/Desktop/data/unmasked_map_by_class/BC_seg/seg33_for_BC", 160) 


s = sns_occ_cluster(np.delete(unmasked_map_BC,[34], 0), seg_BC_33, transpose = True, norm=True, norm_col=0)

lower_list = np.zeros(32)
upper_list = np.zeros(32)

dp = "/Users/shengkai/Desktop/data/unmasked_map_by_class/BC_seg/seg33_for_BC/threshold"
for i in range(1, s.shape[1]):
    s_temp = s[:,i][s[:,i].argsort()]
    plt.figure()
    plt.plot(s_temp)
    plt.title(seg_BC_33_name[i])
    y_temp = s_temp[3:]-s_temp[:-3]
    plt.plot(y_temp)
    y = y_temp.argmax()
    upper = s_temp[y:].mean() - 1*s_temp[y:].std()
    if y == 0:
        lower = upper/2
    else:
        lower = s_temp[:y].mean() + 1*s_temp[y:].std()
    
    lower_list[i] = lower
    upper_list[i] = upper
    
    plt.hlines([upper], 0 , 68, colors = "red")
    plt.hlines([lower], 0 , 68, colors = "blue")
    plt.savefig(dp + "/" + seg_BC_33_name[i].split(".")[0])
    plt.close()
    

s_upper = s>upper_list
s_lower = s<lower_list

columns = ["i", "j", "0_0", "0_1", "1_0", "1_1"]
    
qua = np.zeros((int(s.shape[1]*(s.shape[1]-1)/2),6))

count = 0
for i in range(s.shape[1]-1):
    for j in range(i+1, s.shape[1]):
        qua[count] = solve_uull(s_upper[:,i],s_upper[:,j],s_lower[:,i],s_lower[:,j], i,j)
        count += 1
        
dependency = qua[np.logical_and(qua[:,3]==0, qua[:,4]!=0)][:,0:2] 
dependency = np.vstack((dependency,qua[np.logical_and(qua[:,4]==0, qua[:,3]!=0)][:,0:2][:,::-1]))
cor = qua[np.logical_and(qua[:,3]==0, qua[:,4]==0)][:,0:2]

#%%

G = to_graph(cor)
print(list(connected_components(G)))
#%%



DG=nx.DiGraph()
DG.add_edges_from(dependency.astype(int))

            

#%%

            

eff_mass = pd.read_csv('/Users/shengkai/Desktop/data/unmasked_map_by_class/BC_seg/seg33_for_BC/eff_mass.csv')

ls_eff_mass = np.delete(eff_mass["1"].values, [1])

DG_prune_weighted = nx.DiGraph()
"""
for i, j in enumerate(ls_suc):
    if len(j)>0:
        for k in j:
            DG_prune_weighted.add_edge(i,k,weight = ls_eff_mass[k])
"""
plt.figure()
nx.draw_kamada_kawai(DG_prune_weighted, with_labels = True)

#%%
"""
DG_prune_del_cor = DG_prune 
for i in range(len(list(connected_components(G)))):
    for j in list(list(connected_components(G)).pop(i))[1:]:
        DG_prune_del_cor.remove_node(j)
        
        
DG_prune_del_cor = DG_prune_weighted
for i in range(len(list(connected_components(G)))):
    for j in list(list(connected_components(G)).pop(i))[1:]:
        DG_prune_del_cor.remove_node(j)
        

ls_suc2 = []
for i in DG_prune_del_cor.nodes:
    ls_suc2 += [list(DG_prune_del_cor.successors(i))]
    
for i in range(20):
    for j in range(20):
        if j == i: continue
        if sum(np.isin(np.array(ls_suc2[i]+[list(DG_prune_del_cor.nodes)[i]]), np.array(ls_suc2[j])))==len(ls_suc2[i])+1:
            for k in ls_suc2[i]:
                ls_suc2[j].remove(k)

DG_prune_weighted = nx.DiGraph()
for i, j in enumerate(ls_suc2):
    if len(j)>0:
        for k in j:
            DG_prune_weighted.add_edge(list(DG_prune_del_cor.nodes)[i],k,weight = np.sqrt(ls_eff_mass[k]))
"""            

#%%

DG=nx.DiGraph()
DG.add_edges_from(dependency.astype(int)+1)

 
for i in range(len(list(connected_components(G)))):
    for j in list(list(connected_components(G)).pop(i))[1:]:
        print(j+1)
        DG.remove_node(j+1)
        
DG.remove_nodes_from([11,23,24,27,28,29,30,31,32])
DG_prune = prune_DG(DG)

#%%
seg_PDB_occ = sns_occ_cluster(seg_BC_33, seg)
contact_occ = seg_PDB_occ>0.01

contact = np.zeros((int(32*31/2),3))
count = 0
for i in range(31):
    for j in range(i+1,32):
        temp_c = np.logical_and(contact_occ[i],contact_occ[j]).sum()
        if temp_c ==0:
            contact[count] = [i,j,0]
            count+=1
            continue
        contact[count] = [i,j,seg_PDB_occ[[i,j]][:, np.logical_and(contact_occ[i],contact_occ[j])].sum()]
        count+=1                      

contact_weight = contact[contact[:,2]!=0]
contact_weight[:,0:2] = contact_weight[:,0:2] +1 
contact_weight[:,2]= np.sqrt(1/contact_weight[:,2])

GG=nx.Graph()

GG.add_weighted_edges_from(contact_weight)
GG.remove_nodes_from([11,23,24,27,28,29,30,31,32])   

plt.figure()
nx.draw_kamada_kawai(GG, with_labels = True)


#%% define merge function

def merge_connected(seg, seg_name, connected_list):
    seg_name = [i.split(".")[0] for i in seg_name]
    connected_list = [list(i) for i in connected_list]
    ind = list(range(seg.shape[0]))
    merge_list = []
    for i in connected_list:
        merge_list += i
    merge_list =  [int(i) for i in merge_list]
    retain_ind = np.delete(np.array(ind), merge_list)
    retain_seg = seg[retain_ind]
    retain_seg_name = np.array(seg_name)[retain_ind]
    
    merge_seg = np.zeros((len(connected_list), 160**3))
    merge_name = []
    for i, m in enumerate(connected_list):
        m = [int(n) for n in m]
        merge_seg[i] = seg[m].sum(0)
        merge_name.append("_m_".join(np.array(seg_name)[m]))
    return (np.vstack((retain_seg, merge_seg)), list(retain_seg_name) + merge_name)

def contact_graph(seg, PDB_seg, threshold, mass):
    seg_occ = sns_occ_cluster(seg, PDB_seg)
    contact_occ = seg_occ>threshold
    contact = np.zeros((int(seg.shape[0]*(seg.shape[0]-1)/2),4))
    count = 0
    for i in range(seg.shape[0]-1):
        for j in range(i+1,seg.shape[0]):
            temp_c = np.logical_and(contact_occ[i],contact_occ[j]).sum()
            if temp_c ==0:
                contact[count] = [i,j,0,0]
                count+=1
                continue
            mass_cal = 1/(1/mass[i] + 1/mass[j])
            contact[count] = [i,j,seg_occ[[i,j]][:, np.logical_and(contact_occ[i],contact_occ[j])].sum(), mass_cal]
            count+=1                      

    contact_weight = contact[contact[:,2]!=0]
    contact_weight[:,0:2] = contact_weight[:,0:2] 
    contact_weight[:,2]= np.sqrt((1/contact_weight[:,2])*contact_weight[:,3])
    return(contact_weight)
    
    
seg_BC_33_name = [i.split("soft_")[-1] for i in seg_BC_33_name]

#%%





s = sns_occ_cluster(np.delete(unmasked_map_BC,[34], 0), new_seg_BC_33, transpose = True, norm=True, norm_col=0)
mass = np.zeros(s.shape[1])
for i in range(s.shape[1]):
    mass[i] = np.median(s[s_upper[:,0], i]) * new_seg_BC_33[i].sum()
a =contact_graph(new_seg_BC_33, seg, 0.01)
GG=nx.Graph()
GG.add_weighted_edges_from(a[:,0:3])
GG.remove_nodes_from([8,9,10,11,12,20])



#%%

def qua_analysis(seg, seg_name, test_map, dp, save_pic = True, sigma_num=1, norm_seg = 0, threshold=0.5, step = 3):
    seg_num = seg.shape[0]
    lower_list = np.zeros(seg_num)
    upper_list = np.zeros(seg_num)
    s = sns_occ_cluster(test_map, seg, transpose = True, norm = True, norm_col = norm_seg)
    for i in range(seg_num):
        s_temp = s[:,i][s[:,i].argsort()]
        
        y_temp = s_temp[step:]-s_temp[:-step]
        
        y = y_temp.argmax()
        
        upper = s_temp[y:].mean() - sigma_num*s_temp[y:].std()
        print(y,s_temp[y:].mean(),  sigma_num*s_temp[y:].std() )
        if y == 0 or y == 1:
            lower = upper/2
        else:
            lower = s_temp[:y].mean() + sigma_num*s_temp[:y].std()
        
        if lower > upper:
            lower = (lower+upper)/2
            upper = lower
        if upper <0.4:
            upper = s_temp.max()/2
            lower = upper/2
        
        if sum(s_temp>threshold) == len(s_temp):
            lower = threshold
            upper = threshold
            
            
            
        lower_list[i] = lower
        upper_list[i] = upper
        
        
        if save_pic:
            plt.figure()
            plt.plot(s_temp)
            plt.title(seg_name[i])
            plt.plot(y_temp)
            plt.hlines([upper], 0 , test_map.shape[0], colors = "red")
            plt.hlines([lower], 0 , test_map.shape[0], colors = "blue")
            plt.savefig(dp + "/" + seg_name[i].split(".")[0])
            plt.close()
        
    s_upper = s>=upper_list
    s_lower = s<lower_list

    columns = ["i", "j", "0_0", "0_1", "1_0", "1_1"]
    
    qua = np.zeros((int(seg_num*(seg_num-1)/2),6))

    count = 0
    for i in range(seg_num-1):
        for j in range(i+1, seg_num):
            qua[count] = solve_uull(s_upper[:,i],s_upper[:,j],s_lower[:,i],s_lower[:,j], i,j)
            count += 1
    dependency = qua[np.logical_and(qua[:,3]==0, qua[:,4]!=0)][:,0:2] 
    dependency = np.vstack((dependency,qua[np.logical_and(qua[:,4]==0, qua[:,3]!=0)][:,0:2][:,::-1]))
    cor = qua[np.logical_and(qua[:,3]==0, qua[:,4]==0)][:,0:2]
    
    mass = np.zeros(seg_num)
    for i in range(seg_num):
        mass[i] = np.max(s[s_upper[:,0], i]) * seg[i].sum()
        
    return (s, upper_list, lower_list, qua, dependency, cor, mass)


#%%

s, upper_list, lower_list, q, d, c, mass = qua_analysis(seg_BC_33, seg_BC_33_name, np.delete(unmasked_map, [111], 0), dp, 1.5)

G = to_graph(c)
G.remove_nodes_from([0, 11,23,24,27,28,29,30,31,32])   
print(list(connected_components(G)))
new_seg_BC_33, new_seg_BC_33_name = merge_connected(seg_BC_33, seg_BC_33_name, list(connected_components(G)))
s, upper_list, lower_list, q, d, c, mass = qua_analysis(new_seg_BC_33, new_seg_BC_33_name, np.delete(unmasked_map, [111], 0), dp, 1.5)

DG=nx.DiGraph()
DG.add_edges_from(d.astype(int))
DG_prune = prune_DG(DG)
plt.figure()
nx.draw_kamada_kawai(DG_prune, with_labels = True)

a =contact_cloud(new_seg_BC_33, (160,160,160), 2)
GG=nx.Graph()
GG.add_weighted_edges_from(a[1][a[1][:,3]!=0][:,0:3])
plt.figure()
nx.draw_kamada_kawai(GG, with_labels = True)

#%%
a =contact_graph(new_seg_BC_33, seg, 0.04, mass)
GG=nx.Graph()
GG.add_weighted_edges_from(a[:,0:3])
GG.remove_nodes_from([7,8,9,10,11,18])
plt.figure()
nx.draw_kamada_kawai(GG, with_labels = True)

#%%

dp = "/Users/shengkai/Desktop/data/unmasked_map_by_class/BC_seg/seg_BC_use_all/threshold/"
s1, upper_list1, lower_list1, q1, d1, c1, mass1 = qua_analysis(new_seg_BC_33, new_seg_BC_33_name, np.delete(unmasked_map, [111], 0), dp)

DG1=nx.DiGraph()
DG1.add_edges_from(d1.astype(int))
DG1.remove_nodes_from([7,8,9,10,11,18])
DG1_prune = prune_DG(DG1)

#%%

dp = "/Users/shengkai/Desktop/data/unmasked_map_by_class/BC_seg/seg_BC_use_all/threshold/"
s2, upper_list2, lower_list2, q2, d2, c2, mass2 = qua_analysis(new_seg_BC_33, new_seg_BC_33_name, np.delete(unmasked_map, [111], 0), dp, 1.5)

DG2=nx.DiGraph()
DG2.add_edges_from(d2.astype(int))
DG2.remove_nodes_from([7,8,9,10,11,18])
DG2_prune = prune_DG(DG2)


#%%

def cloud_seg(seg, box, cloud_size):
    seg_temp = np.zeros(box)
    seg_temp = seg_temp.reshape(-1)
    seg_temp[seg==1] = 1
    seg_temp = seg_temp.reshape(box)
    x, y, z = np.where(seg_temp == 1) 
    for i in range(len(x)):
        seg_temp[x[i]-cloud_size:x[i]+cloud_size+1, y[i]-cloud_size:y[i]+cloud_size+1, z[i]-cloud_size:z[i]+cloud_size+1] = 1
    return seg_temp.reshape(-1)

#%%
def contact_cloud(seg, box, cloud_size, mass, norm = True):
    seg_temp = np.zeros(seg.shape)
    for i in range(seg.shape[0]):
        seg_temp[i] = cloud_seg(seg[i], box, cloud_size)
  
    contact = np.zeros((int(seg.shape[0]*(seg.shape[0]-1)/2),4))
    count = 0
    for i in range(seg.shape[0]-1):
        for j in range(i+1,seg.shape[0]):
            temp_c = np.logical_and(seg_temp[i]==1,seg_temp[j]==1).sum()
            if temp_c ==0:
                contact[count] = [i,j,0,0]
                count+=1
                continue
            if norm == True:
                mass_cal = 1/(1/mass[i] + 1/mass[j])
            else:
                mass_cal = 1
            temp_c = mass_cal/temp_c
            contact[count] = [i,j, temp_c, mass_cal]
            count += 1         

    return seg_temp, contact

#%%


s, upper_list, lower_list, q, d, c, mass = qua_analysis(seg, seg_name, np.delete(unmasked_map, [111], 0), dp, 1.5)

G = to_graph(c)
new_seg, new_seg_name = merge_connected(seg, seg_name, list(connected_components(G)))

new_seg_name_sort = np.array(new_seg_name)[np.array(list(map(len,new_seg_name))).argsort()[::-1]]
new_seg_sort = new_seg[np.array(list(map(len,new_seg_name))).argsort()[::-1]]

s, upper_list, lower_list, q, d, c, mass = qua_analysis(new_seg_sort, new_seg_name_sort, np.delete(unmasked_map, [111], 0), dp, sigma_num=1.5, save_pic=False)

G = to_graph(c)
new_seg_2, new_seg_name_2 = merge_connected(new_seg_sort, new_seg_name_sort, list(connected_components(G)))


s, upper_list, lower_list, q, d, c, mass = qua_analysis(new_seg_2, new_seg_name_2, np.delete(unmasked_map, [111], 0), dp, sigma_num=1.5, save_pic=False)


DG=nx.DiGraph()
DG.add_edges_from(d.astype(int))
DG_prune = prune_DG(DG)
plt.figure()
nx.draw_kamada_kawai(DG_prune, with_labels = True)

a =contact_cloud(new_seg_2, (160,160,160), 2, mass)
GG=nx.Graph()
GG.add_weighted_edges_from(a[1][a[1][:,3]!=0][:,0:3])
plt.figure()
nx.draw_kamada_kawai(GG, with_labels = True)

#%%
def contact_prune_DG(contact_G, DG):
    contact = set(contact_G.edges())
    contact = set.union(contact, (set(map(lambda x: x[::-1], contact_G.edges()))))
    DG_temp = nx.DiGraph()
    for i in DG.edges():
        if i in contact:
            DG_temp.add_edge(i[0],i[1], weight = contact_G.edges[i[0],i[1]]["weight"])
    return prune_DG(DG_temp)
    
    
#%%
BC_seg_name, BC_seg = load_mrc("/Users/shengkai/Desktop/data/unmasked_map_by_class/BC_seg/soft", 160)
BC_seg_HD_name, BC_seg_HD = load_mrc("/Users/shengkai/Desktop/data/unmasked_map_by_class/BC_seg/soft/hidedust", 160)

#%%
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

#%%

seg = BC_seg[1]

#%%
def dust_seg(seg, connected_map, size_threshold=100):
    t_temp = set(np.where(seg==1)[0])
    xx = connected_map.subgraph(t_temp)
    connected_xx = list(connected_components(xx))
    xx_len = np.array(list(map(len, connected_xx)))
    xx_len_t = np.where(xx_len>100)[0]
    print(xx_len_t)
    divided_seg =  np.zeros((len(xx_len_t), len(seg)))
    for i,t in enumerate(xx_len_t):
        divided_seg[i, list(connected_xx[t])] = 1           
    
    return xx_len, divided_seg

def mass_cal(voxel_num, pix_size = 2.62, density = 1.32):
    return (voxel_num*(pix_size**3)/10*density*6.02/1000)
    
#%%
import functools


xx_len_list = list(map(functools.partial(dust_seg, connected_map = G_temp), [BC_seg[i] for i in range(BC_seg.shape[0])]))

#%%
seg_DE_name, seg_DE = load_mrc("/Users/shengkai/Desktop/data/unmasked_map_by_class/DE_seg/seg/hidedust",160)

select = [0,1,2,3,4,5,6,7,8,9,11,12,13,14,15,17,18,19,21,22,24,25,26,27,29,30,32,33,36,38]
seg_DE_name =  np.array(seg_DE_name)[select]
seg_DE = seg_DE[select]
seg_DE_name = np.array(seg_DE_name)[seg_DE.sum(1).argsort()[::-1]]
seg_DE = seg_DE[seg_DE.sum(1).argsort()[::-1]]

dp = "/Users/shengkai/Desktop/data/unmasked_map_by_class/DE_seg/seg/hidedust/qua_analysis/"
s, upper_list, lower_list, q, d, c, mass = qua_analysis(seg_DE, seg_DE_name, np.delete(unmasked_map, [111], 0), dp, 1.5)

G = to_graph(c)
new_seg, new_seg_name = merge_connected(seg_DE, seg_DE_name, list(connected_components(G)))

s, upper_list, lower_list, q, d, c, mass = qua_analysis(new_seg, new_seg_name, np.delete(unmasked_map, [111], 0), dp, sigma_num=1.5, save_pic=False)



a =contact_cloud(new_seg, (160,160,160), 2, mass)
GG=nx.Graph()
GG.add_weighted_edges_from(a[1][a[1][:,3]!=0][:,0:3])
plt.figure()
nx.draw_kamada_kawai(GG, with_labels = True)

DG=nx.DiGraph()
DG.add_edges_from(d.astype(int))

DG_contact_prune = contact_prune_DG(GG, DG)


#%%
mapping = {0: "core", 1: "h42-44, L6,10,11", 2: "CP", 3: "L1 misstalk(EngA)", 4: "h89-93", 5: "h68-71", 6: "L14, 19",
           7: "L1 misstalk left", 8: "h76,77,78", 9: "L9 yjgA", 10: "h75,76p,79", 11:"h34-36", 12: "h38pm", 13: "L17",
           14: "h101, movement", 15: "L16", 16: "h71p", 17: "L32", 18:"h74,75p", 19: "seg6?", 20: "seg5?",
           21: "L34", 22: "L33", 23: "L61-66, L2p", 24: "h47-60,105,107", 25:"L28, L9p, h21p"}

H = nx.relabel_nodes(DG_contact_prune, mapping)






#%%

#%%
seg_BC_name, seg_BC = load_mrc("/Users/shengkai/Desktop/data/unmasked_map_by_class/BC_seg/seg33_for_BC/",160)

#select = [0,1,2,3,4,5,6,7,8,9, 10, 11,12,13,14,15,17,18,19,21,22,24,25,26,27,29,30,32,33,36,38]
#seg_DE_name =  np.array(seg_DE_name)[select]
#seg_DE = seg_DE[select]
#seg_DE_name = np.array(seg_DE_name)[seg_DE.sum(1).argsort()[::-1]]
#seg_DE = seg_DE[seg_DE.sum(1).argsort()[::-1]]

dp = "/Users/shengkai/Desktop/data/unmasked_map_by_class/BC_seg/seg_BC_use_all/qua_analysis/"
s, upper_list, lower_list, q, d, c, mass = qua_analysis(seg_BC, seg_BC_name, np.delete(unmasked_map_BC, [34], 0), dp, 1.5)

G = to_graph(c)
new_seg, new_seg_name = merge_connected(seg_BC, seg_BC_name, list(connected_components(G)))

s, upper_list, lower_list, q, d, c, mass = qua_analysis(new_seg, new_seg_name, np.delete(unmasked_map_BC, [34], 0), dp, sigma_num=1.5, save_pic=False)


a =contact_cloud(new_seg, (160,160,160), 2, mass)
GG=nx.Graph()
GG.add_weighted_edges_from(a[1][a[1][:,3]!=0][:,0:3])
plt.figure()
nx.draw_kamada_kawai(GG, with_labels = True)

DG=nx.DiGraph()
DG.add_edges_from(d.astype(int))

DG_contact_prune = contact_prune_DG(GG, DG)


#%%

DG_temp  = contact_prune_DG(GG,DG.subgraph([0,1,2,3,4,5,6,13,14,15,16]))

DG_temp = DG_contact_prune
mapping = {0: "core", 1: "bL32, h99p", 2: "L14,19", 3: "L17", 4: "bL34, h8p", 5: "h67-70", 6: "h70,71,74,75",
           7: "misCP II", 8: "misCP III", 9: "?", 10: "front nn", 11:"right stalk nn", 12: "h21m h16m", 13: "h34,35,51,52,54-60,105-107, L23,29",
           14: "h61-66, L2,22", 15: "h14,16,18,21,22, L28", 16: "h75-79,88, L9", 17: "misCP I"}

H = nx.relabel_nodes(DG_temp, mapping)



#%%
s, upper_list, lower_list, q, d, c, mass = qua_analysis(seg_BC, seg_BC_name, np.delete(unmasked_map, [111], 0), dp, 1.5)

#%%
G = to_graph(c)
new_seg, new_seg_name = merge_connected(seg_BC, seg_BC_name, list(connected_components(G)))

s, upper_list, lower_list, q, d, c, mass = qua_analysis(new_seg, new_seg_name, np.delete(unmasked_map, [111], 0), dp, sigma_num=1.5, save_pic=False)


a =contact_cloud(new_seg, (160,160,160), 1, mass)
GG=nx.Graph()
GG.add_weighted_edges_from(a[1][a[1][:,3]!=0][:,0:3])
plt.figure()
nx.draw_kamada_kawai(GG, with_labels = True)

DG=nx.DiGraph()
DG.add_edges_from(d.astype(int))
DG.add_edge(1,2)
DG_contact_prune = contact_prune_DG(GG, DG)

for (i,j) in DG_contact_prune.edges():
    DG_contact_prune[i][j]["weight"]= GG[i][j]["weight"]

mapping = {0: "core", 1: "h65", 2: "uL22, h61,h73,h97p", 3: "h34,35", 4: "L32", 5: "L14,19", 6: "L17",
           7: "L28", 8: "h14,16,21,22,88", 9: "L34", 10: "h67-70", 11:"h70,71,74,75p", 12: "h76,77,78", 13: "misCP II",
           14: "misCP III", 15: "?", 16: "rignht NN", 17: "h35,51-60,95,96,105-107,L29,L23", 18: "h61-64,66, L2",
           19: "h75-79", 20: "misCP I"}




DG_temp  = contact_prune_DG(GG,DG.subgraph([0,1,2,3,4,5,6,7,8,9,10,11,12,17,18,19]))

"""for (i,j) in DG_temp.edges():
    DG_temp[i][j]["weight"]= GG[i][j]["weight"]
"""

DG_temp = prune_DG(DG_temp)

x = prune_DG2(DG_temp)

H = nx.relabel_nodes(x, mapping)
plt.figure()
nx.draw_kamada_kawai(H, with_labels=True)
sink_nodes = [node for node in H.nodes if H.out_degree(node) == 0]
source_nodes = [node for node in H.nodes if H.in_degree(node) == 0]

#%%

helix_connectivity = [1,2,3,4,5,6,7,5,8,9,10,11,12,13,14,16,18,19,20,
                      18,16,21,22,14,4,23,3,24,2,25,102, #domain 1
                      26,27,28,29,31,27,32,33,34,35,33,103,32,36,37,38,
                      39,40,41,42,43,44,42,41,45,36,46,26,104, #domain 2
                      47,48,49,105,50,51,52,53,54,55,56,57,58,59,56,106,55,54,
                      51,107,49,60,48, #domain 3
                      61,62,63,64,65,66,67,68,69,70,71, 67,64,61,104, #domain 4
                      72,73,74,75,76,77,78,77,76,79,75,80,81,82,83,84,
                      85,83,86,87,82,88,74,89,90,91,92,91,90,93,73, #domain 5
                      94,95,96,97,94,98,99,100,101,99,1 #domain6
                      ]
S5_connectivity = [108,109,110,109,112,111,112,108]

D1 = [1,2,3,4,5,6,7,5,8,9,10,11,12,13,14,16,18,19,20,
                      18,16,21,22,14,4,23,3,24,2,25,102]
D2 = [ 26,27,28,29,31,27,32,33,34,35,33,103,32,36,37,38,
                      39,40,41,42,43,44,42,41,45,36,46,26,104]
D3 = [47,48,49,105,50,51,52,53,54,55,56,57,58,59,56,106,55,54,
                      51,107,49,60,48]
D4 = [61,62,63,64,65,66,67,68,69,70,71, 67,64,61,104] #domain 4

D5= [72,73,74,75,76,77,78,77,76,79,75,80,81,82,83,84,
                      85,83,86,87,82,88,74,89,90,91,92,91,90,93,73]

D6 = [94,95,96,97,94,98,99,100,101,99,1]

seg_con = contact_cloud(seg, (160,160,160), 1, seg.sum(1), norm=False)


seg_contact = seg_con[1][seg_con[1][:,2]!=0]
#%%
h_con = nx.Graph()

for i in range(len(helix_connectivity)-1):
    h_con.add_edge("h%i"%helix_connectivity[i], "h%i"%helix_connectivity[i+1])

for i in range(len(S5_connectivity)-1):
    h_con.add_edge("h%i"%S5_connectivity[i], "h%i"%S5_connectivity[i+1])


fixed_positions = nx.kamada_kawai_layout(h_con)
fixed_nodes = fixed_positions.keys()

frame = copy.deepcopy(h_con) 

rp_con = nx.Graph()
seg_contact_graph = nx.Graph()
for i,j,k in seg_contact[:,0:3]:
    seg_contact_graph.add_edge(seg_name[int(i)],seg_name[int(j)], weight = k)
    if seg_name[int(i)][0]!="h" or seg_name[int(j)][0]!="h":
        rp_con.add_edge(seg_name[int(i)],seg_name[int(j)], weight = k)
        frame.add_edge(seg_name[int(i)],seg_name[int(j)])
#%%

pos = nx.spring_layout(frame, pos = fixed_positions, fixed = fixed_nodes)


sub_seg_contact = seg_contact_graph.subgraph(["h%i"%i for i in set(helix_connectivity)])
sub_h_con = h_con.subgraph(["h%i"%i for i in set(helix_connectivity)])
sub_rp_con = rp_con.subgraph(rp_con.nodes())
plt.figure()

width = [1/sub_seg_contact[i][j]["weight"]/200 for i,j in sub_seg_contact.edges()]
nx.draw(frame, pos, with_labels = True, edge_color='white', width = width,  node_color="none", bbox=dict(facecolor="skyblue", edgecolor='black',boxstyle='round,pad=0.2'))
nx.draw(sub_seg_contact, pos, with_labels = False, edge_color='grey', width = width,  node_color="none")

width_rp = [1/sub_rp_con[i][j]["weight"]/200 for i,j in sub_rp_con.edges()]
nx.draw(sub_rp_con, pos, with_labels = False,edge_color = "red", node_color="none",width = width_rp )
nx.draw(sub_h_con, pos, with_labels = False,node_shape="s", edge_color = "blue", node_color="none",width = 2)

#%%
h_set = set(["h%i"%i for i in D1+D2]) - {"h35"}
p_set = set(["uL3", "uL4", "bL20", "bL21",  "uL24", "uL29"])
for i in p_set:
    p_set = set.union(p_set, h_set & set(rp_con[i].keys()))
    
sub_frame = frame.subgraph(set.union(h_set,p_set))
sub_seg_contact = seg_contact_graph.subgraph(h_set)
sub_h_con = h_con.subgraph(h_set)
sub_rp_con = rp_con.subgraph(p_set)
plt.figure()

width = [1/sub_seg_contact[i][j]["weight"]/200 for i,j in sub_seg_contact.edges()]
nx.draw(sub_frame, pos, with_labels = True, edge_color='white', width = width,  node_color="none", bbox=dict(facecolor="skyblue", edgecolor='black',boxstyle='round,pad=0.2'))
nx.draw(sub_seg_contact, pos, with_labels = False, edge_color='grey', width = width,  node_color="none")

width_rp = [1/sub_rp_con[i][j]["weight"]/200 for i,j in sub_rp_con.edges()]
nx.draw(sub_rp_con, pos, with_labels = False,edge_color = "red", node_color="none",width = width_rp )
nx.draw(sub_h_con, pos, with_labels = False,node_shape="s", edge_color = "blue", node_color="none",width = 2)

#%%
h_set = set(["h%i"%i for i in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,16,18,19,20,21,22,23,24,25,26,27,28,29,31,32,33,36,37,38,39,40,41,45,46,47,48,49,50,72,73,94,95,96,97,98,99,100,101,102,103,104]])
p_set = set(["uL3", "uL4", "bL20", "bL21",  "uL24", "uL29","uL13"])
for i in p_set:
    p_set = set.union(p_set, h_set & set(rp_con[i].keys()))

sub_frame = frame.subgraph(set.union(h_set,p_set))
sub_seg_contact = seg_contact_graph.subgraph(h_set)
sub_h_con = h_con.subgraph(h_set)
sub_rp_con = rp_con.subgraph(p_set)
plt.figure()


fixed_positions = nx.kamada_kawai_layout(sub_h_con)
fixed_nodes = fixed_positions.keys()
pos = nx.spring_layout(sub_frame, pos = fixed_positions, fixed = fixed_nodes)


width = [1/sub_seg_contact[i][j]["weight"]/200 for i,j in sub_seg_contact.edges()]
nx.draw(sub_frame, pos, with_labels = True, edge_color='white', width = width,  node_color="none", bbox=dict(facecolor="skyblue", edgecolor='black',boxstyle='round,pad=0.2'))
nx.draw(sub_seg_contact, pos, with_labels = False, edge_color='grey', width = width,  node_color="none")

width_rp = [1/sub_rp_con[i][j]["weight"]/200 for i,j in sub_rp_con.edges()]
nx.draw(sub_rp_con, pos, with_labels = False,edge_color = "red", node_color="none",width = width_rp )
nx.draw(sub_h_con, pos, with_labels = False,node_shape="s", edge_color = "blue", node_color="none",width = 2)

#%%

h_set = set(["h%i"%i for i in [51, 52, 54, 55,56, 57,58,59,60,95,96]])
p_set = set(["uL29", "uL23"])
for i in p_set:
    p_set = set.union(p_set, h_set & set(rp_con[i].keys()))

sub_frame = frame.subgraph(set.union(h_set,p_set))
sub_seg_contact = seg_contact_graph.subgraph(h_set)
sub_h_con = h_con.subgraph(h_set)
sub_rp_con = rp_con.subgraph(p_set)
plt.figure()


fixed_positions = nx.kamada_kawai_layout(sub_h_con)
fixed_nodes = fixed_positions.keys()
pos = nx.spring_layout(sub_frame, pos = fixed_positions, fixed = fixed_nodes)


width = [1/sub_seg_contact[i][j]["weight"]/200 for i,j in sub_seg_contact.edges()]
nx.draw(sub_frame, pos, with_labels = True, edge_color='white', width = width,  node_color="none", bbox=dict(facecolor="skyblue", edgecolor='black',boxstyle='round,pad=0.2'))
nx.draw(sub_seg_contact, pos, with_labels = False, edge_color='grey', width = width,  node_color="none")

width_rp = [1/sub_rp_con[i][j]["weight"]/200 for i,j in sub_rp_con.edges()]
nx.draw(sub_rp_con, pos, with_labels = False,edge_color = "red", node_color="none",width = width_rp )
nx.draw(sub_h_con, pos, with_labels = False,node_shape="s", edge_color = "blue", node_color="none",width = 2)

#%%

h_set = set(["h%i"%i for i in [14,16,21,22,88]])
p_set = set([])
for i in p_set:
    p_set = set.union(p_set, h_set & set(rp_con[i].keys()))

sub_frame = frame.subgraph(set.union(h_set,p_set))
sub_seg_contact = seg_contact_graph.subgraph(h_set)
sub_h_con = h_con.subgraph(h_set)
sub_rp_con = rp_con.subgraph(p_set)
plt.figure()


fixed_positions = nx.kamada_kawai_layout(sub_h_con)
fixed_nodes = fixed_positions.keys()
pos = nx.spring_layout(sub_frame, pos = fixed_positions, fixed = fixed_nodes)


width = [1/sub_seg_contact[i][j]["weight"]/200 for i,j in sub_seg_contact.edges()]
nx.draw(sub_frame, pos, with_labels = True, edge_color='white', width = width,  node_color="none", bbox=dict(facecolor="skyblue", edgecolor='black',boxstyle='round,pad=0.2'))
nx.draw(sub_seg_contact, pos, with_labels = False, edge_color='grey', width = width,  node_color="none")

width_rp = [1/sub_rp_con[i][j]["weight"]/200 for i,j in sub_rp_con.edges()]
nx.draw(sub_rp_con, pos, with_labels = False,edge_color = "red", node_color="none",width = width_rp )
nx.draw(sub_h_con, pos, with_labels = False,node_shape="s", edge_color = "blue", node_color="none",width = 2)

#%%
Core -> bL34
Core -> h35p, 51-60, 95, 96, L29,L23 -> h34,35
Core -> h75-79 -> h76,77,78

#%%
h_set = set(["h%i"%i for i in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,16,18,19,20,21,22,23,24,25,26,27,28,29,31,32,33,36,37,38,39,40,41,45,46,47,48,49,50,51,52,53,55,60,72,73,94,95,96,97,98,99,100,101,102,103,104]])
p_set = set(["uL3", "uL4", "bL20", "bL21",  "uL24", "uL29","uL23", "bL34"])

h_set = set.union(h_set, set(["h%i"%i for i in [34,35,51,52,53,54,55,56,57,58,59,60,95,96,76,77,78,75,76,77,78,79,105,107]]))

for i in p_set:
    p_set = set.union(p_set, h_set & set(rp_con[i].keys()))

sub_frame = frame.subgraph(set.union(h_set,p_set))
sub_seg_contact = seg_contact_graph.subgraph(h_set)
sub_h_con = h_con.subgraph(h_set)
sub_rp_con = rp_con.subgraph(p_set)



fixed_positions = nx.kamada_kawai_layout(h_con)
fixed_nodes = fixed_positions.keys()
pos = nx.spring_layout(frame, pos = fixed_positions, fixed = fixed_nodes)
"""
pos["h33"] = np.array([-0.364,0.191])
pos["h34"] = np.array([-0.468,0.237])
pos["h35"] = np.array([-0.433,0.101])
pos["uL23"] = np.array([-0.463,0])
pos["bL34"] = np.array([0.086,0.594])
"""
plt.figure()

width = [1/sub_seg_contact[i][j]["weight"]/200 for i,j in sub_seg_contact.edges()]
nx.draw(sub_frame, pos, with_labels = True, edge_color='white', width = width,  node_color="none", bbox=dict(facecolor="skyblue", edgecolor='black',boxstyle='round,pad=0.2'))
nx.draw(sub_seg_contact, pos, with_labels = False, edge_color='grey', width = width,  node_color="none")

width_rp = [1/sub_rp_con[i][j]["weight"]/200 for i,j in sub_rp_con.edges()]
nx.draw(sub_rp_con, pos, with_labels = False,edge_color = "red", node_color="none",width = width_rp )
nx.draw(sub_h_con, pos, with_labels = False,node_shape="s", edge_color = "blue", node_color="none",width = 2)


#%%

cord_id = np.array(range(160**3))
plot_id = cord_id.reshape(160,160,160)[70:75,70:75,70:75]


#%%

dp = "/Users/shengkai/Desktop/data/PDB_seg_dependency"

s_seg, upper_list_seg, lower_list_seg, q_seg, d_seg, c_seg, mass_seg = qua_analysis(seg, seg_name, np.delete(unmasked_map, [111], 0), dp, 1.5, norm_seg = 35)

contact_seg = contact_cloud(seg, (160,160,160), 1, seg.sum(1))[1]

con_seg = contact_seg[contact_seg[:,2:4].sum(1)!=0][:,0:2]
con_c_seg = [i for i in c_seg if i in con_seg]

G_seg = to_graph(np.array(con_c_seg))

#%%
new_seg, new_seg_name = merge_connected(seg, seg_name, list(connected_components(G_seg)))

#%%
core1 = new_seg_name[-7].split("_m_")
cp1 = new_seg_name[-5].split("_m_") 

new_seg_name[-7] = "core1"
new_seg_name[-5] = "cp1"

dp = "/Users/shengkai/Desktop/data/PDB_seg_dependency/seg2"

s_seg_2, upper_list_seg_2, lower_list_seg_2, q_seg_2, d_seg_2, c_seg_2, mass_seg_2 = qua_analysis(new_seg, new_seg_name, np.delete(unmasked_map, [111], 0), dp, 1.5, norm_seg = -7)

contact_seg_2 = contact_cloud(new_seg, (160,160,160), 1, new_seg.sum(1))[1]
con_seg_2 = contact_seg[contact_seg[:,2:4].sum(1)!=0][:,0:2]
con_c_seg_2 = [i for i in c_seg_2 if i in con_seg_2]

#%%
G_seg_2 = to_graph(np.array(con_c_seg_2))

new_seg_2, new_seg_name_2 = merge_connected(new_seg, new_seg_name, list(connected_components(G_seg_2)))
core2 = core1 + new_seg_name_2[-1].split("_m_")
core2.remove("core1")
new_seg_name_2[-1] = "core2" 

dp = "/Users/shengkai/Desktop/data/PDB_seg_dependency/seg3"
s_seg_3, upper_list_seg_3, lower_list_seg_3, q_seg_3, d_seg_3, c_seg_3, mass_seg_3 = qua_analysis(new_seg_2, new_seg_name_2, np.delete(unmasked_map, [111], 0), dp, 1.5, norm_seg = -1)

#%%
a =contact_cloud(new_seg_2, (160,160,160), 2, mass_seg_2)
GG=nx.Graph()
GG.add_weighted_edges_from(a[1][a[1][:,3]!=0][:,0:3])
plt.figure()
nx.draw_kamada_kawai(GG, with_labels = True)

DG=nx.DiGraph()
DG.add_edges_from(d_seg_3.astype(int))
DG_contact_prune = contact_prune_DG(GG, DG)

mapping = {i:j for i,j in enumerate(new_seg_name_2)}
H = nx.relabel_nodes(DG_contact_prune, mapping)

#%%
h_set = set(set(core2) & h_con.nodes())
p_set = set(core2) -h_set
for i in p_set:
    p_set = set.union(p_set, h_set & set(rp_con[i].keys()))

sub_frame = frame.subgraph(set.union(h_set,p_set))
sub_seg_contact = seg_contact_graph.subgraph(h_set)
sub_h_con = h_con.subgraph(h_set)
sub_rp_con = rp_con.subgraph(p_set)
plt.figure()


fixed_positions = nx.kamada_kawai_layout(sub_h_con)
fixed_nodes = fixed_positions.keys()
pos = nx.spring_layout(sub_frame, pos = fixed_positions, fixed = fixed_nodes)


width = [1/sub_seg_contact[i][j]["weight"]/200 for i,j in sub_seg_contact.edges()]
nx.draw(sub_frame, pos, with_labels = True, edge_color='white', width = width,  node_color="none", bbox=dict(facecolor="skyblue", edgecolor='black',boxstyle='round,pad=0.2'))
nx.draw(sub_seg_contact, pos, with_labels = False, edge_color='grey', width = width,  node_color="none")

width_rp = [1/sub_rp_con[i][j]["weight"]/200 for i,j in sub_rp_con.edges()]
nx.draw(sub_rp_con, pos, with_labels = False,edge_color = "red", node_color="none",width = width_rp )
nx.draw(sub_h_con, pos, with_labels = False,node_shape="s", edge_color = "blue", node_color="none",width = 2)




#%%
x = np.array(seg_name)[s_t.sum(0)>=240]
h_set = set(set(x) & h_con.nodes())
p_set = set(x) -h_set
for i in p_set:
    p_set = set.union(p_set, h_set & set(rp_con[i].keys()))

sub_frame = frame.subgraph(set.union(h_set,p_set))
sub_seg_contact = seg_contact_graph.subgraph(h_set)
sub_h_con = h_con.subgraph(h_set)
sub_rp_con = rp_con.subgraph(p_set)
plt.figure()


fixed_positions = nx.kamada_kawai_layout(sub_h_con)
fixed_nodes = fixed_positions.keys()
pos = nx.spring_layout(sub_frame, pos = fixed_positions, fixed = fixed_nodes)


width = [1/sub_seg_contact[i][j]["weight"]/200 for i,j in sub_seg_contact.edges()]
nx.draw(sub_frame, pos, with_labels = True, edge_color='white', width = width,  node_color="none", bbox=dict(facecolor="skyblue", edgecolor='black',boxstyle='round,pad=0.2'))
nx.draw(sub_seg_contact, pos, with_labels = False, edge_color='grey', width = width,  node_color="none")

width_rp = [1/sub_rp_con[i][j]["weight"]/200 for i,j in sub_rp_con.edges()]
nx.draw(sub_rp_con, pos, with_labels = False,edge_color = "red", node_color="none",width = width_rp )
nx.draw(sub_h_con, pos, with_labels = False,node_shape="s", edge_color = "blue", node_color="none",width = 2)

#%%
np.where(seg[np.where(np.array(seg_name) =="bL32")].reshape((160,160,160))==1)

cord_bL32 = np.array([91, 103, 66])
cord_bL17 = np.array([91, 110, 73])

cord_id = np.array(range(160**3))
plot_id = cord_id.reshape(160,160,160)[85:90:2,100:105:2,65:70:2]
sns.clustermap(unmasked_map[:,plot_id.reshape(-1)], cmap = "rocket_r")

plot_id = cord_id.reshape(160,160,160)[85:90:1,102:107:1,65:70:1]
cube_id = np.zeros(160**3)
cube_id[plot_id.reshape(-1)] =1
occ = np.matmul(seg, cube_id.reshape(-1).T)

print(np.array(seg_name)[occ.argsort()][::-1])
print(np.array((occ)[occ.argsort()][::-1]))

# [34,102,31]
# bL32, bL17, h47


seg_temp = seg[[34,102,31]]
for i, pid in enumerate(plot_id.reshape(-1)):
    print(i, seg_temp[:, pid])
    
#%%

seg_DE_name, seg_DE = load_mrc('/Users/shengkai/Desktop/data/unmasked_map_by_class/DE_seg/picked', 160)
#%%
dp = '/Users/shengkai/Desktop/data/unmasked_map_by_class/DE_seg/picked'
s_DE, upper_list_DE, lower_list_DE, q_DE, d_DE, c_DE, mass_DE = qua_analysis(seg_DE, seg_DE_name, np.delete(unmasked_map, [111], 0), dp, norm_seg = -5, step = 3, sigma_num=1.5)

    
a = contact_cloud(seg_DE, (160,160,160), 1, mass_DE)
GG=nx.Graph()
GG.add_weighted_edges_from(a[1][a[1][:,3]!=0][:,0:3])
plt.figure()
nx.draw_kamada_kawai(GG, with_labels = True)
    
DG=nx.DiGraph()
DG.add_edges_from(d_DE.astype(int))
DG_contact_prune = contact_prune_DG(GG, DG)
    
   
#%%
mapping = {0: "uL10,uL11,h42-44", 1: "h42p", 2: "uL6", 3: "uL14, bL19", 4: "bL34", 5: "h74,h75p", 6: "h75p,h76p,h79,h52pp",
           7: "h71p", 8: "h68p,h70p,h69p,h71pp", 9: "bL33", 10: "h89p,h90-93", 11:"h61p,h63-64", 12: "uL16", 13: "h62p,h64pp",
           14: "h65p,h66,uL2p", 15: "bL32", 16: "h33p,h34,h35p,h103pp", 17: "bL17", 18: "h54pp,h55m,h56-59",
           19: "CP(h11p,h37,h38p,h39p,h81-88,\n5S,uL18,bL27,bL25,uL5)", 20: "h47p,h48p,h49-53,h54p,\nh60p,h105,h107p,\nuL29,uL23p",
           21:"h76m,77m,78m", 22:"core", 23:"bL9", 24:"yjgA", 25:"bL9p(bL28)", 26:"bL28"}

H = nx.relabel_nodes(prune_DG2(DG_contact_prune), mapping)

    
#%%

x = set(core2) - set.union(h_set,p_set)

h_set = set(set(x) & h_con.nodes())
p_set = set(x) -h_set
for i in p_set:
    p_set = set.union(p_set, h_set & set(rp_con[i].keys()))

sub_frame = frame.subgraph(set.union(h_set,p_set))
sub_seg_contact = seg_contact_graph.subgraph(h_set)
sub_h_con = h_con.subgraph(h_set)
sub_rp_con = rp_con.subgraph(p_set)
plt.figure()


fixed_positions = nx.kamada_kawai_layout(sub_h_con)
fixed_nodes = fixed_positions.keys()
pos = nx.spring_layout(sub_frame, pos = fixed_positions, fixed = fixed_nodes)


width = [1/sub_seg_contact[i][j]["weight"]/200 for i,j in sub_seg_contact.edges()]
nx.draw(sub_frame, pos, with_labels = True, edge_color='white', width = width,  node_color="none", bbox=dict(facecolor="skyblue", edgecolor='black',boxstyle='round,pad=0.2'))
nx.draw(sub_seg_contact, pos, with_labels = False, edge_color='grey', width = width,  node_color="none")

width_rp = [1/sub_rp_con[i][j]["weight"]/200 for i,j in sub_rp_con.edges()]
nx.draw(sub_rp_con, pos, with_labels = False,edge_color = "red", node_color="none",width = width_rp )
nx.draw(sub_h_con, pos, with_labels = False,node_shape="s", edge_color = "blue", node_color="none",width = 2)

#%%

def subgraph(H):
    sink_nodes = [node for node in H.nodes if H.out_degree(node) == 0]
    source_nodes = [node for node in H.nodes if H.in_degree(node) == 0]
    sub_list = []
    for source in source_nodes:
        for sink in sink_nodes:
            path = nx.all_simple_paths(H, source, sink)
            nodes = []
            for p in path:
                nodes  = nodes + p
            nodes = set(nodes)
            sub_list += [H.subgraph(nodes)]
    return sub_list
            
def hierarchy_pos(G, root, levels=None, width=1., height=1.):
    '''If there is a cycle that is reachable from root, then this will see infinite recursion.
       G: the graph
       root: the root node
       levels: a dictionary
               key: level number (starting from 0)
               value: number of nodes in this level
       width: horizontal space allocated for drawing
       height: vertical space allocated for drawing'''
    TOTAL = "total"
    CURRENT = "current"
    def make_levels(levels, node=root, currentLevel=0, parent=None):
        """Compute the number of nodes for each level
        """
        if not currentLevel in levels:
            levels[currentLevel] = {TOTAL : 0, CURRENT : 0}
        levels[currentLevel][TOTAL] += 1
        neighbors = G.neighbors(node)
        for neighbor in neighbors:
            if not neighbor == parent:
                levels =  make_levels(levels, neighbor, currentLevel + 1, node)
        return levels

    def make_pos(pos, node=root, currentLevel=0, parent=None, vert_loc=0):
        dx = 1/levels[currentLevel][TOTAL]
        left = dx/2
        pos[node] = ((left + dx*levels[currentLevel][CURRENT])*width, vert_loc)
        levels[currentLevel][CURRENT] += 1
        neighbors = G.neighbors(node)
        for neighbor in neighbors:
            if not neighbor == parent:
                pos = make_pos(pos, neighbor, currentLevel + 1, node, vert_loc-vert_gap)
        return pos
    if levels is None:
        levels = make_levels({})
    else:
        levels = {l:{TOTAL: levels[l], CURRENT:0} for l in levels}
    vert_gap = height / (max([l for l in levels])+1)
    return make_pos({})

def draw_hierachical(H, root="core"):
    pos = hierarchy_pos(H, root)
    plt.figure()
    plt.xlim([-1,2])
    nx.draw(H,pos,with_labels = True, edge_color='black', width = 1,  node_color="none", bbox=dict(facecolor="skyblue", edgecolor='black',boxstyle='round,pad=0.2'))

#%%
s_H = subgraph(prune_DG3(H))

pos = hierarchy_pos(prune_DG3(H), "core")


#%%
DE_name, DE = load_mrc('/Users/shengkai/Desktop/data/unmasked_map_by_class/DE_seg/seg/hidedust/violin_by_lim/test_seg',160)

#%%
temp = sns_occ_cluster(np.delete(unmasked_map, [111], 0), DE, norm =True, norm_col=36)

lim = [i[0:3] for i in map_namelist]
lim = np.delete(np.array(lim), [111])

#%%
data = pd.DataFrame({"lim":lim})
for i in range(temp.shape[1]):
    data["occ%02d"%i] = temp[:,i]

#%%

for i,n in enumerate(DE_name):
    plt.figure()
    sns.violinplot(data = data, x = "lim", y = "occ%02d"%i, inner="points")
    temp_n = n.split(".")[0]
    plt.title('%02d_%s.png'%(i,temp_n))
    plt.savefig('/Users/shengkai/Desktop/data/unmasked_map_by_class/DE_seg/seg/hidedust/violin_by_lim/png/%02d_%s.png'%(i,temp_n))
    plt.close()






#%%

def qua_analysis(seg, seg_name, test_map, dp, save_pic = True, sigma_num=1, norm_seg = 0, threshold=0.5, step = 3):
    seg_num = seg.shape[0]
    lower_list = np.zeros(seg_num)
    upper_list = np.zeros(seg_num)
    s = sns_occ_cluster(test_map, seg, transpose = True, norm = True, norm_col = norm_seg)
    for i in range(seg_num):
        s_temp = s[:,i][s[:,i].argsort()]
        
        y_temp = s_temp[step:]-s_temp[:-step]
        
        y = y_temp.argmax()
        
        upper = s_temp[y:].mean() - sigma_num*s_temp[y:].std()
        print(y,s_temp[y:].mean(),  sigma_num*s_temp[y:].std() )
        if y == 0 or y == 1:
            lower = upper/2
        else:
            lower = s_temp[:y].mean() + sigma_num*s_temp[:y].std()
        
        if lower > upper:
            lower = (lower+upper)/2
            upper = lower
        if upper <0.4:
            upper = s_temp.max()/2
            lower = upper/2
        
        if sum(s_temp>threshold) == len(s_temp):
            lower = threshold
            upper = threshold
            
            
            
        lower_list[i] = lower
        upper_list[i] = upper
        
        
        if save_pic:
            plt.figure()
            plt.plot(s_temp)
            plt.title(seg_name[i])
            plt.plot(y_temp)
            plt.hlines([upper], 0 , test_map.shape[0], colors = "red")
            plt.hlines([lower], 0 , test_map.shape[0], colors = "blue")
            plt.savefig(dp + "/" + seg_name[i].split(".")[0])
            plt.close()
        
    s_upper = s>=upper_list
    s_lower = s<lower_list

    columns = ["i", "j", "0_0", "0_1", "1_0", "1_1"]
    
    qua = np.zeros((int(seg_num*(seg_num-1)/2),6))

    count = 0
    for i in range(seg_num-1):
        for j in range(i+1, seg_num):
            qua[count] = solve_uull(s_upper[:,i],s_upper[:,j],s_lower[:,i],s_lower[:,j], i,j)
            count += 1
    dependency = qua[np.logical_and(qua[:,3]<=1, qua[:,4]>1)][:,0:2] 
    dependency = np.vstack((dependency,qua[np.logical_and(qua[:,4]<=1, qua[:,3]>1)][:,0:2][:,::-1]))
    cor = qua[np.logical_and(qua[:,3]<=1, qua[:,4]<=1)][:,0:2]
    
    mass = np.zeros(seg_num)
    for i in range(seg_num):
        mass[i] = np.max(s[s_upper[:,0], i]) * seg[i].sum()
        
    return (s, upper_list, lower_list, qua, dependency, cor, mass)


#%%
s = s_DE
soi = [8,3,10,7]
tr = upper_list_DE
def mrc_order(s, tr, soi):

    bi_s = s[:, soi] > tr[soi]
    bi_s = bi_s.sum(1)
    bi_order = bi_s.argsort(0)
    bi_s = bi_s[bi_s.argsort(0)]
    state_number = np.unique(bi_s)
    #order = bi_order[bi_s == bi_s.min()]
    order = []
    for i in state_number:

        order += list(bi_order[bi_s == i][s[bi_order[bi_s == i]][:, soi].sum(1).argsort(0)]) 
    return bi_order, order
#%%

#%%

        
        