#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 15:35:26 2022

@author: shengkai
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 19 17:35:21 2022

@author: shengkai
"""

import numpy as np
import mrcfile
import pandas as pd
import seaborn as sns
import networkx as nx
import pickle
from networkx.algorithms.components.connected import connected_components
from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage
import os
import umap
import hdbscan

import copy
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
G_temp = pickle.load(open("./G_temp", "rb"))

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
            matrix[j,i] = -diff[1]
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
            matrix[j,i] = -diff[1]
    return(matrix)

#%%

def mass_diff_matrix_add(matrix):
    temp_matrix = np.zeros(matrix.shape)
    for i in range(matrix.shape[0]-1):
        for j in range(i+1,matrix.shape[0]):
            temp_matrix[i,j] = np.abs(matrix[i,j])+np.abs(matrix[j,i])
            temp_matrix[j,i] = temp_matrix[i,j]
    return 

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

#%% function hdb and sub_divide
def hdb(data,dp, min_cluster_size_range= [50, 100, 150, 200],min_samples_range= [50, 100, 150, 200], fn = "hdbscan"):
    os.system("mkdir " + dp + "/" + fn)
    for min_cluster_size in min_cluster_size_range:
        for min_samples in min_samples_range:
            hdb_test=hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, prediction_data=True).fit(data)
            print("hdbscan, min_cluster_size: %i, min_samples: %i"%(min_cluster_size,min_samples))
            
            pickle.dump(hdb_test, open(dp + "/%s/hdbscan_%i_%i"%(fn, min_cluster_size,min_samples),"wb"))
            cluster_num=len(np.unique(hdb_test.labels_))
            name= "hdbscan, min_cluster_size: %i, min_samples: %i, cluster_num%i"%(min_cluster_size,min_samples,cluster_num)
            plt.figure()
            plt.title(name)
            plt.scatter(data[:,0],data[:,1],c="grey",s=1,alpha=0.2)
            plt.scatter(data[:,0][hdb_test.labels_!=-1],data[:,1][hdb_test.labels_!=-1], c=hdb_test.labels_[hdb_test.labels_!=-1], cmap="rainbow", s=1)
        
            fig_name="%i_%i"%(min_cluster_size,min_samples)
            plt.savefig(dp + "/%s/"%fn + fig_name)
            plt.close()
            
    return

def sub_divide(test_map, seg, fn, output_dp, pca = True, u_metric = "canberra", n_neighbors = 15):
   # os.system("cd %s"%output_dp)
   # temp_map = test_map[:, mask]
    if seg.reshape((-1, 160**3)).shape[0]!=1:
        seg = seg.sum(0)
    temp_map = test_map[:, seg==1]
    
    if pca:
        p_sub = PCA().fit_transform(temp_map.T)    
        u_sub = umap.UMAP(verbose = True, metric = u_metric, n_neighbors = n_neighbors).fit_transform(p_sub[:,0:6])
    else:
        u_sub = umap.UMAP(verbose = True, metric = u_metric, n_neighbors = n_neighbors).fit_transform(temp_map.T)
    
    
    
    np.save(output_dp+"/u_"+fn+".npy", u_sub)
    plt.figure()
    plt.scatter(u_sub[:,0], u_sub[:,1], s =1)
    plt.savefig(output_dp+"/umap_" + fn + ".png")
    plt.close()
    
 #   os.system("mkdir %s/hdbscan"%output_dp)
    
    hdb(u_sub, output_dp, [10, 20, 30, 40, 60, 100, 200], [10, 20, 30, 50, 80, 150, 200] )
    return


#%% segmentation_based_on_hdb_wonoise

def segmentation_based_on_hdb_wonoise(data, hdb_model, box, tag, mask, dp, fn = "seg"):
    if mask.reshape((-1, 160**3)).shape[0]!=1:
        mask = mask.sum(0)
#    hdb_pred = hdbscan.all_points_membership_vectors(hdb_model)
    seg_argmax=hdb_model.labels_
    name_list = ["%s_sub%02i"%(tag, i) for i in range(0,np.max(seg_argmax)+1)]
    os.system("mkdir "+ dp + "/%s"%fn)
    for i in range(0,np.max(seg_argmax)+1):
        name = dp +"/%s/soft_%s"%(fn, name_list[i])
        print(name)
        fig=plt.figure()
        plt.scatter(data[:,0],data[:,1],c="grey",s=1,alpha=0.2)
        plt.scatter(data[seg_argmax==i,0],data[seg_argmax==i,1], c="red", s=1)
        fig.savefig(name+".png")
        plt.close(fig)
        
        seg=np.zeros(len(mask))
        seg[mask==1] = seg_argmax==i
    
        
        save_density(seg.reshape(box, box, box),(apix, apix, apix),name+".mrc")   
#%% import data

seg = np.load("/Users/shengkai/Desktop/data/PDBs_forAlignment_box160/occ_origin_matrix.npy")
unmasked_map = np.load("/Users/shengkai/Desktop/data/Biopaper_unmaskedmaps/unmasked_map_wo70S.npy")

bin_name = np.load("/Users/shengkai/Desktop/data/PDBs_forAlignment_box160/bin_name.npy")
bin_map = np.load("/Users/shengkai/Desktop/data/Biopaper_unmaskedmaps/bin_map_wo70S.npy")

map_namelist = np.load("/Users/shengkai/Desktop/data/Biopaper_unmaskedmaps/map_namelist_wo70S.npy")
seg_name = [i.split("_")[1].split(".")[0] for i in bin_name]
pattern = "[L]"

#%%
def draw_mask_dendro(dist_mat, label, mask, threshold = 60):
    plt.figure()
    mask_temp = np.array(mask)
    dend_temp = dist_mat[mask_temp][:,mask_temp]
    label_temp = np.array(label)[mask_temp]
    linkage_matrix = linkage(np.array([dend_temp[i,j] for i in range(dend_temp.shape[0]-1) for j in range(i+1,dend_temp.shape[0])]), "average")
    d = dendrogram(linkage_matrix, color_threshold=threshold, labels=label_temp,show_leaf_counts=True,leaf_font_size=12)
    return linkage_matrix, d, label_temp

#%%
def extract_jidMask(jid, id_extract):
    jid = np.array([i.split("_J")[-1] for i in label])
    m = [np.where(jid == str(i))[0] for i in id_extract]
    M = []
    for i in m:
        if len(i) != 0:
            M = M + [i[0]]
    return M
#%%
    

dp = '/Users/shengkai/Desktop/data/csda_srmb_cs/segmentation/seg'
seg_cs_name, seg_cs = load_mrc(dp, 160)
occ = sns_occ_cluster(seg_cs, seg)


#%%
l = (occ>0.25).sum()
pdb_seg_name = list(np.zeros(l))
cs_seg_name = list(np.zeros(l))
occ_seg = list(np.zeros(l))
count = 0
for i in range(occ.shape[0]):
    for j in range(occ.shape[1]):
        if occ[i,j]>0.25:
            pdb_seg_name[count] = seg_name[j]
            cs_seg_name[count] = seg_cs_name[i].split(".")[0]
            occ_seg[count] = occ[i,j]
            count += 1
            
df = pd.DataFrame({"pdb": pdb_seg_name, "cs_seg": cs_seg_name, "occ": occ_seg})
df.to_csv("csda_srmb_seg_r1_occ_0p25.csv")


#%% segment refinement?
dp = "/Users/shengkai/Desktop/script/segmentation_dev/"

u = np.load(dp + "useg_canberra_NN100.npy")
hdb(u, "/Users/shengkai/Desktop/script/segmentation_dev/", range(200,1000,100), [1], "p_seg_umap_canberra_ncom2_NN100")

hdb_test = pickle.load(open("/Users/shengkai/Desktop/script/segmentation_dev/p_seg_umap_canberra_ncom12_NN100/hdbscan_200_1", "rb"))

mask = mrcfile.open("mask_box160_percentile_98.0.mrc").data.reshape(-1)



csda_name, csda = load_mrc("/Users/shengkai/Desktop/paper/deaD/merge_rf/picked/", 160)
csda_bi = csda>1
csda_bi = csda_bi*1
seg = np.load("/Users/shengkai/Desktop/data/PDBs_forAlignment_box160/occ_origin_matrix.npy")
bin_name = np.load("/Users/shengkai/Desktop/data/PDBs_forAlignment_box160/bin_name.npy")
seg_name = [i.split("_")[1].split(".")[0] for i in bin_name]
sns_occ_cluster(csda, seg)

occ = np.matmul(csda_bi, seg[[77,119,120, 21,0,22,134,42,111,81,92,105,40]].T)/seg[[77,119,120, 21,0,22,134,42,111,81,92,105,40]].sum(1)
sns.clustermap(occ[csda_bi.sum(1).argsort()], row_cluster = False, cmap = "Blues")

#%%
u = np.load('/Users/shengkai/Desktop/paper/deaD/merge_rf/picked/seg/p21_seg_umap_canberra_ncom2_NN100_perc99.00_cl0_r60_cn1.npy')
mask = mrcfile.open('/Users/shengkai/Desktop/paper/deaD/merge_rf/picked/seg/mask_box160_percentile_99.0.mrc').data.reshape(-1)

#%%

hdb(u, "//Users/shengkai/Desktop/paper/deaD/merge_rf/picked/seg/", [100], range(25,200,25), "p21_seg_umap_canberra_ncom2_NN100_perc99.00_cl0_r60_cn1")
hdb_test = pickle.load(open('/Users/shengkai/Desktop/paper/deaD/merge_rf/picked/seg/p21_seg_umap_canberra_ncom2_NN100_perc99.00_cl0_r60_cn1/hdbscan_100_100', "rb"))
segmentation_based_on_hdb_wonoise(u, hdb_test, 160, "seg_100_100", mask, '/Users/shengkai/Desktop/paper/deaD/merge_rf/picked/seg/', fn = "seg_100_100")


#%%
seg_r1_name, seg_r1 = load_mrc("/Users/shengkai/Desktop/paper/deaD/merge_rf/picked/seg//seg_100_100/", 160)

occ = np.matmul(csda, seg_r1[[0, 1, 2, 3, 5, 6, 7, 8]].T)/seg_r1[[0, 1, 2, 3, 5, 6, 7, 8]].sum(1)
sns.clustermap(occ[csda_bi.sum(1).argsort()], row_cluster = False, cmap = "Blues")
#%%

dp =  '/Users/shengkai/Desktop/paper/deaD/merge_rf/picked/seg/seg_100_100/round2/'
for i in [0, 1, 2, 3, 5, 6, 7, 8]:
    fn = "seg_r2"
    if np.array([i]).T.shape[0]>1:
        for j in i:
            fn = fn + "_%02d"%j
    else:
        fn = fn + "_%02d"%i
    
    os.system("mkdir %s"%fn)
    sub_divide(csda, seg_r1[i], fn, dp + fn, False)

#%%

dp =  '/Users/shengkai/Desktop/paper/deaD/merge_rf/picked/seg/seg_100_100/round2/'
for i in [0, 1, 2, 3, 5, 6, 7, 8]:
    fn = "seg_r2"
    if np.array([i]).T.shape[0]>1:
        for j in i:
            fn = fn + "_%02d_pca"%j
    else:
        fn = fn + "_%02d_pca"%i
    
    os.system("mkdir %s"%fn)
    sub_divide(csda, seg_r1[i], fn, dp + fn, True)
    
#%%
dp =  '/Users/shengkai/Desktop/paper/deaD/merge_rf/picked/seg/seg_100_100/round2/'
for i in [10]:
    fn = "seg_r2"
    if np.array([i]).T.shape[0]>1:
        for j in i:
            fn = fn + "_%02d"%j
    else:
        fn = fn + "_%02d"%i
    
    os.system("mkdir %s"%fn)
    sub_divide(csda, seg_r1[i], fn, dp + fn, False)

#%% seg00_r2
dp = "/Users/shengkai/Desktop/paper/deaD/merge_rf/picked/seg/seg_100_100/round2/"
num = 0
data = np.load(dp + "seg_r2_%02d/u_seg_r2_%02d.npy"%(num, num))
tag = "seg%02d"%num
a = 40
b = 80
hdb_model = pickle.load(open(dp + "/seg_r2_%02d/hdbscan/hdbscan_%i_%i"%(num, a, b),"rb"))
segmentation_based_on_hdb_wonoise(data, hdb_model, 160, tag, seg_r1[num], dp+"/seg_r2_%02d"%num)
#%%
_, sub = load_mrc(dp+"/seg_r2_%02d/seg/"%num, 160)
sns_occ_cluster(csda, np.vstack((sub,seg_r1[num])))
#%%
# combine 3,4
dp = "/Users/shengkai/Desktop/paper/deaD/merge_rf/picked/seg/seg_100_100/round3/"

seg_temp = sub[3] + sub[4]
fn = "seg_r3_%02d"%num
os.system("mkdir %s"%fn)
sub_divide(csda, seg_temp, fn, dp + fn, False)

#%% seg00_r2
dp = "/Users/shengkai/Desktop/paper/deaD/merge_rf/picked/seg/seg_100_100/round3/"
num = 0
data = np.load(dp + "seg_r3_%02d/u_seg_r3_%02d.npy"%(num, num))
tag = "seg%02d"%num
a = 100
b = 200
hdb_model = pickle.load(open(dp + "/seg_r3_%02d/hdbscan/hdbscan_%i_%i"%(num, a, b),"rb"))
segmentation_based_on_hdb_wonoise(data, hdb_model, 160, tag, seg_temp, dp+"/seg_r3_%02d"%num)

save_density(seg_temp.reshape((160,160,160)), (2.62,2.62,2.62), "soft_seg_100_100_sub00_3_4.mrc")
#%% seg01_r2
dp = "/Users/shengkai/Desktop/paper/deaD/merge_rf/picked/seg/seg_100_100/round2/"
num = 1
data = np.load(dp + "seg_r2_%02d/u_seg_r2_%02d.npy"%(num, num))
tag = "seg%02d"%num
a = 20
b = 50
hdb_model = pickle.load(open(dp + "/seg_r2_%02d/hdbscan/hdbscan_%i_%i"%(num, a, b),"rb"))
segmentation_based_on_hdb_wonoise(data, hdb_model, 160, tag, seg_r1[num], dp+"/seg_r2_%02d"%num)
#%%
_, sub = load_mrc(dp+"/seg_r2_%02d/seg/"%num, 160)
sns_occ_cluster(csda, np.vstack((sub,seg_r1[num])))

#%% seg02_r2
dp = "/Users/shengkai/Desktop/paper/deaD/merge_rf/picked/seg/seg_100_100/round2/"
num = 2
data = np.load(dp + "seg_r2_%02d/u_seg_r2_%02d.npy"%(num, num))
tag = "seg%02d"%num
a = 10
b = 10
hdb_model = pickle.load(open(dp + "/seg_r2_%02d/hdbscan/hdbscan_%i_%i"%(num, a, b),"rb"))
segmentation_based_on_hdb_wonoise(data, hdb_model, 160, tag, seg_r1[num], dp+"/seg_r2_%02d"%num)
#%%
num=2
_, sub = load_mrc(dp+"/seg_r2_%02d/seg/"%num, 160)
sns_occ_cluster(csda, np.vstack((sub,seg_r1[num])))

#segment!
#%% seg02_r3

dp = "/Users/shengkai/Desktop/paper/deaD/merge_rf/picked/seg/seg_100_100/round3/"
num = 2
seg_temp = sub[1]
fn = "seg_r3_%02d"%num
os.system("mkdir %s"%fn)
sub_divide(csda, seg_temp, fn, dp + fn, False)

#%%
data = np.load(dp + "seg_r3_%02d/u_seg_r3_%02d.npy"%(num, num))
seg_temp = sub[1]
tag = "seg%02d"%num
a = 10
b = 200
hdb_model = pickle.load(open(dp + "/seg_r3_%02d/hdbscan/hdbscan_%i_%i"%(num, a, b),"rb"))
segmentation_based_on_hdb_wonoise(data, hdb_model, 160, tag, seg_temp, dp+"/seg_r3_%02d"%num)

# not segment
#%% seg01_r2
dp = "/Users/shengkai/Desktop/paper/deaD/merge_rf/picked/seg/seg_100_100/round2/"
num = 5
data = np.load(dp + "seg_r2_%02d/u_seg_r2_%02d.npy"%(num, num))
tag = "seg%02d"%num
a = 10
b = 50
hdb_model = pickle.load(open(dp + "/seg_r2_%02d/hdbscan/hdbscan_%i_%i"%(num, a, b),"rb"))
segmentation_based_on_hdb_wonoise(data, hdb_model, 160, tag, seg_r1[num], dp+"/seg_r2_%02d"%num)
#%%
_, sub = load_mrc(dp+"/seg_r2_%02d/seg/"%num, 160)
sns_occ_cluster(csda, np.vstack((sub,seg_r1[num])))

#%% seg01_r2
dp = "/Users/shengkai/Desktop/paper/deaD/merge_rf/picked/seg/seg_100_100/round2/"
num = 6
data = np.load(dp + "seg_r2_%02d/u_seg_r2_%02d.npy"%(num, num))
tag = "seg%02d"%num
a = 20
b = 20
hdb_model = pickle.load(open(dp + "/seg_r2_%02d/hdbscan/hdbscan_%i_%i"%(num, a, b),"rb"))
segmentation_based_on_hdb_wonoise(data, hdb_model, 160, tag, seg_r1[num], dp+"/seg_r2_%02d"%num)
#%%
_, sub = load_mrc(dp+"/seg_r2_%02d/seg/"%num, 160)
sns_occ_cluster(csda, np.vstack((sub,seg_r1[num])))
#%%
dp = "/Users/shengkai/Desktop/paper/deaD/merge_rf/picked/seg/seg_100_100/round2/"
num = 7
data = np.load(dp + "seg_r2_%02d/u_seg_r2_%02d.npy"%(num, num))
tag = "seg%02d"%num
a = 20
b = 10
hdb_model = pickle.load(open(dp + "/seg_r2_%02d/hdbscan/hdbscan_%i_%i"%(num, a, b),"rb"))
segmentation_based_on_hdb_wonoise(data, hdb_model, 160, tag, seg_r1[num], dp+"/seg_r2_%02d"%num)

#%%
dp = "/Users/shengkai/Desktop/paper/deaD/merge_rf/picked/seg/seg_100_100/round2/"
num = 10
data = np.load(dp + "seg_r2_%02d/u_seg_r2_%02d.npy"%(num, num))
tag = "seg%02d"%num
a = 200
b = 80
hdb_model = pickle.load(open(dp + "/seg_r2_%02d/hdbscan/hdbscan_%i_%i"%(num, a, b),"rb"))
segmentation_based_on_hdb_wonoise(data, hdb_model, 160, tag, seg_r1[num], dp+"/seg_r2_%02d"%num)
#%%
dp = "/Users/shengkai/Desktop/paper/deaD/merge_rf/picked/seg/seg_100_100/round2/"
_, sub = load_mrc(dp+"/seg_r2_%02d/seg/"%num, 160)

dp = "/Users/shengkai/Desktop/paper/deaD/merge_rf/picked/seg/seg_100_100/round3/"
num = 10
seg_temp = sub[1]
fn = "seg_r3_%02d_sub01"%num
os.system("mkdir %s"%fn)
sub_divide(csda, seg_temp, fn, dp + fn, False)

#%%
dp = "/Users/shengkai/Desktop/paper/deaD/merge_rf/picked/seg/seg_100_100/round3/"
num = 10
data = np.load(dp + "seg_r3_%02d_sub01/u_seg_r3_%02d_sub01.npy"%(num, num))
tag = "seg%02d"%num
a = 200
b = 50
hdb_model = pickle.load(open(dp + "/seg_r3_%02d_sub01/hdbscan/hdbscan_%i_%i"%(num, a, b),"rb"))
segmentation_based_on_hdb_wonoise(data, hdb_model, 160, tag, seg_temp, dp+"/seg_r3_%02d_sub01"%num)
#%%
_, sub = load_mrc(dp+"/seg_r2_%02d/seg/"%num, 160)
sns_occ_cluster(csda, np.vstack((sub,seg_r1[num])))

#%%
dp = "/Users/shengkai/Desktop/paper/deaD/merge_rf/picked/seg/seg_100_100/round2/"
_, sub = load_mrc(dp+"/seg_r2_%02d/seg/"%num, 160)

dp = "/Users/shengkai/Desktop/paper/deaD/merge_rf/picked/seg/seg_100_100/round3/"
num = 10
seg_temp = sub[2]
fn = "seg_r3_%02d_sub02"%num
os.system("mkdir %s"%fn)
sub_divide(csda, seg_temp, fn, dp + fn, False)

#%%
dp = "/Users/shengkai/Desktop/paper/deaD/merge_rf/picked/seg/seg_100_100/round3/"
num = 10
data = np.load(dp + "seg_r3_%02d_sub02/u_seg_r3_%02d_sub02.npy"%(num, num))
tag = "seg%02d"%num
a = 30
b = 50
hdb_model = pickle.load(open(dp + "/seg_r3_%02d_sub02/hdbscan/hdbscan_%i_%i"%(num, a, b),"rb"))
segmentation_based_on_hdb_wonoise(data, hdb_model, 160, tag, seg_temp, dp+"/seg_r3_%02d_sub02"%num)
#%%
_, sub = load_mrc(dp+"/seg_r2_%02d/seg/"%num, 160)
sns_occ_cluster(csda, np.vstack((sub,seg_r1[num])))
"""
#%%
dp = "/Users/shengkai/Desktop/paper/deaD/merge_rf/picked/seg/seg_100_100/round2/"
num = 7
data = np.load(dp + "seg_r2_%02d_pca/u_seg_r2_%02d_pca.npy"%(num, num))
tag = "seg%02d_pca"%num
a = 10
b = 200
hdb_model = pickle.load(open(dp + "/seg_r2_%02d_pca/hdbscan/hdbscan_%i_%i"%(num, a, b),"rb"))
segmentation_based_on_hdb_wonoise(data, hdb_model, 160, tag, seg_r1[num], dp+"/seg_r2_%02d_pca"%num)
#%%
_, sub = load_mrc(dp+"/seg_r2_%02d_pca/seg/"%num, 160)
sns_occ_cluster(csda, np.vstack((sub,seg_r1[num])))
"""

#%%
seg_csda_HD_name, seg_csda_HD = load_mrc('/Users/shengkai/Desktop/paper/deaD/merge_rf/picked/seg/passthrough_seg/hidedust/', 160) 

#%%
occ = np.matmul(csda_bi, seg_csda_HD.T)/seg_csda_HD.sum(1)
#occ = occ/occ[:,4].reshape([-1,1])
#occ[occ>1] = 1
sns.clustermap(occ[csda_bi.sum(1).argsort()], row_cluster = False, cmap = "Blues")

#%%
def seg_subsetvoxel(test_map, seg, fraction, tag = "seg"):
    index = np.where(seg == 1)[0]
    index_subset = np.random.choice(index, int(len(index) * fraction), replace = False)
    return test_map[:, index_subset], [tag]*len(index_subset)

#%%

def prepare_umap_with_label_train(test_map, segs, fraction, tag = "seg"):
    labels = segs.shape[0]
    train_size = np.sum(np.array([int(i) for i in segs.sum(1)*fraction]))
    train_voxel = np.zeros([test_map.shape[0], train_size])
    train_label = [""]*train_size
    pos = 0
    for i in range(labels):
        voxel_subset_temp, tag_temp = seg_subsetvoxel(test_map, segs[i], fraction, i)
        train_voxel[:, pos : (pos + len(tag_temp))] = voxel_subset_temp
        train_label[pos : (pos + len(tag_temp))] = tag_temp
        pos = pos + len(tag_temp)
    return train_voxel, train_label

a,b = prepare_umap_with_label_train(csda, seg_csda_HD, 0.1)
c = umap.UMAP(n_neighbors=100, n_components=2, verbose = True).fit(a.T, b)
c1 = c.transform(csda[:,mask ==1].T)

_, l = prepare_umap_with_label_train(csda, seg_csda_HD, 1)

plt.scatter(*c1[np.array(seg_csda_HD.sum(0)==1)[mask ==1],0:2].T, s = 1, c = l, cmap = "rainbow")

#%%
a1,b1 = prepare_umap_with_label_train(csda, np.vstack([seg_csda_HD, sub[2]]), 0.3)
d = umap.UMAP(n_neighbors=100, n_components=2, verbose = True).fit(a1.T, b1)
d1 = d.transform(csda[:,mask ==1].T)
hdb(d1, "//Users/shengkai/Desktop/paper/deaD/merge_rf/picked/seg/", fn =  "p21_seg_umap_canberra_ncom2_NN100_perc99.00_cl0_r60_cn1_with_label")
plt.scatter(*d1[np.array(seg_csda_HD.sum(0)==1)[mask ==1],0:2].T, s = 1, c = l, cmap = "rainbow")
#%%
dp = "Users/shengkai/Desktop/paper/deaD/merge_rf/picked/seg/p21_seg_umap_canberra_ncom2_NN100_perc99.00_cl0_r60_cn1_with_label"
num = 0
data = d1
tag = "seg%02d_wlabel"%num
a = 200
b = 150
hdb_model = pickle.load(open('/Users/shengkai/Desktop/paper/deaD/merge_rf/picked/seg/p21_seg_umap_canberra_ncom2_NN100_perc99.00_cl0_r60_cn1_with_label/hdbscan_200_150',"rb"))
segmentation_based_on_hdb_wonoise(data, hdb_model, 160, tag, mask, dp='/Users/shengkai/Desktop/paper/deaD/merge_rf/picked/seg/seg_wlabel/', fn="seg")



#%%
seg_wlabel_name, seg_wlabel = load_mrc('/Users/shengkai/Desktop/paper/deaD/merge_rf/picked/seg/seg_wlabel/seg/hidedust/', 160)
noise = mrcfile.open('/Users/shengkai/Desktop/paper/deaD/merge_rf/picked/seg/seg_wlabel/seg/soft_seg00_wlabel_sub05.mrc').data.reshape(-1)
a2,b2 = prepare_umap_with_label_train(csda, np.vstack([seg_wlabel, noise]), 0.3)

e = umap.UMAP(n_neighbors=100, n_components=2, verbose = True).fit(a2.T, b2)
e1 = e.transform(csda[:,mask ==1].T)
_, l = prepare_umap_with_label_train(csda, seg_wlabel, 1)
#%%
hdb(e1, "//Users/shengkai/Desktop/paper/deaD/merge_rf/picked/seg/", fn =  "p21_seg_umap_canberra_ncom2_NN100_perc99.00_cl0_r60_cn1_with_label_r2")
#%%
plt.figure()
plt.scatter(*e1[np.array(seg_wlabel.sum(0)==1)[mask ==1],0:2].T, s = 1, c = l, cmap = "rainbow")

#%%
dp = "Users/shengkai/Desktop/paper/deaD/merge_rf/picked/seg/p21_seg_umap_canberra_ncom2_NN100_perc99.00_cl0_r60_cn1_with_label_r2"
num = 0
data = e1
tag = "seg%02d_wlabel_r2"%num
a = 10
b = 10
hdb_model = pickle.load(open('/Users/shengkai/Desktop/paper/deaD/merge_rf/picked/seg/p21_seg_umap_canberra_ncom2_NN100_perc99.00_cl0_r60_cn1_with_label_r2/hdbscan_10_10',"rb"))
segmentation_based_on_hdb_wonoise(data, hdb_model, 160, tag, mask, dp='/Users/shengkai/Desktop/paper/deaD/merge_rf/picked/seg/seg_wlabel_r2/', fn="seg")

#%%
csda_name, csda = load_mrc("/Users/shengkai/Desktop/paper/deaD/merge_rf/picked/", 160)
csda_bi = csda >1
occ = np.matmul(csda_bi, seg[[77,119,120, 21,0,22,134,42,111,81,92,105,40]].T)/seg[[77,119,120, 21,0,22,134,42,111,81,92,105,40]].sum(1)
sns.clustermap(occ[csda_bi.sum(1).argsort()], row_cluster = False, cmap = "Blues")

plt.figure()
plt.plot(occ[csda_bi.sum(1).argsort()])

#%%
fold_mat = np.load('/Users/shengkai/Desktop/paper/deaD/merge_rf/hier_result_tr_1.00/fold_mat.npy')
label = np.load('/Users/shengkai/Desktop/paper/deaD/merge_rf/hier_result_tr_1.00/label.npy')


mask = np.array([int(i[:2]) == 11 for i in label])

l = label[mask]
f_mat = fold_mat[mask][:,mask]

#%%
occ_med = np.zeros((csda.shape[0], seg_csda_HD.shape[0]))

for i, map_temp in enumerate(csda):
    for j, seg_temp in enumerate(seg_csda_HD):
        occ_med[i,j] = np.median(map_temp[seg_temp == 1])

#%%
occ = np.matmul(csda, seg_csda_HD.T)/seg_csda_HD.sum(1)
df_occ = pd.DataFrame(occ[csda_bi.sum(1).argsort()], index = [i.split("P11_")[1][:4] for i in np.array(csda_name)[csda_bi.sum(1).argsort()]], columns = seg_csda_HD_name)
sns.clustermap(df_occ, row_cluster = False, cmap = "Blues")


occ = np.matmul(csda_bi, seg_csda_HD.T)/seg_csda_HD.sum(1)
df_occ = pd.DataFrame(occ[csda_bi.sum(1).argsort()], index = [i.split("P11_")[1][:4] for i in np.array(csda_name)[csda_bi.sum(1).argsort()]], columns = seg_csda_HD_name)
sns.clustermap(df_occ, row_cluster = False, cmap = "Blues")

#%%
u = np.load('/Users/shengkai/Desktop/paper/deaD/merge_rf/picked/seg/p21_seg_umap_canberra_ncom2_NN100_perc99.00_cl0_r60_cn1.npy')
mask = mrcfile.open('/Users/shengkai/Desktop/paper/deaD/merge_rf/picked/seg/mask_box160_percentile_99.0.mrc').data.reshape(-1)



#%%
u = np.load('/Users/shengkai/Desktop/paper/deaD/merge_rf/picked/seg/p21_seg_umap_canberra_ncom2_NN100_perc99.00_cl0_r60_cn1.npy')
hdb(u, "//Users/shengkai/Desktop/paper/deaD/merge_rf/picked/seg/", [100], range(25,200,25), "p21_seg_umap_canberra_ncom2_NN100_perc99.00_cl0_r60_cn1")
#%%
hdb_test = pickle.load(open('/Users/shengkai/Desktop/paper/deaD/merge_rf/picked/seg/p21_seg_umap_canberra_ncom2_NN100_perc99.00_cl0_r60_cn1/hdbscan_100_50', "rb"))
segmentation_based_on_hdb_wonoise(u, hdb_test, 160, "seg_100_100", mask, '/Users/shengkai/Desktop/paper/deaD/merge_rf/picked/seg/', fn = "seg_100_50")
#%%
mask = mrcfile.open('/Users/shengkai/Desktop/paper/deaD/merge_rf/picked/seg/mask_box160_percentile_99.0.mrc').data.reshape(-1)

#%%
hdb(u[seg2[-2][mask==1]==1], "//Users/shengkai/Desktop/paper/deaD/merge_rf/picked/seg/", range(50,100),[5], "p21_seg_umap_canberra_ncom2_NN100_perc99.00_cl0_r60_cn1_seg100_50_seg9")
#%%
dp =  '/Users/shengkai/Desktop/paper/deaD/merge_rf/picked/seg/seg_100_100/round2/'
for i in [-2]:
    fn = "seg_100_50"
    if np.array([i]).T.shape[0]>1:
        for j in i:
            fn = fn + "_%02d"%j
    else:
        fn = fn + "_%02d"%i
    
    os.system("cd %s"%dp)
    os.system("mkdir %s"%fn)
    sub_divide(csda, seg2[i], fn, dp + fn, False)


#%%
occ = np.matmul(seg_csda_HD, seg.T)/seg.sum(1)
#%%
num=1
occ[num][occ[num].argsort()[::-1]]
np.array(seg_name)[occ[num].argsort()[::-1]]


#%%
s = sns_occ_cluster(csda_bi, seg_csda_HD, norm = True, norm_col = 5)
#%%
seg_num = 10
upper_list = np.array([0.5, 0.5, 0.7, 0.6, 0.5, 0.5, 0.5, 0.5, 0.7, 0.8])
lower_list = np.array([0.5, 0.5, 0.5, 0.6, 0.5, 0.5, 0.5, 0.5, 0.5, 0.7])

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
        
#%%  use more strict constraint

seg_num = 10
upper_list = np.array([0.5, 0.5, 0.7, 0.6, 0.5, 0.5, 0.5, 0.5, 0.7, 0.8])
lower_list = np.array([0.5, 0.5, 0.5, 0.6, 0.5, 0.5, 0.5, 0.5, 0.5, 0.7])


s_upper = s>=upper_list
s_lower = s<lower_list

columns = ["i", "j", "0_0", "0_1", "1_0", "1_1"]
    
qua = np.zeros((int(seg_num*(seg_num-1)/2),6))
count = 0
for i in range(seg_num-1):
    for j in range(i+1, seg_num):
        qua[count] = solve_uull(s_upper[:,i],s_upper[:,j],s_lower[:,i],s_lower[:,j], i,j)
        count += 1
    dependency = qua[np.logical_and(qua[:,3]==0, qua[:,4]>0)][:,0:2] 
    dependency = np.vstack((dependency,qua[np.logical_and(qua[:,4]<1, qua[:,3]>1)][:,0:2][:,::-1]))
    cor = qua[np.logical_and(qua[:,3]==0, qua[:,4]==0)][:,0:2]

#%%
DG=nx.DiGraph()
DG.add_edges_from(dependency.astype(int))
DG.remove_node(0)
#DG.remove_node(1)
DG_prune = prune_DG(DG)
plt.figure()
nx.draw_kamada_kawai(prune_DG2(DG_prune), with_labels = True)


#%%
class_order = [16, 9, 17 ,6, 18, 8, 20 ,19, 7,13 ,14 ,15 ,3 ,12 ,5 ,4  ,2 ,1 ,10 ,11 ,0]

#%%

s_lower = (s>=upper_list).T

s_upper = (s<=lower_list).T
columns = ["i", "j", "0_0", "0_1", "1_0", "1_1"]
seg_num = 21
qua = np.zeros((int(seg_num*(seg_num-1)/2),6))

count = 0
for i in range(seg_num-1):
    for j in range(i+1, seg_num):
        qua[count] = solve_uull(s_upper[:,i],s_upper[:,j],s_lower[:,i],s_lower[:,j], i,j)
        count += 1
    dependency = qua[np.logical_and(qua[:,3]<1, qua[:,4]>0)][:,0:2] 
    dependency = np.vstack((dependency,qua[np.logical_and(qua[:,4]<1, qua[:,3]>0)][:,0:2][:,::-1]))
    cor = qua[np.logical_and(qua[:,3]<1, qua[:,4]<1)][:,0:2]

#%%
DG=nx.DiGraph()
DG.add_edges_from(dependency.astype(int))

DG_prune = prune_DG(DG)
plt.figure()
nx.draw_kamada_kawai(DG_prune, with_labels = True)

#%%
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
h_set = set(["h%i"%i for i in list(range(1,120))]) 
p_set = set([ "uL4"])
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

#%% core
h_set = set(["h%i"%i for i in list(range(2,25))+[27,28,29,31]]) 
p_set = set([ "uL4", "uL22",  "uL24", "uL29"])
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

#%% block2 (23)
h_set = set(["h%i"%i for i in list(range(47,55))+[60,105,107]]) 
p_set = set([  "uL23"])
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

#%% block2 (23)
h_set = set(["h%i"%i for i in list(range(36,43))+[25,26,32,45,46,72,102]]) - {"h42"}
p_set = set([  "uL13", "bL20", "bL21", "bL34"])
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
#%% block2 (23)
h_set = set(["h%i"%i for i in list(range(81,89))+list(range(108,113)) + [38]])
p_set = set([  "uL15", "uL5", "uL18", "bL27", "bL25", "uL30"])
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
dp = "/Users/shengkai/Desktop/paper/deaD/merge_rf/picked/scatter_seg/"

for i in seg_name:
    plt.figure(figsize = (10,10))
    plt.scatter(*u.T, s = 3, color = "grey", alpha = 0.5)
    plt.scatter(*u[seg[np.where(np.array(seg_name) == i)[0]][:,mask ==1].reshape(-1)==1].T, c = "red", s = 3)
    plt.savefig(dp + "%s.png"%i)
    plt.close()
    
#%%
Block1 = ["bL21", "uL24", "uL3", "uL13","uL24" ,"bL34", "uL15", "uL22", "uL4", "bL19", "uL29","uL23"] +["h%i"%i for i in  
                                [1,94,46,7,24,5,104,13,2,12,100,14,19,28,41,72,6,50,40,9,45,
                                   3,102,20,26,36,22,8,29,18,4, 98,10,47,95,21,37,97,39,31,
                                   11,16,32,73,23,27,25,99,96,101,33,103,61,54,52,107,60,51,53,49,105,48]]

Block2 = ["bL25", "uL5", "bL27", "uL30", "uL18"] + ["h%i"%i for i in 
                     [38,112,84,86,108,109,110,81,82,83,111,85,87,88]]

Block3 = ["uL2", "uL14"] + ["h%i"%i for i in 
                     [55, 56, 57, 58, 59, 106,66,67,65,34,35,62,63,64]]

Block4 = ["bL28", "uL16", "uL6", "bL36"] + ["h%i"%i for i in 
                     [76,77,75,79,74,80,93,43,44,42,89,90,92,91]]

Block5 = ["bL17", "bL32", "bL35","bL33", "bL9", "uL10","uL11"] + ["h%i"%i for i in 
                     [68,70,69,71,78]]

for i, b in enumerate([Block1, Block2, Block3, Block4, Block5]):
    temp = np.zeros(160**3)
    for s in b:
        temp = temp + seg[np.where(np.array(seg_name)==s)]
    save_density(temp.reshape((160,160,160)), (2.62,2.62,2.62), "/Users/shengkai/Desktop/paper/deaD/L17_structurepaper_blocks/block%i.mrc"%(i+1))
#%%
block_name, block= load_mrc('/Users/shengkai/Desktop/paper/deaD/L17_structurepaper_blocks/', 160)
df_occ = pd.DataFrame(np.matmul(block, seg_csda_HD.T)/seg_csda_HD.sum(1), index = block_name,  columns = seg_csda_HD_name)
df_occ.to_csv('/Users/shengkai/Desktop/paper/deaD/L17_structurepaper_blocks/seg_occ.csv')


#%%
fold_mat = np.load("/Users/shengkai/Desktop/paper/deaD/merge_rf/hier_result_tr_1.00/fold_mat.npy")
label = np.load("/Users/shengkai/Desktop/paper/deaD/merge_rf/hier_result_tr_1.00/label.npy")
pid = np.array([i.split("_")[0] for i in label])

fold_mat_dead = np.abs(fold_mat[np.where(pid == "11")[0]][:,np.where(pid == "11")[0]])
l_dead = label[np.where(pid == "11")[0]]
#%%

fold_tr = 100
unfold_tr = 30
pathway = nx.DiGraph()
for i in range(21):
    for j in range(21):
        if fold_mat_dead[i,j] < unfold_tr and fold_mat_dead[j,i] < fold_tr and i!=j:
            pathway.add_edge(l_dead[j], l_dead[i])

#%%
unfold_tr = 100
minfold_offset = 1
fold_mat_dead = np.abs(fold_mat[np.where(pid == "11")[0]][:,np.where(pid == "11")[0]])

pathway = nx.DiGraph()
for i in range(21):
    temp_id = np.where(fold_mat_dead[i] < unfold_tr)[0]
    if temp_id.shape[0] > 1:
        temp_min = fold_mat_dead[temp_id, i][fold_mat_dead[temp_id, i].argsort()[1]]
        temp_fold_tr = temp_min * minfold_offset
        for j in temp_id:
            if fold_mat_dead[j, i] <= temp_fold_tr:
                pathway.add_edge(l_dead[i], l_dead[j])#,  weight = fold_mat_dead[j, i])

fold_mat_dead = fold_mat_dead.T
for i in range(21):
    temp_id = np.where(fold_mat_dead[i] < unfold_tr)[0]
    if temp_id.shape[0] > 1:
        temp_min = fold_mat_dead[temp_id, i][fold_mat_dead[temp_id, i].argsort()[1]]
        temp_fold_tr = temp_min * minfold_offset
        for j in temp_id:
            if fold_mat_dead[j, i] <= temp_fold_tr:
                pathway.add_edge(l_dead[j], l_dead[i])#, weight = fold_mat_dead[j, i])
#%%     

l17_name, l17 = load_mrc('/Users/shengkai/Desktop/paper/deaD/merge_rf/l17_merge_rf/', 160)
label = np.load("/Users/shengkai/Desktop/paper/deaD/merge_rf/hier_result_tr_1.00/label.npy")
l17_name = [i.split("P26_")[-1].split("_")[0] for i in l17_name]
l17_bi = l17 > 1
pid = np.array([i.split("_")[0] for i in label])
l17_label = label[np.where(pid == "26")[0]]

#%%
temp_order = [1830,1831,1832,1833,1810,1811,1812,1813,1814,1815,1804,1805,1806,1807,1801,1802,1803,1808,1809,1822,1823,1824,1825,1826,1827,1828,1829,1816,1817,1818,1819,1820,1821]
temp_label_order = [np.where(l17_label == "26_J%i"%i)[0] for i in temp_order]
pd.DataFrame(l17_bi.sum(1)[np.array(temp_label_order)]).to_csv("l17_MW.csv")
#%%
srmb_name, srmb = load_mrc('/Users/shengkai/Desktop/paper/deaD/merge_rf/srmb_merge_rf/', 160)
label = np.load("/Users/shengkai/Desktop/paper/deaD/merge_rf/hier_result_tr_1.00/label.npy")
srmb_name = [i.split("P15_")[-1].split("_")[0] for i in srmb_name]
srmb_bi = srmb > 1
#%%
srmb_label = label[np.where(pid == "15")[0]]

#%%
temp_order = [190,202,192,193,194,195,196,197,198,199,200,201]
temp_label_order = [np.where(srmb_label == "15_J%i"%i)[0] for i in temp_order]

pd.DataFrame(srmb_bi.sum(1)[np.array(temp_label_order)]).to_csv("srmb_MW.csv")

#%%
meta = pd.read_csv('/Users/shengkai/Desktop/paper/deaD/class_meta_data_summary.csv')
M = np.where(np.logical_and(label!="26_J1817", label!="15_J199"))[0]
name_dict = {i:j for i, j in zip(meta.label, meta.name)}
name_list = [name_dict[i] for i in label[M]]



#%%
dist_mat = np.load("/Users/shengkai/Desktop/paper/deaD/merge_rf/hier_result_tr_1.00/dist_mat.npy")*1.2
dist_mat = dist_mat[M][:,M]
plt.figure()
linkage_matrix = linkage(np.array([dist_mat[i,j] for i in range(dist_mat.shape[0]-1) for j in range(i+1,dist_mat.shape[0])]), "average")
dendrogram(linkage_matrix, color_threshold=210, labels=name_list,show_leaf_counts=True,leaf_font_size=12)
plt.axhline(y=220, color = "grey", lw=1, linestyle='dashed')
#plt.savefig("./hier_result/fdendrogram.png")
#plt.close()


#%%
M = np.where(pid == '11')[0]
name_list = [name_dict[i] for i in label[M]]
dist_mat = np.load("/Users/shengkai/Desktop/paper/deaD/merge_rf/hier_result_tr_1.00/dist_mat.npy")*1.2
dist_mat = dist_mat[M][:,M]
f1 = plt.figure()
linkage_matrix = linkage(np.array([dist_mat[i,j] for i in range(dist_mat.shape[0]-1) for j in range(i+1,dist_mat.shape[0])]), "average")
dendrogram(linkage_matrix, color_threshold=210, labels=name_list,show_leaf_counts=True,leaf_font_size=12)
f1 = plt.figure("GOOD1")
ax = f1.add_subplot(111)
ax.axis('off')
#%%
aa = dendrogram(linkage_matrix, color_threshold=220, labels=name_list,show_leaf_counts=True,leaf_font_size=12)
csda_name, csda = load_mrc("/Users/shengkai/Desktop/paper/deaD/merge_rf/picked/", 160)
csda_bi = csda >1

#%%
s = sns_occ_cluster(csda_bi, seg_csda_HD, norm = True, norm_col = 5)

#%%
ss = s[aa["leaves"]]
df_occ = pd.DataFrame(ss, index = aa["ivl"],  columns = seg_csda_HD_name)
df_occ = df_occ.T
df_occ.to_csv('/Users/shengkai/Desktop/paper/deaD/hierachical_occ_dead.csv')
#%%

adj_matrix = [[1,0,0,0,0,0,0,0,0],
              [1,1,0,0,0,0,0,0,0],
              [1,0,1,0,0,0,0,0,0],
              [1,0,0,1,0,0,0,0,0],
              [0,1,0,0,1,0,0,0,0],
              [0,0,0,1,1,1,0,0,0],
              [0,0,1,0,0,0,1,0,0],
              [0,0,1,0,1,0,0,1,0],
              [0,0,0,0,0,1,0,1,1]]
adj_matrix = np.array(adj_matrix)

#%%
all_map = []
for i in range(256, 512):
    temp_map = bin(i).split("b")[-1]
    all_map = all_map + [temp_map]
#%%
unqualified_map = []
for i,temp_map in enumerate(all_map):
    temp = [m.start() for m in re.finditer('1', temp_map)]
    for j in temp:
        if set(np.where(adj_matrix[j]==1)[0]).issubset(temp)==False:
            unqualified_map += [i]
            break

#%%

qualified_map = [list(m) for i,m in enumerate(all_map) if i not in unqualified_map]
qualified_map_len = np.array([i.count("1") for i in qualified_map])

#%%
path = nx.DiGraph()

for i in range(1,9):
    for j in np.where(qualified_map_len==i)[0]:
        for k in np.where(qualified_map_len==i+1)[0]:
            if np.sum(np.array(qualified_map[j]) != np.array(qualified_map[k])) == 1:
                path.add_edge(''.join(qualified_map[j]), ''.join(qualified_map[k]))


#%%
plt.figure()
nx.draw_kamada_kawai(path, with_labels = True)


#%%
col_list = ["chocolate", "skyblue", "yellowgreen", "purple", "lightcoral", "red", "royalblue", "gold", "darkgreen", "violet"]
u = np.load('/Users/shengkai/Desktop/paper/deaD/merge_rf/picked/seg/p21_seg_umap_canberra_ncom2_NN100_perc99.00_cl0_r60_cn1.npy')
mask = mrcfile.open('/Users/shengkai/Desktop/paper/deaD/merge_rf/picked/seg/mask_box160_percentile_99.0.mrc').data.reshape(-1)
#%%
plt.figure(figsize=(10,10))
plt.scatter(*u.T, s = 1, alpha = 0.5, color = "grey")
for i, temp_seg in enumerate(seg_csda_HD):
    plt.scatter(*u[temp_seg[mask == 1] ==1].T, color = col_list[i], s = 1)


#%%
#exit_tunnel plot
for i in [19,20,23,36,40,41,47,52,69,79,90,132,133,134,137,131,21, 10, 85]:
    run(session, "volume #6.%i color brown level 0.15"%i)

for i in [94,95,111,113,116,128]:
    run(session, "volume #6.%i color cyan level 0.15"%i)
    
#%% iSAT dependency
d = pd.read_csv("dependancy(1).csv", index_col=0).values
adj_matrix = d
all_map = []
for i in range(2**13, 2**14):
    temp_map = bin(i).split("b")[-1]
    all_map = all_map + [temp_map]
#%%
unqualified_map = []
for i,temp_map in enumerate(all_map):
    temp = [m.start() for m in re.finditer('1', temp_map)]
    for j in temp:
        if set(np.where(adj_matrix[j]==1)[0]).issubset(temp)==False:
            unqualified_map += [i]
            break

#%%

qualified_map = [list(m) for i,m in enumerate(all_map) if i not in unqualified_map]
qualified_map_len = np.array([i.count("1") for i in qualified_map])

#%%
path = nx.DiGraph()

for i in range(1,9):
    for j in np.where(qualified_map_len==i)[0]:
        for k in np.where(qualified_map_len==i+1)[0]:
            if np.sum(np.array(qualified_map[j]) != np.array(qualified_map[k])) == 1:
                path.add_edge(''.join(qualified_map[j]), ''.join(qualified_map[k]))


