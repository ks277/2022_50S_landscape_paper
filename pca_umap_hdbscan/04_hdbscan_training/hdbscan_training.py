#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 15:23:01 2022

@author: shengkai
"""


#%% load package and function
import numpy as np
import os
import mrcfile
import matplotlib.pyplot as plt
import umap
from sklearn.decomposition import PCA
import hdbscan
import pickle
import sys

u_name = sys.argv[1]
data = np.load(u_name)
#apix = float(sys.argv[2])
#box = int(sys.argv[3])

#%%
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


#%%
def load_mrc(dp, box):
    if dp[-1]!="/": dp = dp + "/"
    name_list = [i for i in os.listdir(dp) if i.split(".")[-1]=="mrc"]
    name_list.sort()
    num = len(name_list)
    temp = np.zeros((num, box**3))
    for i, name in enumerate(name_list):
        temp[i] = mrcfile.open(dp + name).data.reshape(-1)
    return (name_list, temp)


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
        

#%%

hdb(data, ".", fn = "hdbscan_parameter_screening")

#%%
"""
hdb_model = pickle.load(open(dp+ "/u_p64/hdbscan_200_10","rb"))

segmentation_based_on_hdb_wonoise(data, box=160, dp=dp, tag="seg_200_10", mask=mask, hdb_model=hdb_model)

#%%
new_seg_name, new_seg = load_mrc("/Users/shengkai/Desktop/paper/deaD/segmentation/threedataset_seg/seg/hidedust", 160)
pass_seg_name, pass_seg = load_mrc('/Users/shengkai/Desktop/paper/deaD/merge_rf/picked/seg/passthrough_seg/hidedust/rename/', 160)

#%%
occ_mat = np.matmul(new_seg, pass_seg.T)
occ_norm_mat = occ_mat/pass_seg.T.sum(0)
df = pd.DataFrame(occ_norm_mat, columns  = [i.split(".")[0] for i in pass_seg_name], index = [i.split(".")[0].split("_")[-1] for i in new_seg_name])

df.to_csv("/Users/shengkai/Desktop/paper/deaD/segmentation/threedataset_seg/seg/hidedust/occ_mat_old_new.csv")
#%%
occ_norm_mat = occ_mat/new_seg.T.sum(0).reshape(-1,1)

#%%

for seg, n in zip(new_seg, new_seg_name):
    fig, ax = plt.subplots(figsize = (5,5))
    ax.scatter(*temp.T, s = 1, alpha = 0.1, c = "grey")
    

    ax.scatter(*temp[seg[mask==1] ==1].T, s = 1, alpha = 0.1, c = "black")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(direction = "in")
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.tick_params(direction = "in", width = 2)

    plt.savefig("/Users/shengkai/Desktop/paper/deaD/segmentation/threedataset_seg/%s_seg.png"%n.split(".")[0].split("_")[-1])
    plt.close()
#    plt.savefig(dp + i + "block.png")

#%% import data

seg = np.load("/Users/shengkai/Desktop/data/PDBs_forAlignment_box160/occ_origin_matrix.npy")
unmasked_map = np.load("/Users/shengkai/Desktop/data/Biopaper_unmaskedmaps/unmasked_map_wo70S.npy")

bin_name = np.load("/Users/shengkai/Desktop/data/PDBs_forAlignment_box160/bin_name.npy")
bin_map = np.load("/Users/shengkai/Desktop/data/Biopaper_unmaskedmaps/bin_map_wo70S.npy")

map_namelist = np.load("/Users/shengkai/Desktop/data/Biopaper_unmaskedmaps/map_namelist_wo70S.npy")
seg_name = [i.split("_")[1].split(".")[0] for i in bin_name]
pattern = "[L]"

#%%
occ_mat = np.matmul(new_seg, seg.T)
occ_norm_mat = occ_mat/seg.T.sum(0)
df = pd.DataFrame(occ_norm_mat, columns  = seg_name, index = [i.split(".")[0].split("_")[-1] for i in new_seg_name])
df.T.to_csv("/Users/shengkai/Desktop/paper/deaD/segmentation/threedataset_seg/seg/hidedust/occ_mat_4ybb_new.csv")




#%% renamed seg
new_seg_rename_name, new_seg_rename = load_mrc("/Users/shengkai/Desktop/paper/deaD/segmentation/threedataset_seg/seg/hidedust/rename", 160)

occ_mat = np.matmul(new_seg_rename, seg.T)
occ_norm_mat = occ_mat/seg.T.sum(0)
df = pd.DataFrame(occ_norm_mat, columns  = seg_name, index = [i.split(".")[0].split("_")[-1] for i in new_seg_rename_name])
df.T.to_csv("/Users/shengkai/Desktop/paper/deaD/segmentation/threedataset_seg/seg/hidedust/rename/occ_mat_4ybb_new.csv")

#%%
pass_seg_name, pass_seg = load_mrc('/Users/shengkai/Desktop/paper/deaD/merge_rf/picked/seg/passthrough_seg/hidedust/rename/', 160)

occ_mat = np.matmul(new_seg_rename, pass_seg.T)
occ_norm_mat = occ_mat/pass_seg.T.sum(0)
df = pd.DataFrame(occ_norm_mat, columns  = [i.split(".")[0] for i in pass_seg_name], index = [i.split(".")[0].split("_")[-1] for i in new_seg_rename_name])

df.to_csv("/Users/shengkai/Desktop/paper/deaD/segmentation/threedataset_seg/seg/hidedust/rename/occ_mat_old_new.csv")

#%%
plt.rcParams.update({'font.size': 25 ,'font.sans-serif': 'Arial'})

for seg, n in zip(new_seg_rename, new_seg_rename_name):
    fig, ax = plt.subplots(figsize = (5,5))
    ax.scatter(*temp.T, s = 1, alpha = 0.1, c = "grey")
    

    ax.scatter(*temp[seg[mask==1] ==1].T, s = 1, alpha = 0.1, c = "black")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    plt.title(n.split(".")[0])

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(direction = "in")
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.tick_params(direction = "in", width = 2)

    plt.savefig("/Users/shengkai/Desktop/paper/deaD/segmentation/threedataset_seg/seg/hidedust/rename/%s.png"%n.split(".")[0])
    plt.close()

#%%
mask = np.load('/Users/shengkai/Desktop/paper/deaD/segmentation/threedataset_seg/mask_box160_percentile_99.0.npy')

dp = "/Users/shengkai/Desktop/paper/deaD/segmentation/threedataset_seg/seg/hidedust/rename/"
plt.rcParams.update({'font.size': 25})
for seg, col, n in zip(pass_seg, seg_color, pass_seg_name):
    fig, ax = plt.subplots(figsize = (5,5))

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    plt.title(n.split(".")[0])

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(direction = "in")
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.tick_params(direction = "in", width = 2)
    
    ax.scatter(*temp.T, s = 1, alpha = 0.1, c = "grey")
    ax.scatter(*temp[seg[mask ==1] ==1].T, s = 1, alpha = 0.3, c = col)
    plt.savefig(dp + n + "_block.png")

    plt.close()
    
#%%
blk_name, blk = load_mrc('/Users/shengkai/Desktop/paper/deaD/segmentation/threedataset_seg/seg/hidedust/rename', 160)
blk_name = [i.split(".")[0] for i in blk_name]
mask = np.load('/Users/shengkai/Desktop/paper/deaD/segmentation/threedataset_seg/mask_box160_percentile_99.0.npy')
blk = blk[:,mask == 1]

#%%
temp = np.load('/Users/shengkai/Desktop/paper/deaD/segmentation/threedataset_seg/p64_seg_umap_canberra_ncom2_NN100.npy')
plt.rcParams.update({'font.size': 12 ,'font.sans-serif': 'Arial'})

fig, ax = plt.subplots(figsize = (5,5))
ax.scatter(*temp[blk.sum(0)==0].T, s = 1, alpha = 0.1, c = "grey")
viridis = cm.get_cmap('viridis', 18)

for  seg, n in zip(blk, blk_name):
    ax.scatter(*temp[seg ==1].T, s = 1, alpha = 0.1, c = matplotlib.colors.rgb2hex(viridis(int(n.split("blk")[-1]))))
  #  print(int(n.split("blk")[-1]))
    ax.annotate("%s"%n, np.median(temp[seg ==1],0), font = "Arial", fontsize = 12, fontweight='bold')
    
ax.set_xlabel("UMAP 1")
ax.set_ylabel("UMAP 2")

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(direction = "in")
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.tick_params(direction = "in", width = 2)


"""
