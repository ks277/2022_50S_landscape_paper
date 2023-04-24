#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 18 19:10:44 2022

@author: shengkai
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import mrcfile
import seaborn as sns
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


#%%
mask = np.load("/Users/shengkai/Desktop/paper/deaD/segmentation/mask_box160_percentile_99.0.npy")
pass_seg_name, pass_seg = load_mrc('/Users/shengkai/Desktop/paper/deaD/merge_rf/picked/seg/passthrough_seg/hidedust/rename/', 160)

pass_seg = pass_seg[:, mask == 1]
noise = pass_seg.sum(0)==0
#%%
seg_color = ["skyblue", "pink", "darkgreen", "royalblue", "lightcoral", "red", "chocolate", "purple", "gold", "greenyellow"]

#%%
pass_seg_occ = np.matmul(seg[:,mask ==1],  pass_seg.T)
pass_seg_occ_norm  = pass_seg_occ / seg.sum(1).reshape(-1,1)

#%%
df_pass_seg = pd.DataFrame(pass_seg_occ_norm, index = seg_name, columns = [i.split(".")[0] for i in pass_seg_name] )
plt.figure()
sns.clustermap(df_pass_seg.T, cmap = "Blues", #row_cluster=False,
                             method = "ward", metric = "euclidean",
                             yticklabels=True, xticklabels=True)
df_pass_seg.to_csv("/Users/shengkai/Desktop/paper/deaD/segmentation/df_pass_seg_occ.csv")
#%%
dp = "/Users/shengkai/Desktop/paper/deaD/segmentation/pca_umap/"
name_list = [i for i in os.listdir(dp) 
             if i.split(".")[-1] == "npy"]
name_list.sort()

for i in name_list:
    temp = np.load(dp + i)
    plt.figure()
    
    plt.scatter(*temp.T, s = 1, alpha = 0.3, c = "grey")
    plt.savefig(dp + i + ".png")
    plt.close()

for i in name_list:
    temp = np.load(dp + i)
    plt.figure()
    
    plt.scatter(*temp[noise ==1].T, s = 1, alpha = 0.3, c = "grey")
    for seg, col in zip(pass_seg, seg_color):
        plt.scatter(*temp[seg ==1].T, s = 1, alpha = 0.3, c = col)
    plt.savefig(dp + i + "block.png")
    plt.close()

#%%
dp = "/Users/shengkai/Desktop/paper/deaD/segmentation/pca_umap/canberra/"
name_list = [i for i in os.listdir(dp) 
             if i.split(".")[-1] == "npy"]
name_list.sort()
#%%

figure, axis = plt.subplots(4, 5,figsize=(16,12))#, layout="constrained")
plt.rcParams['figure.constrained_layout.use'] = True
#figure.tight_layout()
ax_l = [[i,j] for i in range(4) for j in range(5)]

for count, i in enumerate(name_list):
    
    temp = np.load(dp + i)
    
    
#    axis[ax_l[count][0], ax_l[count][1]].scatter(*temp[noise ==1].T, s = 0.5, alpha = 0.1, c = "grey")
    for seg, col in zip(pass_seg, seg_color):
        axis[ax_l[count][0], ax_l[count][1]].scatter(*temp[seg ==1].T, s = 0.5, alpha = 0.1, c = col)
    axis[ax_l[count][0], ax_l[count][1]].set_title("%i Features"%(count+2))
    count = count + 1
  #  plt.savefig(dp + i + "block.png")
  #  plt.close()

#%%
figure, axis = plt.subplots(4, 5,figsize=(16,12))#, layout="constrained")
plt.rcParams['figure.constrained_layout.use'] = True
#figure.tight_layout()
ax_l = [[i,j] for i in range(4) for j in range(5)]

for count, i in enumerate(name_list):
    
    temp = np.load(dp + i)
    
    
    axis[ax_l[count][0], ax_l[count][1]].scatter(*temp.T, s = 0.5, alpha = 0.1, c = "grey")
   
    axis[ax_l[count][0], ax_l[count][1]].set_title("%i Features"%(count+2))
    count = count + 1
  #  plt.savefig(dp + i + "block.png")
#%%
dp = "/Users/shengkai/Desktop/paper/deaD/segmentation/pca_umap/euclidean/"
name_list = [i for i in os.listdir(dp) 
             if i.split(".")[-1] == "npy"]
name_list.sort()
#%%

figure, axis = plt.subplots(4, 5,figsize=(16,12))#, layout="constrained")
plt.rcParams['figure.constrained_layout.use'] = True
#figure.tight_layout()
ax_l = [[i,j] for i in range(4) for j in range(5)]

for count, i in enumerate(name_list):
    
    temp = np.load(dp + i)
    
    
#    axis[ax_l[count][0], ax_l[count][1]].scatter(*temp[noise ==1].T, s = 0.5, alpha = 0.1, c = "grey")
    for seg, col in zip(pass_seg, seg_color):
        axis[ax_l[count][0], ax_l[count][1]].scatter(*temp[seg ==1].T, s = 0.5, alpha = 0.1, c = col)
    axis[ax_l[count][0], ax_l[count][1]].set_title("%i Features"%(count+2))
    count = count + 1
  #  plt.savefig(dp + i + "block.png")
  #  plt.close()
#%%
figure, axis = plt.subplots(4, 5,figsize=(16,12))#, layout="constrained")
plt.rcParams['figure.constrained_layout.use'] = True
#figure.tight_layout()
ax_l = [[i,j] for i in range(4) for j in range(5)]

for count, i in enumerate(name_list):
    
    temp = np.load(dp + i)
    
    
    axis[ax_l[count][0], ax_l[count][1]].scatter(*temp.T, s = 0.5, alpha = 0.1, c = "grey")
   
    axis[ax_l[count][0], ax_l[count][1]].set_title("%i Features"%(count+2))
    count = count + 1
  #  plt.savefig(dp + i + "block.png")

#%%
dp = "/Users/shengkai/Desktop/paper/deaD/segmentation/direct_umap/"
name_list = [i for i in os.listdir(dp) 
             if i.split(".")[-1] == "npy"]
name_list.sort()

for i in name_list:
    temp = np.load(dp + i)
    plt.figure()
    
    plt.scatter(*temp.T, s = 1, alpha = 0.3, c = "grey")
    plt.savefig(dp + i + ".png")
    plt.close()

for i in name_list:
    temp = np.load(dp + i)
    plt.figure()
    
    plt.scatter(*temp[noise ==1].T, s = 1, alpha = 0.3, c = "grey")
    for seg, col in zip(pass_seg, seg_color):
        plt.scatter(*temp[seg ==1].T, s = 1, alpha = 0.3, c = col)
    plt.savefig(dp + i + "block.png")
    plt.close()
    
    
#%%
temp = np.load("/Users/shengkai/Desktop/paper/deaD/segmentation/pca_umap/p21_seg_umap_canberra_ncom2_NN100.npy")
p21_seg = np.load("/Users/shengkai/Desktop/paper/deaD/segmentation/pca_umap/raw_p/p21_seg.npy")
#%%
a = p21_seg
np.save("level.npy", np.std(a,0))
plt.hist(a.T[0], bins = 1000)
aa = a[:,0]>np.std(a[:,0])*2
plt.figure()
plt.scatter(*temp.T, s = 1, alpha = 0.3, c = "grey")
plt.scatter(*temp[aa ==1].T, s = 1, alpha = 0.3, c = "blue")
aa = a[:,1]>np.std(a[:,1])*2
plt.scatter(*temp[aa ==1].T, s = 1, alpha = 0.3, c = "red")


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
for c,i in enumerate(a.T):
    temp = np.zeros(160**3)
    temp[mask ==1] = i
    save_density(temp.reshape((160,160,160)), (2.62,2.62,2.62), 
                 "/Users/shengkai/Desktop/paper/deaD/segmentation/pca_umap/raw_p/component_%02d_pos.mrc"%c)

#%%
#%%
for c,i in enumerate(a.T):
    temp = np.zeros(160**3)
    temp[mask ==1] = i
    save_density(-temp.reshape((160,160,160)), (2.62,2.62,2.62), 
                 "/Users/shengkai/Desktop/paper/deaD/segmentation/pca_umap/raw_pcomponent_%02d_neg.mrc"%c)
#%%
import matplotlib.image as img

dp = '/Users/shengkai/Desktop/paper/deaD/segmentation/pca_umap/raw_p/'
name_list = [i for i in os.listdir(dp) 
             if i.split("_")[-1] == "pos.png"]

figure, axis = plt.subplots(3, 7,figsize=(14, 6))#, layout="constrained")
plt.rcParams['figure.constrained_layout.use'] = True
#figure.tight_layout()
ax_l = [[i,j] for i in range(3) for j in range(7)]

name_list.sort()
for count, i in enumerate(name_list):
    im = img.imread(dp + i)
    axis[ax_l[count][0], ax_l[count][1]].imshow(im)
    axis[ax_l[count][0], ax_l[count][1]].set_title("Component %i"%(count+1))
    axis[ax_l[count][0], ax_l[count][1]].axis('off')
    count = count + 1

#%%
import matplotlib.image as img

dp = '/Users/shengkai/Desktop/paper/deaD/segmentation/pca_umap/raw_p/'
name_list = [i for i in os.listdir(dp) 
             if i.split("_")[-1] == "combined.png"]

figure, axis = plt.subplots(4, 5,figsize=(15, 12))#, layout="constrained")
plt.rcParams['figure.constrained_layout.use'] = True
#figure.tight_layout()
ax_l = [[i,j] for i in range(4) for j in range(5)]

name_list.sort()
for count, i in enumerate(name_list):
    im = img.imread(dp + i)
    axis[ax_l[count][0], ax_l[count][1]].imshow(im)
    axis[ax_l[count][0], ax_l[count][1]].set_title("Component %i"%(count+1))
    axis[ax_l[count][0], ax_l[count][1]].axis('off')
  #  count = count + 1
        
#%% import data

seg = np.load("/Users/shengkai/Desktop/data/PDBs_forAlignment_box160/occ_origin_matrix.npy")
unmasked_map = np.load("/Users/shengkai/Desktop/data/Biopaper_unmaskedmaps/unmasked_map_wo70S.npy")

bin_name = np.load("/Users/shengkai/Desktop/data/PDBs_forAlignment_box160/bin_name.npy")
bin_map = np.load("/Users/shengkai/Desktop/data/Biopaper_unmaskedmaps/bin_map_wo70S.npy")

map_namelist = np.load("/Users/shengkai/Desktop/data/Biopaper_unmaskedmaps/map_namelist_wo70S.npy")
seg_name = [i.split("_")[1].split(".")[0] for i in bin_name]
pattern = "[L]"

#%%
p21_seg_std = np.std(p21_seg,0)

p21_seg_bi = p21_seg > p21_seg_std*1
p21_seg_99 = np.percentile(p21_seg, 95,0)
#p21_seg_bi = p21_seg > p21_seg_99
p21_occ = np.matmul(seg[:,mask ==1],  p21_seg_bi)

p21_occ_norm  = p21_occ / seg.sum(1).reshape(-1,1)

#%%
p21_seg_std = np.std(p21_seg,0)

p21_seg_bi = p21_seg < -p21_seg_std*1
p21_seg_5 = np.percentile(p21_seg, 5, 0)
#p21_seg_bi = p21_seg < p21_seg_5
p21_occ_neg = np.matmul(seg[:,mask ==1],  p21_seg_bi)

p21_occ_neg_norm  = p21_occ_neg / seg.sum(1).reshape(-1,1)

p21_occ_cul = sns.clustermap(np.hstack((p21_occ_norm, p21_occ_neg_norm)), cmap = "Blues", col_cluster=False)

#%%
p21_occ_cul = sns.clustermap(np.hstack((p21_occ_norm, -p21_occ_neg_norm)), cmap = "RdBu", col_cluster=False)
#%%

df_p21_norm_occ = pd.DataFrame(np.hstack((p21_occ_norm, -p21_occ_neg_norm))[p21_occ_cul.dendrogram_row.dendrogram["leaves"]], 
             index = np.array(seg_name)[p21_occ_cul.dendrogram_row.dendrogram["leaves"]],
             columns=["Pos_%02d"%(i+1) for i in range(21)] + ["Neg_%02d"%(i+1) for i in range(21)])

#%%
#df_p21_norm_occ.T.to_csv("/Users/shengkai/Desktop/paper/deaD/segmentation/df_p21_norm_occ_5_95_perc.csv")
df_p21_norm_occ.to_csv("/Users/shengkai/Desktop/paper/deaD/segmentation/df_p21_norm_occ_sigma1_T.csv")

#%%
df_p21_norm_combined = pd.DataFrame(p21_occ_norm - p21_occ_neg_norm,
                                    index = seg_name, columns=["PC %02d"%(i+1) for i in range(21)]) 

df_p21_norm_combined.to_csv("/Users/shengkai/Desktop/paper/deaD/segmentation/df_p21_norm_occ_sigma1_combined_T.csv")

""" to draw the occ_mat
python OccMat_PosNeg_Dendro.py --csvfile df_p21_norm_occ_sigma1_combined_T.csv --cluster_cols False --xsize 15 --ysize 15 --border_color w
"""
#%%
sns.clustermap(df_p21_norm_occ.T, cmap = "RdBu", row_cluster=False,
                             method = "average", metric = "cosine",
                             yticklabels=True)
#%%
sns.clustermap(df_p21_norm_occ.T, cmap = "RdBu", row_cluster=False,
                             method = "ward", metric = "euclidean",
                             yticklabels=True, xticklabels=True)


#%%
plt.rcParams.update({'font.size': 18})
plt.figure(figsize = (10,10))
plt.scatter(p21_seg[:,1],p21_seg[:,2],s=1, color = "grey", alpha = 0.5)
plt.xlabel("PC-2")
plt.ylabel("PC-3")
plt.axvline([p21_seg_std[0]], color = "grey", linestyle="--")
plt.axvline([-p21_seg_std[0]], color = "grey", linestyle="--")
pos_mask = p21_seg[:,1]>p21_seg_std[0]
neg_mask = p21_seg[:,1]<-p21_seg_std[0]
plt.scatter(p21_seg[pos_mask,1],p21_seg[pos_mask,2],s=1, color = "navy", alpha = 0.5)
plt.scatter(p21_seg[neg_mask,1],p21_seg[neg_mask,2],s=1, color = "darkred", alpha = 0.5)

#%%
plt.rcParams.update({'font.size': 18})
plt.figure(figsize = (10,10))

pos_mask = p21_seg[:,1]>p21_seg_std[0]
neg_mask = p21_seg[:,1]<-p21_seg_std[0]
center_mask = np.logical_and(~pos_mask,~neg_mask)

#%%

plt.hist([p21_seg[neg_mask,1],p21_seg[center_mask,1],p21_seg[pos_mask,1]], bins =1000,
          color = ["darkred","grey","navy"], alpha = 0.9, stacked = True)
plt.axvline([p21_seg_std[0]], color = "grey", linestyle="--")
plt.axvline([-p21_seg_std[0]], color = "grey", linestyle="--")

#%%
def scatter_hist(x, y, ax, ax_histx, ax_histy):
    # no labels
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    # the scatter plot:
    ax.scatter(x, y, s=1,color = "grey", alpha = 0.5)
    ax.scatter(p21_seg[pos_mask,1],p21_seg[pos_mask,2],s=1, color = "navy", alpha = 0.5)
    ax.scatter(p21_seg[neg_mask,1],p21_seg[neg_mask,2],s=1, color = "darkred", alpha = 0.5)

    # now determine nice limits by hand:
    binwidth = 0.25
    xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
    lim = (int(xymax/binwidth) + 1) * binwidth

    bins = np.arange(-lim, lim + binwidth, binwidth)
    ax_histx.hist([p21_seg[neg_mask,1],p21_seg[center_mask,1],p21_seg[pos_mask,1]], bins =1000,
          color = ["darkred","grey","navy"], alpha = 0.9, stacked = True)
    ax_histy.hist(p21_seg[:,2], bins =1000,
          color = ["grey"], alpha = 0.9, stacked = True, orientation='horizontal')
    
    
# Start with a square Figure.
fig = plt.figure(figsize=(6, 6))
# Add a gridspec with two rows and two columns and a ratio of 1 to 4 between
# the size of the marginal axes and the main axes in both directions.
# Also adjust the subplot parameters for a square plot.
gs = fig.add_gridspec(2, 2,  width_ratios=(4, 1), height_ratios=(1, 4),
                      left=0.1, right=0.9, bottom=0.1, top=0.9,
                      wspace=0.05, hspace=0.05)
# Create the Axes.
ax = fig.add_subplot(gs[1, 0])
ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)
# Draw the scatter plot and marginals.
scatter_hist(p21_seg[:,1],p21_seg[:,2], ax, ax_histx, ax_histy)

#%%
ax.axvline([p21_seg_std[0]], color = "grey", linestyle="--")
ax.axvline([-p21_seg_std[0]], color = "grey", linestyle="--")
ax.set_xlabel("PC-2")
ax.set_ylabel("PC-3")

ax_histx.axvline([p21_seg_std[0]], color = "grey", linestyle="--")
ax_histx.axvline([-p21_seg_std[0]], color = "grey", linestyle="--")
