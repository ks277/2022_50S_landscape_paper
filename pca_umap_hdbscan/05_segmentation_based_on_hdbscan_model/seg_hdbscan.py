#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 16:09:44 2023

@author: shengkai
"""
import numpy as np
import os
import matplotlib.pyplot as plt
import mrcfile
import sys
import hdbscan
import pickle

data = np.load("%s"%sys.argv[1])
hdb_model = pickle.load(open("%s"%sys.argv[2], "rb"))
box = int(sys.argv[3])
apix = float(sys.argv[4])
tag = sys.argv[5]
mask = mrcfile.open("%s"%sys.argv[6]).data.reshape(-1)
dp = "."


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

#%% segmentation_based_on_hdb_wonoise

def segmentation_based_on_hdb_wonoise(data, hdb_model, tag, mask, dp, fn = "seg"):
    if mask.reshape((-1, box**3)).shape[0]!=1:
        mask = mask.sum(0)
#    hdb_pred = hdbscan.all_points_membership_vectors(hdb_model)
    seg_argmax=hdb_model.labels_
    name_list = ["%s_sub%02i"%(tag, i) for i in range(0,np.max(seg_argmax)+1)]
    os.system("mkdir "+ dp + "/%s"%fn)
    for i in range(0,np.max(seg_argmax)+1):
        name = dp +"/%s/%s"%(fn, name_list[i])
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

segmentation_based_on_hdb_wonoise(data, hdb_model, tag, mask, dp)