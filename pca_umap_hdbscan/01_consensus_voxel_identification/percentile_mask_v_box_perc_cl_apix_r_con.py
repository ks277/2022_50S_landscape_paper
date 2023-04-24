#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 12:26:37 2022

creating percentile mask based on multiple mrc

@author: shengkai
"""

import numpy as np
import mrcfile
import os
import sys


box = int(sys.argv[1])
percentile = float(sys.argv[2])
cloud_size = int(sys.argv[3])
apix = float(sys.argv[4])
r = int(sys.argv[5])
con = int(sys.argv[6])
"""
test; 
box = 256
percentile = 98
cloud_size = 1
apix = 2.62
r = 60
"""
mapd = box
fn = "useg_perc%.2f_cl%i_r%i_cn%i"%(percentile, cloud_size, r, con)
os.system("mkdir %s"%fn)

#%%
radius = np.zeros((mapd, mapd, mapd))
cart = np.zeros((mapd, mapd, mapd, 3))
med = (mapd-1)/2

for i in range(mapd):
    for j in range(mapd):
        for k in range(mapd):
            radius[i,j,k] = np.sqrt((i-med)**2 + (j-med)**2 + (k-med)**2)
            cart[i,j,k] = np.array([i,j,k])

cart = cart - med
mask_r = radius<r

np.save("./%s/radius_box%i.npy"%(fn, mapd), radius)
np.save("./%s/cart_box%i.npy"%(fn, mapd), cart.reshape(-1,3))
np.save("./%s/r%i_box%i.npy"%(fn, r,mapd), mask_r)

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
    
name_list = [i for i in os.listdir(".") if i.split(".")[-1]=="mrc"]

mask = np.zeros(box**3)

for i in name_list:
    print("process:" + i)
    temp = mrcfile.open(i, permissive=True).data.reshape(-1)
    mask = mask + 1*(temp > np.percentile(temp, percentile))

combine_mask = np.logical_and(cloud_seg(mask>= con, (box,box,box), cloud_size)!=0 , mask_r.reshape(-1)!=0)

np.save("./%s/masked_radius_box%i.npy"%(fn, mapd), radius.reshape(-1)[combine_mask==1])
np.save("./%s/masked_cart_box%i.npy"%(fn, mapd), cart.reshape(-1,3)[combine_mask==1])
np.save("./%s/mask_box%i_percentile_%.1f.npy"%(fn, box, percentile), combine_mask)
save_density(combine_mask.reshape((box,box,box)), (apix,apix,apix), "./%s/mask_box%i_percentile_%.1f.mrc"%(fn, box, percentile))

