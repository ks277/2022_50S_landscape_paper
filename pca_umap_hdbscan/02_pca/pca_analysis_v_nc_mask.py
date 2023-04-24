#!/bin/bash

import os
import numpy as np
from sklearn.decomposition import PCA
import mrcfile
import pickle
import sys

n_components = int(sys.argv[1])
mask_name = sys.argv[2]

mrcname = [i for i in os.listdir() if i.split(".")[-1] == "mrc"]
mrcname.sort()

print(mrcname)
csmap = [np.zeros(1) for _ in range(len(mrcname))]

np.save("namelist.npy", mrcname)

print("loading radius mask")
mask = np.load(mask_name)
mask = mask.reshape(-1)

for i, n in enumerate(mrcname):
    print(n)
    with mrcfile.open(n, permissive=True) as mrc:
        csmap[i] = mrc.data.reshape(-1)[mask==1]

csmap = np.array(csmap)
np.save("csmap.npy", csmap)

print("data loaded")
print(len(mrcname))

print(csmap.shape)
print(csmap[0])

#print("loading radius mask")
#r = np.load("radius.npy")

pca_path = PCA(n_components = n_components)

p_path = pca_path.fit_transform(csmap)

print("done PCA, saving models")
np.save("p%i_path.npy"%(n_components), p_path)
#pickle.dump(pca_path, open("pca_path","wb"))

pca_seg = PCA(n_components = n_components)
p_seg = pca_seg.fit_transform(csmap.T)
np.save("p%i_seg.npy"%(n_components), p_seg)

print("finished")
