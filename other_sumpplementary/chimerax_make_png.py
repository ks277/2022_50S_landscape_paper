#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 09:48:40 2022

ChimeraX code
make ribosome png
@author: shengkai
"""

session = 1

import chimerax.core.commands.run as run
import os
import numpy as np

ref = '/Users/shengkai/Desktop/paper/deaD/segmentation/mask_box160_percentile_99.0.mrc'
dp = '/Users/shengkai/Desktop/paper/deaD/segmentation/pca_umap/raw_p/'

level_ls = np.load(dp + "level.npy")
sub_list = [i for i in os.listdir(dp) if i.split(".")[-1] == "mrc"]
sub_list.sort()
color = "grey"
for c,i in enumerate(sub_list):
    run(session, "windowsize 1000 1000")
    run(session, "lighting flat")
    run(session, "graphics silhouettes true")
    run(session, "set bgColor white")
    run(session, "open %s"%ref) 
    run(session, "volume #1 step 1 level 0.98 color white transparency 0.8")
    run(session, "open %s/%s"%(dp, i)) 
    run(session, "volume #2 step 1 level %f color %s"%(level_ls[c],color))
    run(session, "surface dust #1 size 26.2")
    run(session, "surface dust #2 size 26.2")
    run(session, "turn y 85")
    run(session, "turn x 160")
    run(session, "view")
    run(session, "save %s/%s.png"%(dp, i.split(".")[0]))
    run(session, "close all")