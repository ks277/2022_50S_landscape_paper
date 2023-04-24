#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 07:10:50 2022

@author: jrwill
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from matplotlib.textpath import TextPath
from matplotlib.patches import PathPatch
from matplotlib.font_manager import FontProperties
from matplotlib.transforms import Affine2D
from scipy.cluster import hierarchy
import sys
import csv
import argparse


def PaintPixel(value,npixels = 40,value_hue = 0.62, border_col = 'w'):
    hsv = np.zeros((npixels,npixels,3))
    hsv[...,0] = value_hue # blue center
    hsv[...,1] = value
    hsv[...,2] = 1
    if border_col == 'w':
        border_hue = [0,0,1]
    else:
        border_hue = [0.16,1,1]
    hsv[0,:] = border_hue # yellow border
    hsv[-1,:] = border_hue
    hsv[:,0] = border_hue
    hsv[:,-1] = border_hue
    rgb = hsv_to_rgb(hsv)
    return(rgb)

def ClusterData(mat):
    clust = hierarchy.linkage(mat, 'ward')
    dendro = hierarchy.dendrogram(clust, no_plot = True)
    xdendro = np.array(dendro["icoord"])
    ydendro = np.array(dendro["dcoord"])
    order = [int(l) for l in dendro["ivl"]]
    return(xdendro,ydendro,order)

def ReadCSVmatrix(csvfile):
    data = np.array(list(csv.reader(open(csvfile))))
    col_labels = list(data[0,1:])
    row_labels = list(data[1:,0])
    mat =np.asarray(data[1:,1:], dtype=float)
    return(mat,row_labels,col_labels)



parser = argparse.ArgumentParser(description='Occupancy Matrix Clustering Arguments')
parser.add_argument('csvfile', help= 'csv file containing occupancy matrix')
parser.add_argument('--no_cluster_rows', dest='cluster_rows', action='store_false', help= ' Flag to cluster Rows, default = True')
parser.add_argument('--no_cluster_cols', dest='cluster_cols', action='store_false', help= ' Flag to cluster Columns, default = True')
parser.add_argument('--no_log_row_dendro', dest='log_row_dendro', action='store_false', help= ' Flag to take log2 of row dendrogram, default = True')
parser.add_argument('--no_log_col_dendro', dest='log_col_dendro', action='store_false', help= ' Flag to take log2 of column dendrogram, default = True')
parser.add_argument('--hide_row_labels', dest='show_row_labels', action='store_false', help= ' Flag to show row labels, default = True')
parser.add_argument('--hide_col_labels', dest='show_col_labels', action='store_false', help= ' Flag to show column labels, default = True')
parser.add_argument('--xsize', dest='xsize', default = 8,type = float, help= 'X size of output plot in inches, default = 8')
parser.add_argument('--ysize', dest='ysize', default = 8,type = float, help= 'Y size of output plot in inches, default = 8')
parser.add_argument('--pixel_size', dest='pixel_size', default = 40, type = int, help= 'number of pixels per element of the occupancy matrix, default = 40')
parser.add_argument('--dendro_linewidth', dest='dendro_linewidth', default = 1.0, type = float, help= 'linewidth to use for drawing dendrograms, default = 1.0')
parser.add_argument('--font', dest='font', default = 'Helvetica', help= 'font for dendrogram labels, default = ''Helvetica')
parser.add_argument('--font_path_width', dest='font_path_width', default = 0.02, type = float, help= 'linewidth to use for drawing dendrograms, default = 0.02')
parser.add_argument('--row_label_x_offset', dest='row_label_x_offset', default = 0.2, type = float, help= 'x offset for labels in fraction of row, default = 0.2')
parser.add_argument('--row_label_y_offset', dest='row_label_y_offset', default = 0.3, type = float, help= 'y offset for labels in fraction of row, default = 0.3')
parser.add_argument('--col_label_x_offset', dest='col_label_x_offset', default = 0.3, type = float, help= 'x offset for labels in fraction of column, default = 0.3')
parser.add_argument('--col_label_y_offset', dest='col_label_y_offset', default = -0.2, type = float, help= 'y offset for labels in fraction of column, default = -0.2')
parser.add_argument('--row_label_size', dest='row_label_size', default = 0.6, type = float, help= 'row label size as fraction of row, default = 0.6')
parser.add_argument('--col_label_size', dest='col_label_size', default = 0.6, type = float, help= 'column label size as fraction of column, default = 0.6')
parser.add_argument('--no_pdf', dest = 'no_pdf', action = 'store_true', help = ' Flag to supress pdf output, default = True')
parser.add_argument('--hue', dest='hue', default = 0.62, type = float, help= 'hue (for HSV) for pixels, default = 0.62 (blue)')
parser.add_argument('--border_color', dest='border_color', default = 'y', type = str, help= 'border color between pixels (y or w), default = y (yellow)')


args = parser.parse_args()

# csvfile = args.csvfile
occmat, row_labels_in, col_labels_in = ReadCSVmatrix(args.csvfile)
nr, nc = occmat.shape

# dendrogram options

cluster_rows = args.cluster_rows
cluster_cols = args.cluster_cols
log_row_dendro = args.log_row_dendro
log_col_dendro = args.log_col_dendro
show_row_labels = args.show_row_labels
show_col_labels = args.show_col_labels

# PLOT TWEAKS

plotscale = args.pixel_size  # number of pixels for each occmat element
dendro_width = args.dendro_linewidth

# text rendered as path so it scales with plot size
fp = FontProperties(family=args.font) # plot label font
font_path_width = args.font_path_width    # stroke width controls font weight

# row labels are particle_labels
# right edge of occmat image is nc
row_label_x_offset = args.row_label_x_offset # fraction of an occmat square
row_label_y_offset = args.row_label_y_offset
row_label_size = args.row_label_size   # font size is fraction of occmat square

# col labels are helix_labels
# bottom edge of occmat image is 0
col_label_x_offset = args.col_label_x_offset # fraction of an occmat square
col_label_y_offset = args.col_label_y_offset
col_label_size = args.col_label_size

# set up plot
fig, ax = plt.subplots(figsize = (nc,nr))  # plot internal coords are 1 unit/row-col
fig.set_size_inches(args.xsize,args.ysize) # absolute plot size for display in inches
plt.rcParams["figure.autolayout"] = True
plt.axis('off')

# build occmat image
imgmat = np.zeros((nr,nc,plotscale,plotscale,3))
for row in range(nr):
    for col in range(nc):
        imgmat[row,col] = PaintPixel(occmat[row,col], npixels = plotscale, value_hue =args.hue, border_col = args.border_color)

# clustering 
if cluster_rows:
    print("clustering rows")
    row_yscale = 0.1  # dendrograms from scipy cluster are scaled by 10X
    row_xscale = 0.1
    row_dendro_x, row_dendro_y, row_order = ClusterData(occmat)  # row dendrogram
    if log_row_dendro:
        row_dendro_y = np.log2(row_dendro_y + 1) # log transform  
        row_yscale = float(np.log2(nc))/np.amax(row_dendro_y)
    row_dendro_coords = np.array([-row_dendro_y * row_yscale, row_dendro_x * row_xscale]).transpose(1,0,2)
    row_nd,npt,ns = row_dendro_coords.shape

    # shuffle labels to dendrogram order
    row_labels = []
    for i in range(nr):
        row_labels.append(row_labels_in[row_order[i]])
    row_labels.reverse()
    print(row_labels)
    imgmat = imgmat[row_order,:,:,:,:] # shuffle row order
    
else:
    row_labels = row_labels_in
    row_labels_in.reverse()
    
if cluster_cols:
    col_yscale = 0.1
    col_xscale = 0.1
    col_dendro_x, col_dendro_y, col_order = ClusterData(occmat.transpose()) # column dendrogram
    if log_col_dendro:
        col_dendro_y = np.log2(col_dendro_y+1)
        col_yscale = float(np.log2(nr)/np.amax(col_dendro_y)) 
    col_dendro_coords = np.array([col_dendro_x * col_xscale, nr + col_dendro_y * col_yscale]).transpose(1,0,2)
    col_nd,npt,ns = col_dendro_coords.shape
    col_labels = []
    for i in range(nc):
        col_labels.append(col_labels_in[col_order[i]])
    imgmat = imgmat.transpose(1,0,2,3,4)[col_order,:,:,:,:].transpose(1,0,2,3,4) # shuffle column order
 
else:
    col_labels = col_labels_in

# reshape array of images (nx,ny,plotscale,plotscale,3) to single image (nx*plotscale,ny*plotscale,3)
imgmat = imgmat.transpose([0,2,1,3,4])
imgmat = imgmat.reshape(nr*plotscale,nc*plotscale,3)

# display occmat
im = ax.imshow(imgmat, extent=[0, nc, 0, nr])

# plot dendrogram segments
if cluster_cols:
    for i in range(col_nd):
        ax.plot(col_dendro_coords[i,0,:],col_dendro_coords[i,1,:], color = 'k', lw = dendro_width)
    
if cluster_rows:
    for i in range(row_nd):
        ax.plot(row_dendro_coords[i,0,:],row_dendro_coords[i,1,:], color = 'k', lw = dendro_width)

# plot labels

if show_row_labels:
    for i in range(nr):
        labx = nc + row_label_x_offset
        laby = i + row_label_y_offset
        tp = TextPath((labx , laby), str(row_labels[i]), size=row_label_size, prop = fp)
        ax.add_patch(PathPatch(tp, color="black",lw = font_path_width))
    
if show_col_labels:
    for i in range(nc):
        labx = i + col_label_x_offset
        laby = col_label_y_offset
        tp = TextPath((labx, laby), str(col_labels[i]), size=col_label_size, prop = fp)
        rtp = Affine2D().rotate_around(labx,laby,-np.pi/2).transform_path(tp) # rotate label
        ax.add_patch(PathPatch(rtp, color="black",lw = font_path_width))

print("args.no_pdf",args.no_pdf)
if args.no_pdf is False:
    pdffile = args.csvfile.split(".")[0] + ".pdf"
    print(pdffile)
    fig.savefig(pdffile, format='pdf', dpi=300)

plt.show()



    
