# 2022_50S_landscape_paper

Needs packages:
numpy-1.20.1
pandas-1.2.4
mrcfile-1.3.0
umap-0.5.1
sklearn-0.24.1
networkx-2.5
hdbscan
pickle


To get example results:

Please run the following command line in the example files in each directory: (Linux, MacOS)

01_consensus_voxel_identification
box_size = 160
percentile = 98.5
cloud_size = 0
radius_mask = 60
consensus_number = 1

python ../percentile_mask_v_box_perc_cl_apix_r_con.py 160 98.5 0 60 1

02_pca
number_of_pricinple_components = 5
mask_array = mask_box160_percentile_98.5.npy

python ../pca_analysis_v_nc_mask.py 5 mask_box160_percentile_98.5.npy

03_umap
input_principle_components = p21_seg.npy
n_components = 2
nearest_neighbor = 100
metric = canberra

python ../umap_seg_pca_v_ncom_NN_metric.py p21_seg.npy 2 100 canberra

04_hdbscan_training
input_umap = p21_seg_umap_canberra_ncom2_NN100.npy

python ../hdbscan_training.py p21_seg_umap_canberra_ncom2_NN100.npy 

05_segmentation_based_on_hdbscan_model
input umap = p21_seg_umap_canberra_ncom2_NN100.npy
hdbscan_model = hdbscan_150_150
box size = 160
pixel size = 2.62
segments tag = test
mask = mask_box160_percentile_99.0.mrc

python ../seg_hdbscan.py p21_seg_umap_canberra_ncom2_NN100.npy hdbscan_150_150 160 2.62 test mask_box160_percentile_99.0.mrc


The raw codes are stored in raw_script directory
