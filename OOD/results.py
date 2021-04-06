# Computing metrics and results

# python results.py -f features/v1


import numpy as np
import pandas as pd
import glob,argparse
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier,NearestNeighbors,LocalOutlierFactor
import pathlib
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from re_rank import re_ranking

from results_gram import detect_mean


# Parsing arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--feature_path", required=True,
	help="path to a folder containing Training data, Testing In-distribution data and Testing OOD data features")

args = vars(ap.parse_args())

# Training set features
root = args['feature_path']

class_names = ["gametocyte","ring","schizont","trophozoite"]
all_features,all_labels = [],[]

for index_cls,cls in enumerate(class_names):
    print (cls)
    for feat in glob.glob(root+"/"+cls+"/*.npy"):
        all_features.append(np.load(feat))
        all_labels.append(index_cls)
    
all_features = np.array(all_features).reshape(len(all_features),256)

neigh = KNeighborsClassifier(n_neighbors=5,metric="cosine")
neigh.fit(all_features,all_labels)

print ("Training features: ",all_features.shape)

test_indist_features = []
for feat in glob.glob(root+"/test_id/*"):
    test_indist_features.append(np.load(feat))

test_indist_features = np.array(test_indist_features).reshape(len(test_indist_features),256)
print ("Testing ID features: ",test_indist_features.shape)


d = re_ranking(test_indist_features,all_features,15,6,lambda_value=0.3)
min_distances_indist = np.min(d,1)
min_distances_indist_median = np.median(np.sort(d,1)[:,:7],1)

results = []

class_names = ["bbox","bbox_70","red_blood_cell","coin_fusion","flag_fusion","imagenet_crops","nct_crops"]
for cls in class_names:
    all_features_ood = []
    print (cls)
    for feat in glob.glob(root+"/"+cls+"/*.npy"):    
        all_features_ood.append(np.load(feat))

    test_ood_features = np.array(all_features_ood).reshape(len(all_features_ood),256)
    d_ood = re_ranking(test_ood_features,all_features,15,6,lambda_value=0.3)
    
    min_distances_ood = np.min(d_ood,1)
    min_distances_ood_median = np.median(np.sort(d_ood,1)[:,:7],1)
    
    print ("1 Nearest neighbor")
    detect_mean(min_distances_indist,min_distances_ood)

    print ("median Nearest neighbor")
    detect_mean(min_distances_indist_median,min_distances_ood_median)