import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from unsupervised.utilities import plot3d_combinations
from unsupervised.kmeans import kmeans_clustering
from unsupervised.fuzzy_c_means import fuzzy_clustering, plot_fuzzy_cmeans
from cluster_validity_index import cluster_index
from unsupervised.mountain import mountain_clustering
from unsupervised.subclust import subtractive_clustering


ecg_features = pd.read_csv('features_ecg_personality_traits.csv', index_col=0)

# ----------------------------------------------------------------------------------
# Fuzzy_c_means
#centroids, membership_matrix = fuzzy_clustering(ecg_features, k=7, m =2)
#kmeans = KMeans(n_clusters=7, random_state=0, n_init="auto").fit(X)

# ----------------------------------------------------------------------------------
#K-means clustering
# use kmeans to cluster the data
#centroids, clusters = kmeans_clustering(ecg_features, 5)
# ecg_features['cluster'] = clusters

kmeans = KMeans(n_clusters=7).fit(ecg_features.to_numpy())





#index_kmeans = cluster_index(ecg_features.to_numpy(),kmeans.labels_)
# index_mountain = cluster_index(ecg_features.to_numpy(),clusters_m)
# index_sc = cluster_index(ecg_features.to_numpy(),clusters_sc)
# print(index_sc, index_mountain, index_sc)



