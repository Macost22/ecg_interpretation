import pandas as pd
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score, silhouette_score
from unsupervised.kmeans import kmeans_clustering



def cluster_index(data, clusters):
    davies_bouldin = davies_bouldin_score(data, clusters)
    calinski_harabasz = calinski_harabasz_score(data, clusters)
    silhouette = silhouette_score(data, clusters)
    return {'davies_bouldin_score': davies_bouldin, 'calinski_harabasz_score': calinski_harabasz, 'silhouette_score': silhouette}


if __name__ == "__main__":

    X = datasets.load_iris().data

    # K-Means con sklearn
    k_means = KMeans(n_clusters=3)
    k_means.fit(X) # K-means training
    labels = k_means.predict(X)

    # K-Means santiago
    centroids, clusters = kmeans_clustering(pd.DataFrame(X), 3)

    # print(davies_bouldin_score(X, labels))
    index_sk = cluster_index(X,labels)
    index_s = cluster_index(X,clusters)


