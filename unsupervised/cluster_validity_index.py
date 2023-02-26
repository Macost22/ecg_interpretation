import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score, silhouette_score
from unsupervised.kmeans import kmeans_clustering
import matplotlib.pyplot as plt



def cluster_index(data, clusters):
    davies_bouldin = davies_bouldin_score(data, clusters)
    calinski_harabasz = calinski_harabasz_score(data, clusters)
    silhouette = silhouette_score(data, clusters)
    return {'davies_bouldin_score': davies_bouldin, 'calinski_harabasz_score': calinski_harabasz, 'silhouette_score': silhouette}


def plot_score(data):
    sil = []
    db = []
    ch = []
    kmax = 10
    ks=[]
    # dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be 2
    for k in range(2, kmax+1):
      ks.append(k)
      centroids, clusters = kmeans_clustering(data.to_numpy, k)
      labels = clusters
      sil.append(silhouette_score(data.to_numpy(), labels, metric = 'euclidean'))
      db.append(davies_bouldin_score(data.to_numpy(), labels))
      ch.append(calinski_harabasz_score(data.to_numpy(), labels))

    plt.plot(ks, sil, label='Silhouette')
    plt.scatter(ks[np.argmax(sil)],sil[np.argmax(sil)])
    plt.plot(ks,db, label='Davies Bouldin')
    plt.scatter(ks[np.argmin(db)],db[np.argmin(db)])
    plt.title('Mejor k utilizando unsupervised')
    plt.xlabel('k')
    plt.ylabel('score')
    plt.legend()
    plt.show()


    plt.title('Mejor k utilizando unsupervised')
    plt.plot(ks, ch, label='Calinski harabasz')
    plt.scatter(ks[np.argmax(ch)],ch[np.argmax(ch)])
    plt.xlabel('k')
    plt.ylabel('score')
    plt.legend()
    plt.show()


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


