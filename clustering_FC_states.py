"""
K-means clustering of the PCA components vectors with silhouette analysis on 
the best number or clusters.

Katerina Capouskova 2018, kcapouskova@hotmail.com
"""

import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tqdm import tqdm


def kmeans_clustering(pca_components, output_path):
    """
    Performs a K-means clustering with silhouette analysis

    :param pca_components: PCA matrix
    :type pca_components: np.ndarray
    :param output_path: path to output directory 
    :type output_path: str
    :return: clustered matrix
    :rtype: np.ndarray
    """

    # collapse a matrix into one dimension to prepare for K-means clustering
    flatten_vector = pca_components.flatten().reshape(-1, 1)
    results = []
    n_clusters = []
    for clusters in tqdm(range(2, 20)):
        n_clusters.append(clusters)
        clusterer = KMeans(n_clusters=clusters)
        cluster_labels = clusterer.fit_predict(flatten_vector)
        # perform the silhouette analysis as a metric for the clustering model
        silhouette_avg = silhouette_score(flatten_vector, cluster_labels)
        print 'For n_clusters =', clusters, 'The average silhouette_score is :',\
            silhouette_avg
        results.append(silhouette_avg)
    # select the best performing number of clusters
    index_of_best = results.index(max(results))
    print 'The best number of clusters:', n_clusters[index_of_best]
    best_clusterer = KMeans(n_clusters=n_clusters[index_of_best])
    clusters = best_clusterer.fit_predict(flatten_vector)
    np.savez(os.path.join(output_path, 'clustered_matrix'), clusters)
    data_clusters = {'flatten_vector': flatten_vector, 'clusters': clusters}
    cluster_df = pd.DataFrame(data_clusters)
    return clusters
