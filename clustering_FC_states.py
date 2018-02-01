"""
K-means clustering of the PCA components vectors with silhouette analysis on 
the best number or clusters.

Katerina Capouskova 2018, kcapouskova@hotmail.com
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def kmeans_clustering(input_path, output_path):
    """
    Performs a K-means clustering with silhouette analysis

    :param input_path: path to directory with PCA matrix
    :type input_path: str
    :param output_path: path to output directory 
    :type output_path: str
    :return: FCD matrix
    :rtype: np.ndarray
    """

    pca_components = np.genfromtxt(input_path, delimiter=',')
    # collapse a matrix into one dimension to prepare for K-means clustering
    flatten_vector = pca_components.flatten()
    results = []
    n_clusters = []
    for clusters in range(2,20):
        n_clusters.append(clusters)
        clusterer = KMeans(n_clusters=clusters)
        cluster_labels = clusterer.fit_predict(flatten_vector)
        # perform the silhouette analysis as a metric for the clustering model
        silhouette_avg = silhouette_score(flatten_vector, cluster_labels)
        print('For n_clusters =', clusters,
              'The average silhouette_score is :', silhouette_avg)
        results.append(silhouette_avg)
    # select the best performing number of clusters
    max_result = max(results)
    index_of_best = results.index(max_result)
    print('The best number of clusters:', n_clusters[index_of_best])
