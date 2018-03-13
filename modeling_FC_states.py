"""
K-means clustering of the reduced components vectors with silhouette analysis on 
the best number or clusters.

Hidden Markov Model on reduced dim data with log likelihood score to choose the 
best number of n_components.

Katerina Capouskova 2018, kcapouskova@hotmail.com
"""

import os
import numpy as np
from hmmlearn import hmm
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tqdm import tqdm


def kmeans_clustering(reduced_components, output_path):
    """
    Performs a K-means clustering with silhouette analysis

    :param reduced_components: reduced components matrix
    :type reduced_components: np.ndarray
    :param output_path: path to output directory 
    :type output_path: str
    :return: clustered matrix
    :rtype: np.ndarray
    """

    # collapse a matrix into one dimension to prepare for K-means clustering
    flatten_vector = reduced_components.flatten().reshape(-1, 1)
    results = []
    n_clusters = []
    for clusters in tqdm(range(2, 20)):
        n_clusters.append(clusters)
        clusterer = KMeans(n_clusters=clusters)
        cluster_labels = clusterer.fit_predict(flatten_vector)
        # perform the silhouette analysis as a metric for the clustering model
        silhouette_avg = silhouette_score(flatten_vector, cluster_labels)
        print 'For n_clusters =', clusters, 'The average silhouette_score is:',\
            silhouette_avg
        results.append(silhouette_avg)
    # select the best performing number of clusters
    index_of_best = results.index(max(results))
    print 'The best number of clusters:', n_clusters[index_of_best]
    best_clusterer = KMeans(n_clusters=n_clusters[index_of_best])
    clusters = best_clusterer.fit_predict(flatten_vector)
    np.savez(os.path.join(output_path, 'clustered_matrix'), clusters)
    data_clusters = {'flatten_vector': flatten_vector, 'clusters': clusters}
    return clusters


def hidden_markov_model(reduced_components, output_path):
    """
    Performs a Hidden Markov Model with log likelihood scoring

    :param reduced_components: reduced components matrix
    :type reduced_components: np.ndarray
    :param output_path: path to output directory 
    :type output_path: str
    :return: predicted matrix, probabilities matrix
    :rtype: np.ndarray, np.ndarray
    """
    results = []
    n_components = []
    for component in tqdm(range(2,20)):
        n_components.append(component)
        model = hmm.GaussianHMM(n_components=component, covariance_type="full")
        model.fit(reduced_components)
        log_likelihood = model.score(reduced_components)
        print 'For number of components:', component, 'the score is:', \
            log_likelihood
        results.append(log_likelihood)
    index_of_best = results.index(max(results))
    print 'The best number of components:', n_components[index_of_best]
    best_hmm = hmm.GaussianHMM(n_components=n_components[index_of_best],
                               covariance_type="full")
    hidden_states = best_hmm.predict(reduced_components)
    np.savez(os.path.join(output_path, 'HMM_state_sequence'), hidden_states)
    predict_proba = best_hmm.predict_proba(reduced_components)
    np.savez(os.path.join(output_path, 'HMM_posteriors'), predict_proba)
    return hidden_states, predict_proba
