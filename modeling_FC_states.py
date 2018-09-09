"""
K-means clustering of the reduced components vectors with silhouette analysis on 
the best number or clusters.

Hidden Markov Model on reduced dim data with log likelihood score to choose the 
best number of components.

Hidden Markov Model implemented in pomegranate on reduced dim data with log likelihood
score to choose the best number of components.

Katerina Capouskova 2018, kcapouskova@hotmail.com
"""

import logging

import numpy as np
from hmmlearn import hmm
from pomegranate import *
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, silhouette_samples
from tqdm import tqdm

from states_features import probability_of_state, mean_lifetime_of_state
from visualizations import plot_silhouette_analysis


def kmeans_clustering(reduced_components, output_path):
    """
    Performs a K-means clustering with silhouette analysis

    :param reduced_components: reduced components matrix
    :type reduced_components: np.ndarray
    :param output_path: path to output directory 
    :type output_path: str
    :return: clustered array
    :rtype: np.ndarray
    """
    logging.basicConfig(filename=os.path.join(output_path,
                                              'clustering_FC_states.log'),
                        level=logging.INFO)
    samples, timesteps, features = reduced_components.shape
    # new matrix ((time steps x subjects) x number of features(brain areas * n_components))
    reduced_components_2d = np.reshape(reduced_components, (samples * timesteps, features))
    results = []
    n_clusters = []
    for clusters in tqdm(range(2, 3)):
        n_clusters.append(clusters)
        kmeans = KMeans(n_clusters=clusters).fit(reduced_components_2d)
        # perform the silhouette analysis as a metric for the clustering model
        silhouette_avg = silhouette_score(reduced_components_2d, kmeans.labels_,
                                          sample_size=300)
        logging.info('For n_clusters = {}, the average silhouette score is: {}'
                     .format(clusters, silhouette_avg))
        cluster_center, sum_sqr_d = kmeans.cluster_centers_, kmeans.inertia_
        logging.info('For n_clusters = {}, the cluster centers are: {} and the '
                     'sum of squared distances of samples to their closest '
                     'cluster center are: {}'.format(clusters, cluster_center,
                                                     sum_sqr_d))
        results.append(silhouette_avg)
    # select the best performing number of clusters
    index_of_best = results.index(max(results))
    logging.info('The best number of clusters: {}'.format(
        n_clusters[index_of_best]))
    best_clusterer = KMeans(n_clusters=n_clusters[index_of_best])
    clusters_array = best_clusterer.fit_predict(reduced_components_2d)
    np.savez(os.path.join(output_path, 'clustered_matrix'), clusters_array)
    probability_of_state(clusters_array, n_clusters[index_of_best], output_path)
    mean_lifetime_of_state(clusters_array, n_clusters[index_of_best], output_path)
    data_clusters = {'reduced_components': reduced_components_2d, 'clusters': clusters_array}
    return clusters_array


def kmeans_clustering_mean_score(reduced_components, output_path, n_clusters):
    """
    Performs a K-means clustering with pre-set number of clusters more times and
    returns the average silhouette score

    :param reduced_components: reduced components matrix
    :type reduced_components: np.ndarray
    :param output_path: path to output directory
    :type output_path: str
    :param n_clusters: number of clusters
    :type n_clusters: int
    :return: clustered array, features array with labels
    :rtype: np.ndarray, np.ndarray
    """
    logging.basicConfig(filename=os.path.join(output_path,
                                              'clustering_FC_states_mean_score.log'),
                        level=logging.INFO)
    results = []
    for iter in tqdm(range(1)):
        kmeans = KMeans(n_clusters=n_clusters).fit(reduced_components)
        # perform the silhouette analysis as a metric for the clustering model
        silhouette = silhouette_score(reduced_components, kmeans.labels_,
                                      sample_size=500)
        logging.info('For iteration = {}, the average silhouette score is: {}'
                     .format(iter, silhouette))
        cluster_center, sum_sqr_d = kmeans.cluster_centers_, kmeans.inertia_
        logging.info('For iteration = {}, the cluster centers are: {} and the '
                     'sum of squared distances of samples to their closest '
                     'cluster center are: {}'.format(iter, cluster_center,
                                                     sum_sqr_d))
        results.append(silhouette)
        clusters_array = kmeans.predict(reduced_components)
        sample_silhouette_values = silhouette_samples(reduced_components,
                                                      clusters_array)
    # average silhouette score
    avg_silhouette = sum(results)/float(len(results))
    logging.info('The average silhouette score: {}'.format(avg_silhouette))
    np.savez(os.path.join(output_path, 'clustered_matrix'), clusters_array)
    probability_of_state(clusters_array, n_clusters, output_path)
    mean_lifetime_of_state(clusters_array, n_clusters, output_path)
    plot_silhouette_analysis(reduced_components, output_path, n_clusters, avg_silhouette,
                             sample_silhouette_values, clusters_array, cluster_center)
    clusters_array = np.expand_dims(clusters_array, axis=1)
    data_clusters = np.hstack((reduced_components, clusters_array))
    np.savez(os.path.join(output_path, 'concatentated_matrix_clusters'),
             data_clusters)
    return clusters_array, data_clusters


def hidden_markov_model(reduced_components, output_path):
    """
    Performs a Hidden Markov Model with log likelihood scoring

    :param reduced_components: reduced components matrix
    :type reduced_components: np.ndarray
    :param output_path: path to output directory 
    :type output_path: str
    :return: predicted matrix, probabilities matrix, number of components, 
    markov array
    :rtype: np.ndarray, np.ndarray, int, np.ndarray
    """
    logging.basicConfig(
        filename=os.path.join(output_path, 'hmm_modeling_FC_states.log'),
        level=logging.INFO)
    samples, timesteps, features = reduced_components.shape
    components_swapped = np.swapaxes(reduced_components, 0, 1)
    reduced_components_2d = np.reshape(components_swapped, (timesteps, (samples*features)))
    results = []
    n_components = []
    for component in tqdm(range(100, 300, 4)):
        n_components.append(component)
        model = hmm.GaussianHMM(n_components=component)
        model.fit(reduced_components_2d)
        log_likelihood = model.score(reduced_components_2d)
        logging.info('For number of components: {}, the score is: {}'.format(
            component, log_likelihood))
        results.append(log_likelihood)
    index_of_best = results.index(max(results))
    logging.info('The best number of components: {}'.format(
        n_components[index_of_best]))
    best_hmm = hmm.GaussianHMM(n_components=n_components[index_of_best])
    best_hmm.fit(reduced_components_2d)
    hidden_states = best_hmm.predict(reduced_components_2d)
    np.savez(os.path.join(output_path, 'HMM_state_sequence'), hidden_states)
    predict_proba = best_hmm.predict_proba(reduced_components_2d)
    np.savez(os.path.join(output_path, 'HMM_posteriors'), predict_proba)
    hidden_states_expanded = np.expand_dims(hidden_states, axis=1)
    markov_array = np.append(reduced_components_2d, hidden_states_expanded, axis=1)
    return hidden_states, predict_proba, n_components[index_of_best], \
           markov_array


def hidden_markov_model_pomegranate(reduced_components, output_path):
    """
    Performs a Hidden Markov Model with log likelihood scoring

    :param reduced_components: reduced components matrix
    :type reduced_components: np.ndarray
    :param output_path: path to output directory 
    :type output_path: str
    :return: predicted matrix, probabilities matrix, number of components, 
    markov array
    :rtype: np.ndarray, np.ndarray, int, np.ndarray
    """
    logging.basicConfig(
        filename=os.path.join(output_path, 'pom_hmm_modeling_FC_states.log'),
        level=logging.INFO)
    samples, timesteps, features = reduced_components.shape
    components_swapped = np.swapaxes(reduced_components, 0, 1)
    reduced_components_2d = np.reshape(components_swapped, (timesteps, (samples*features)))
    sample_array = random.sample(reduced_components_2d, 50)
    results = []
    n_components = []
    for component in tqdm(range(2, 20, 1)):
        n_components.append(component)
        model = HiddenMarkovModel.from_samples(NormalDistribution,
                                               n_components=component,
                                               X=reduced_components_2d)
        model.fit(reduced_components_2d, algorithm='baum-welch')
        accuracy = model.log_probability(sample_array)
        logging.info('For number of components: {}, the score is: {}'.format(
            component, accuracy))
        results.append(accuracy)
    index_of_best = results.index(max(results))
    logging.info('The best number of components: {}'.format(
        n_components[index_of_best]))
    best_hmm = HiddenMarkovModel.from_samples(NormalDistribution,
                                              n_components=n_components[index_of_best],
                                              X=reduced_components_2d)
    best_hmm.fit(reduced_components_2d, algorithm='viterbi')
    hidden_states = best_hmm.predict(reduced_components_2d)
    np.savez(os.path.join(output_path, 'HMM_state_sequence'), hidden_states)
    predict_proba = best_hmm.predict_proba(reduced_components_2d)
    np.savez(os.path.join(output_path, 'HMM_posteriors'), predict_proba)
    hidden_states_expanded = np.expand_dims(hidden_states, axis=1)
    markov_array = np.append(reduced_components_2d, hidden_states_expanded, axis=1)
    return hidden_states, predict_proba, n_components[index_of_best], \
           markov_array


def kmeans_clustering_missing(reduced_components, output_path,
                              n_clusters=2, max_iter=10):
    """
    Performs a K-means clustering with missing data.

    :param reduced_components: reduced components matrix
    :type reduced_components: np.ndarray
    :param output_path: path to output directory
    :type output_path: str
    :param n_clusters: number of clusters
    :type n_clusters: int
    :param max_iter: maximum iterations for convergence
    :type max_iter: int
    :return: clustered array, centroids of clusters, filled matrix
    :rtype: np.ndarray, list, np.ndarray
    """
    logging.basicConfig(filename=os.path.join(output_path,
                                              'clustering_FC_states_missing.log'),
                        level=logging.INFO)
    # Initialize missing values to their column means
    missing = ~np.isfinite(reduced_components)
    mu = np.nanmean(reduced_components, axis=0)
    X_filled = np.where(missing, mu, reduced_components)

    for i in tqdm(range(max_iter)):
        if i > 0:
            # k means with previous centroids
            cls = KMeans(n_clusters, init=prev_centroids)
        else:
            # do multiple random initializations in parallel
            cls = KMeans(n_clusters, n_jobs=-1)
        # perform clustering on the filled-in data
        labels = cls.fit_predict(X_filled)
        centroids = cls.cluster_centers_
        # fill in the missing values based on their cluster centroids
        X_filled[missing] = centroids[labels][missing]
        # when the labels have stopped changing then we have converged
        if i > 0 and np.all(labels == prev_labels):
            break

        prev_labels = labels
        prev_centroids = cls.cluster_centers_
        # perform the silhouette analysis as a metric for the clustering model
    silhouette_avg = silhouette_score(X_filled, cls.labels_,
                                          sample_size=300)
    logging.info('For n_clusters = {}, the average silhouette score is: {}'
                     .format(n_clusters, silhouette_avg))
    logging.info('For n_clusters = {}, the cluster centers are: {} and the '
                     'sum of squared distances of samples to their closest '
                     'cluster center are: {}'.format(n_clusters, centroids,
                                                     cls.inertia_))
    np.savez(os.path.join(output_path, 'clustered_matrix'), labels)
    return labels, centroids, X_filled


def dbscan(reduced_components, output_path):
    """
    Performs a K-means clustering with pre-set number of clusters more times and
    returns the average silhouette score

    :param reduced_components: reduced components matrix
    :type reduced_components: np.ndarray
    :param output_path: path to output directory
    :type output_path: str
    :return: clustered array, features array with labels
    :rtype: np.ndarray, np.ndarray
    """
    logging.basicConfig(filename=os.path.join(output_path,
                                              'clustering_FC_states_mean_score.log'),
                        level=logging.INFO)

    dbscan, labels = DBSCAN(eps=0.3, min_samples=10).fit_predict(reduced_components)
    # perform the silhouette analysis as a metric for the clustering model
    silhouette = silhouette_score(reduced_components, labels,
                                  sample_size=500)
    # Number of clusters in labels, ignoring noise if present.
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    logging.info('For n clusters = {}, the silhouette score is: {}'
                 .format(n_clusters, silhouette))
    core_samples = dbscan.core_sample_indices_
    logging.info('Indices of core samples'.format(core_samples))
    sample_silhouette_values = silhouette_samples(reduced_components, labels)
    np.savez(os.path.join(output_path, 'clustered_matrix'), labels)
    probability_of_state(labels, n_clusters, output_path)
    mean_lifetime_of_state(labels, n_clusters, output_path)
    plot_silhouette_analysis(reduced_components, output_path, n_clusters, silhouette,
                             sample_silhouette_values, labels, dbscan.components_)
    clusters_array = np.expand_dims(labels, axis=1)
    data_clusters = np.hstack((reduced_components, clusters_array))
    np.savez(os.path.join(output_path, 'concatentated_matrix_clusters'),
             data_clusters)
    return labels, data_clusters
