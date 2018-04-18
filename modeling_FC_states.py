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
import os
import numpy as np

from pomegranate import *
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
    logging.basicConfig(filename=os.path.join(output_path, 'clustering_FC_states.log'),
                        level=logging.INFO)
    # collapse a matrix into 2 dimensions to prepare for K-means clustering
    samples, timesteps, features = reduced_components.shape
    components_swapped = np.swapaxes(reduced_components, 0, 1)
    # new matrix (time steps x number of features (subjects x brain areas))
    reduced_components_2d = np.reshape(components_swapped, (timesteps, (samples*features)))
    results = []
    n_clusters = []
    for clusters in tqdm(range(39, 80)):
        n_clusters.append(clusters)
        clusterer = KMeans(n_clusters=clusters)
        cluster_labels = clusterer.fit_predict(reduced_components_2d)
        # perform the silhouette analysis as a metric for the clustering model
        silhouette_avg = silhouette_score(reduced_components_2d, cluster_labels, sample_size=300)
        logging.info('For n_clusters = {}, the average silhouette score is: {}'
                     .format(clusters, silhouette_avg))
        results.append(silhouette_avg)
    # select the best performing number of clusters
    index_of_best = results.index(max(results))
    logging.info('The best number of clusters: {}'.format(
        n_clusters[index_of_best]))
    best_clusterer = KMeans(n_clusters=n_clusters[index_of_best])
    clusters = best_clusterer.fit_predict(reduced_components_2d)
    np.savez(os.path.join(output_path, 'clustered_matrix'), clusters)
    data_clusters = {'reduced_components_2d': reduced_components_2d, 'clusters': clusters}
    return clusters


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
