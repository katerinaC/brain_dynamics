"""
Functions for identifying different states features.

Katerina Capouskova 2018, kcapouskova@hotmail.com
"""
import json
import logging
import os

import scipy
import pandas as pd
import numpy as np
from permute.core import two_sample
from scipy.spatial.distance import mahalanobis
from scipy.stats import stats

from utilities import create_dir


def probability_of_state(clusters, n_clusters, output_path):
    """
    Computes the probability of states.

    :param clusters: clusters array
    :type clusters: np.ndarray
    :param n_clusters: number of clusters
    :type n_clusters: int
    :param output_path: path to output directory
    :type output_path: str
    :return: dictinory {state: probability}
    :rtype: dict
    """
    logging.basicConfig(
        filename=os.path.join(output_path, 'probability_of_states.log'),
        level=logging.INFO)
    dict_p = {}
    for n in range(n_clusters):
        n_list = [c for c in clusters if c == n]
        p = float(len(n_list))/float(len(clusters))
        dict_p.update({n: p})
        logging.info('Probability of state: {} is: {}'.format(n, p))
    with open(os.path.join(output_path, 'probability.json'), 'w') as fp:
        json.dump(dict_p, fp)
    return dict_p


def mean_lifetime_of_state(clusters, n_clusters, output_path, TR):
    """
    Computes the mean lifetime of states.

    :param clusters: clusters array
    :type clusters: np.ndarray
    :param n_clusters: number of clusters
    :type n_clusters: int
    :param output_path: path to output directory
    :type output_path: str
    :param TR: imaging repetition time in seconds
    :type TR: int
    :return: dictinory {state: mean lifetime}
    :rtype: dict
    """
    dict_lt = {}
    logging.basicConfig(
        filename=os.path.join(output_path, 'mean_lifetime_of_states.log'),
        level=logging.INFO)
    for n in range(n_clusters):
        state_true = []
        for c in clusters:
            if c == n:
                state_true.append(1)
            else:
                state_true.append(0)
        # create differences list
        diff_list = np.diff(state_true).tolist()
        # detect swithces in and out of states
        out_state = [i for i, j in enumerate(diff_list, 1) if j == 1]
        in_state = [i for i, j in enumerate(diff_list, 1) if j == -1]
        # discard cases where state starts or ends
        if len(in_state) > len(out_state):
            in_state.pop(0)
        elif len(out_state) > len(in_state):
            out_state.pop(-1)
        elif out_state and in_state and out_state[0] > in_state[0]:
            in_state.pop(0)
            out_state.pop(-1)
        else:
            pass
        if out_state and in_state:
            # minus two lists
            c_duration = []
            for i, z in zip(in_state, out_state):
                diff = i - z
                c_duration.append(diff)
        else:
            c_duration = 0
        mean_duration = np.mean(c_duration) * TR
        logging.info('Mean lifetime of state: {} is: {}'.format(n, mean_duration))
        dict_lt.update({n: mean_duration})
    with open(os.path.join(output_path, 'mean_state_lifetime.json'), 'w') as fp:
        json.dump(dict_lt, fp)
    return dict_lt


def students_t_test(group_a, group_b, output_path):
    """
    Computes a student's t-test and p value for two groups.

    :param group_a: clusters array of a first group
    :type group_a: np.ndarray
    :param group_b: clusters array of a second group
    :type group_b: np.ndarray
    :param output_path: path to output directory
    :type output_path: str
    """
    create_dir(output_path)
    logging.basicConfig(
        filename=os.path.join(output_path, 'students_t_test.log'),
        level=logging.INFO)
    t, p = stats.ttest_ind(group_a, group_b)
    logging.info('T-test value: {}, p-value: {}'.format(t, p))
    dict = {'T-test value': t, 'p-value': p}
    with open(os.path.join(output_path, 'students_t_test.json'), 'w') as fp:
        json.dump(dict, fp)
    return t, p


def permutation_t_test(group_a, group_b, output_path):
    """
    Computes a permutation test based on a t-statistic. Returns and t value,
    p value, a H0 for two groups.

    :param group_a: clusters array of a first group
    :type group_a: np.ndarray
    :param group_b: clusters array of a second group
    :type group_b: np.ndarray
    :param output_path: path to output directory
    :type output_path: str
    """
    create_dir(output_path)
    logging.basicConfig(
        filename=os.path.join(output_path, 'permutation_t_test.log'),
        level=logging.INFO)
    p, t = two_sample(group_a, group_b, reps=5000, stat='t',
                      alternative='two-sided', seed=20)
    logging.info('Permutation T-test value: {}, p-value: {}'.format(t, p))
    dict = {'Permutation T-test value': t, 'p-value': p}
    with open(os.path.join(output_path, 'permutation_t_test.json'), 'w') as fp:
        json.dump(dict, fp)
    return p, t


def p_value_stars(p_value):
    """
    Returns the appropriate number of stars for the p_value

    :param p_value: p_value
    :type p_value: int
    :param output_path: path to output directory
    :type output_path: str
    """
    if p_value < 0.0001:
        return "****"
    elif (p_value < 0.001):
        return "***"
    elif (p_value < 0.01):
        return "**"
    elif (p_value < 0.05):
        return "*"
    else:
        return "-"


def distribution_probability_lifetime(clusters, output_path, n_clusters, TR):
    """
    Creates a distribution of probabilities and life times of states

    :param clusters: clusters array
    :type clusters: np.ndarray
    :param output_path: path to output directory
    :type output_path: str
    :param n_clusters: number of clusters
    :type n_clusters: int
    :param TR: imaging repetition time in seconds
    :type TR: int
    :return: proba_list:array of probabilities, lifetimes_list: array of lifetimes,
    df: dataframe
    :rtype: np.ndarray, np.ndarray, pd.DataFrame
    """
    probas = probability_of_state(clusters, n_clusters, output_path)
    lifetimes = mean_lifetime_of_state(clusters, n_clusters, output_path, TR)
    proba_list = []
    lifetimes_list = []
    for elem in clusters:
        proba_list.append(probas[elem])
        lifetimes_list.append(lifetimes[elem])
    probas_dict = {'clusters': clusters, 'probabilities': proba_list}
    df = pd.DataFrame(data=probas_dict)
    return np.array(proba_list), np.array(lifetimes_list), df


def variance_of_states(reduced_components, output_path):
    """
    Computes the variance in states

    :param reduced_components: array with reduced components and labels
    predicted from clustering
    :type reduced_components: np.ndarray
    :param output_path: path to output directory
    :type output_path: str
    :return: variance: array of variances
    :rtype: np.ndarray
    """
    variance = np.var(reduced_components, axis=1)
    np.savez(os.path.join(output_path, 'variance'), variance)
    return variance


def entropy_of_states(probabilities, output_path, n_cluster):
    """
    Computes the entropy of probabilities of states

    :param probabilities: array with states probabilities
    :type probabilities: np.ndarray
    :param output_path: path to output directory
    :type output_path: str
    :param n_cluster: number of the cluster
    :type: n_cluster: int
    :return: entropy: calculated entropy
    :rtype: int
    """
    entropy = scipy.stats.entropy(probabilities)
    dict = {'State': n_cluster, 'entropy': entropy}
    with open(os.path.join(output_path, 'entropy_n_cluster_{}.json'.format(
            n_cluster)), 'w') as fp:
        json.dump(dict, fp)
    return entropy


def kl_distance_symm(prob_a, prob_b):
    """
    Computes the Kullback-Leibler (KL) divergence and symmetrize it.

    :param prob_a: array with probabilities of condition a
    :type prob_a: pd.Series
    :param prob_b: array with probabilities of condition b
    :type prob_b: pd.Series
    :return: kl divergence: calculated kl divergence symmetrized
    :rtype: float
    """
    if not isinstance(prob_a, np.ndarray):
        prob_a = np.array(prob_a.tolist())
    if not isinstance(prob_b, np.ndarray):
        prob_b = np.array(prob_b.tolist())
    # Epsilon is used here to avoid conditional code for
    # checking that neither P nor Q is equal to 0.
    epsilon = 0.00001

    P = prob_a + epsilon
    Q = prob_b + epsilon

    kl_ab = np.sum(P * np.log(P/Q))
    kl_ba = np.sum(Q * np.log(Q/P))

    return (kl_ab + kl_ba) / 2


def transition_matrix(states, condition, output_path):
    """
    Computes the transition matrix.

    :param states: array with states
    :type states: []
    :param condition: condition
    :type: str
    :param output_path: path to output directory
    :type output_path: str
    :return: M: Markov transitions matrix
    :rtype: []
    """
    n = 1 + max(states) # number of states
    M = [[0]*n for _ in range(n)]

    # how many times it changes from state i to state j
    for (i,j) in zip(states, states[1:]):
        M[i][j] += 1

    # now convert to probabilities:
    for row in M:
        s = sum(row)
        s = float(s)
        if s > 0:
            row[:] = [round(f/s, 3) for f in row]
    for i in M:
        with open(os.path.join(output_path, 'transition_matrix_{}_{}.txt'.format(max(states), condition)),
                  "a+") as text_file:
            text_file.write(str(i))
    return M


def mahalanobis_dictance(array_a, array_b):
    """
    Computes the mahalanobis distance between two 1D arrays

    :param array_a: array a
    :type array_a: pd.Series
    :param array_b: array b
    :type array_b: pd.Series
    :return: mahalanobis distance
    :rtype: float
    """
    if not isinstance(array_a, np.ndarray):
        array_a = np.array(array_a.tolist())
    if not isinstance(array_b, np.ndarray):
        array_b = np.array(array_b.tolist())
    V = np.cov(np.array([array_a, array_b]).T)
    IV = np.linalg.inv(V)

    return mahalanobis(array_a, array_b, IV)
