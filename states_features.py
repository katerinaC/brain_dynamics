"""
Functions for identifying different states features.

Katerina Capouskova 2018, kcapouskova@hotmail.com
"""
import json
import logging
import os
import scipy

import numpy as np
from scipy.stats import stats


def probability_of_state(clusters, n_clusters, output_path):
    """
    Computes the probability of a state.

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
    dict = {}
    for n in range(n_clusters):
        separate_cluster = []
        for c in clusters:
            if c == n:
                separate_cluster.append(c)
        p = float(len(separate_cluster))/float(len(clusters))
        dict.update({n: p})
        logging.info('Probability of state: {} is: {}'.format(n, p))
    with open(os.path.join(output_path, 'probability.json'), 'w') as fp:
        json.dump(dict, fp)
    return dict


def mean_lifetime_of_state(clusters, n_clusters, output_path):
    """
    Computes the mean lifetime of a state.

    :param clusters: clusters array
    :type clusters: np.ndarray
    :param n_clusters: number of clusters
    :type n_clusters: int
    :param output_path: path to output directory
    :type output_path: str
    :return: dictinory {state: mean lifetime}
    :rtype: dict
    """
    # Feature of the data: Repetition time in seconds
    TR = 2
    dict = {}
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
        dict.update({n: mean_duration})
    with open(os.path.join(output_path, 'mean_state_lifetime.json'), 'w') as fp:
        json.dump(dict, fp)
    return dict


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
    logging.basicConfig(
        filename=os.path.join(output_path, 'students_t_test.log'),
        level=logging.INFO)
    t, p = stats.ttest_ind(group_a, group_b, equal_var=False)
    logging.info('T-test value: {}, p-value: {}'.format(t, p))
    return t, p


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


def distribution_probability_lifetime(clusters, output_path, n_clusters):
    """
    Creates a distribution of probabilities and life times of states

    :param clusters: clusters array
    :type clusters: np.ndarray
    :param output_path: path to output directory
    :type output_path: str
    :param n_clusters: number of clusters
    :type n_clusters: int
    :return: proba_list:array of probabilities, lifetimes_list: array of lifetimes
    :rtype: np.ndarray, np.ndarray
    """
    probas = probability_of_state(clusters, n_clusters, output_path)
    lifetimes = mean_lifetime_of_state(clusters, n_clusters, output_path)
    proba_list = []
    lifetimes_list = []
    for n in range(n_clusters):
        for elem in clusters:
            if elem == n:
                proba_list.append(probas[n])
                lifetimes_list.append(lifetimes[n])
    return np.array(proba_list), np.array(lifetimes_list)


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


def entropy_of_states(probabilities, output_path, n_clusters):
    """
    Computes the entropy of probabilities of states

    :param probabilities: array with states probabilities
    :type probabilities: np.ndarray
    :param output_path: path to output directory
    :type output_path: str
    :param n_clusters: number of clusters
    :type: n_clusters: int
    :return: entropy: calculated entropy
    :rtype: int
    """
    logging.basicConfig(
        filename=os.path.join(output_path, 'entropy_n_clusters_{}.log'.format(
            n_clusters)),
        level=logging.INFO)
    entropy = scipy.stats.entropy(probabilities)
    logging.info('State {} entropy is {}'.format(n_clusters, entropy))
    return entropy
