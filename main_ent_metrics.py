"""
Script that calculates different (entropy) metrics for the probabilities of
different states.

Katerina Capouskova 2019, kcapouskova@hotmail.com
"""
import argparse
import itertools
import os

import scipy

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from states_features import kl_distance_symm, transition_matrix, \
    permutation_t_test, mahalanobis_dictance
from utilities import symarray, create_dir
from visualizations import plot_kl_distance, plot_ent_boxplot, \
    plot_transition_matrix
from scipy.stats import stats


def parse_args():
    """
    Argument parser.
    """

    parser = argparse.ArgumentParser('Run ent metrics.')

    parser.add_argument('--input', type=str,
                        help='Path to the probabilities file (.csv)',
                        required=True)
    parser.add_argument('--output', type=str,
                        help='Path to output folder', required=True)

    return parser.parse_args()


def main():
    """
    Probability metrics
    """
    args = parse_args()
    input_path = args.input
    output_path = args.output

    create_dir(output_path)

    df = pd.read_csv(input_path)
    conditions = (df.condition.unique()).tolist()
    n_subjects = df['subject'].max()
    clusters = (df.cluster.unique()).tolist()

    # Make Rest the last item
    conditions.pop(conditions.index('Rest'))
    conditions.append('Rest')

    df_sub_p = pd.DataFrame()
    # Selecting from a csv file
    for condition in conditions:
        df_cond = df[df['condition'] == condition]
        for subject in range(n_subjects+1):
            df_sub = df_cond[df_cond['subject'] == subject]
            for cluster in clusters:
                df_c = df_sub[df_sub['cluster'] == cluster]
                sub_prob = (df_c['probability'].unique()).tolist()
                df_sub_p = df_sub_p.append({'Condition': condition, 'subject': subject,
                                        'Subject_probability': sub_prob, 'Cluster': cluster},
                      ignore_index=True)

    #  Kullback-Leibler (KL) divergence with symmetrization
    kl_matrix = symarray(np.ones((len(conditions), len(conditions))))

    # Mahalanobis distance
    #m_matrix = symarray(np.ones((len(conditions), len(conditions))))

    for cond_a, cond_b in itertools.combinations(conditions, 2):
        cond_a = str(cond_a)
        cond_b = str(cond_b)
        df_a = df_sub_p[df_sub_p['Condition'] == cond_a]
        prob_a = df_a['Subject_probability']
        df_b = df_sub_p[df_sub_p['Condition'] == cond_b]
        prob_b = df_b['Subject_probability']
        kl_symm = kl_distance_symm(prob_a, prob_b)
        #mahal = mahalanobis_dictance(prob_a, prob_b)
        index_a = conditions.index(cond_a)
        index_b = conditions.index(cond_b)
        kl_matrix[index_a, index_b] = kl_symm
        #m_matrix[index_a, index_b] = mahal

    # plot distances
    plot_kl_distance(kl_matrix, conditions, output_path, 'kl')
    #plot_kl_distance(m_matrix, conditions, output_path, 'mahalanobis')

    # Entropies box plot for each condition
    df_ent = pd.DataFrame()
    for condition in conditions:
        df_ent_1 = df_sub_p[df_sub_p['Condition'] == condition]
        for subject in range(n_subjects+1):
            df_ent_2 = df_ent_1[df_ent_1['subject'] == subject]
            subject_p = df_ent_2['Subject_probability'].tolist()
            entropy = scipy.stats.entropy(subject_p)
            df_ent = df_ent.append({'Condition': condition, 'Entropy(H)': float(entropy)},
                               ignore_index=True)
    p_ent_list = []
    conds = []
    for cond_a, cond_b in itertools.combinations(conditions, 2):
        df_ent_a = df_ent[df_ent['Condition'] == cond_a]
        df_ent_b = df_ent[df_ent['Condition'] == cond_b]
        p_ent, t_ent = permutation_t_test(df_ent_a['Entropy(H)'].tolist(),
                                            df_ent_b['Entropy(H)'].tolist(),
                                            os.path.join(output_path,
                                                         '{}_{}_t_test'.format(
                                                        str(cond_a), str(cond_b))))

        p_ent_list.append(p_ent)
        conds.append(str(cond_a) + str(cond_b))
    ent_p_adjusted = multipletests(p_ent_list, alpha=0.05,
                                      method='bonferroni')
    p_values = pd.DataFrame({'ent_p': p_ent_list,
                             'bonferroni_ent_p': ent_p_adjusted[1].tolist(),
                             'conditions': conds})
    p_values.to_csv(os.path.join(output_path, 'ent_p_values.csv'))

    plot_ent_boxplot(df_ent, output_path)

    # Transition matrices
    for condition in conditions:
        df_cond = df[df.condition == condition]
        list = df_cond.cluster.tolist()
        list_int = map(int, list)
        m = transition_matrix(list_int, condition, output_path)
        plot_transition_matrix(m, condition, output_path)


if __name__ == '__main__':
    main()
