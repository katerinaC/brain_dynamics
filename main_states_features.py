"""
Script that covers states post-processing like comparing the lifetimes and
probabilities of states estimating the p-value. It does it for the whole
concatenated array or separately. It also computes the variance and entropy of
each state. Lastly, visualizations are performed.

Katerina Capouskova 2018, kcapouskova@hotmail.com
"""

import argparse
import itertools
import json
import os

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

from states_features import distribution_probability_lifetime, \
    variance_of_states, entropy_of_states, students_t_test, \
    mean_lifetime_of_state, probability_of_state, permutation_t_test
from utilities import create_dir, separate_concat_array
from visualizations import plot_variance, plot_probabilities_barplots, \
    plot_lifetimes_barplots


def parse_args():
    """
    Argument parser.
    """

    parser = argparse.ArgumentParser('Run FC states features.')

    parser.add_argument('--input', type=str, help='Path to the concatenated '
                                                    'matrix with clusters',
                        required=True)
    parser.add_argument('--output', type=str,
                        help='Path to output folder', required=True)
    parser.add_argument('--n_clusters', type=int,
                        help='Number of clusters', required=True)
    parser.add_argument('--starts', type=str,
                        help='Path to json with starts to separate in concat',
                        required=False)
    parser.add_argument('--sub_t', type=str,
                        help='Path to json with task names and corresponding '
                             'time points and number of subjects',
                        required=False)
    parser.add_argument('--separate', action='store_true', default=False,
                        help='Separate tasks from in the concatenated matrix',
                        required=False)
    parser.add_argument('--clusters', type=str,
                        help='Path to clusters file clustered_matrix', required=True)
    parser.add_argument('--tr', type=int,
                        help='TR of imaging method', required=True)
    return parser.parse_args()


def main():
    """
    FC states features
    """
    args = parse_args()
    input_path = args.input
    separate = args.separate
    output_path = args.output
    n_clusters = args.n_clusters
    starts_json = args.starts
    clusters = args.clusters
    sub_t = args.sub_t
    TR = args.tr

    create_dir(output_path)
    reduced_components = np.load(input_path)['arr_0'][:, :-1]
    variance = variance_of_states(reduced_components, output_path)
    labels = np.load(clusters)['arr_0']
    plot_variance(labels, variance, output_path)
    probabilities, lifets, df_prob = distribution_probability_lifetime(labels, output_path, n_clusters, TR)
    for i in range(n_clusters):
        probs = df_prob[df_prob['clusters'] == i]
        entropy_of_states(probs['probabilities'], output_path, i)
    times_subjects = json.load(open(sub_t))

    if separate:
        probas_p_values = []
        lifetimes_p_values = []
        conditions = []
        new_paths = separate_concat_array(input_path, starts_json, output_path,
                                          n_clusters)
        for path in tqdm(new_paths):
            output_p = os.path.join(output_path,
                                    os.path.basename(os.path.dirname(path)))
            create_dir(output_p)
            matrix = np.load(path)['arr_0']
            clusters = matrix[:, -1]
            reduced_task = matrix[:, :-1]
            task_var = variance_of_states(reduced_task, output_p)
            plot_variance(clusters, task_var, output_p)

        for a, b in itertools.combinations(new_paths, 2):
            a_name = os.path.basename(os.path.dirname(a))
            b_name = os.path.basename(os.path.dirname(b))
            s_t_a = times_subjects[a_name]
            a_labels = np.load(a)['arr_0'][:, -1]
            group_a = np.reshape(a_labels, (s_t_a[0], s_t_a[1]))
            s_t_b = times_subjects[b_name]
            b_labels = np.load(b)['arr_0'][:, -1]
            group_b = np.reshape(b_labels, (s_t_b[0], s_t_b[1]))
            output = os.path.join(output_path, a_name + '_' + b_name)
            create_dir(output)
            a_probas = []
            b_probas = []
            a_lt = []
            b_lt = []

            for s_a in range(s_t_a[0]):
                proba_a = probability_of_state(group_a[s_a, :], n_clusters, output)
                a_proba_list = [proba_a[i] for i in group_a[s_a, :]]
                a_probas.extend(a_proba_list)
                lt_a = mean_lifetime_of_state(group_a[s_a, :], n_clusters,
                                              output, TR)
                a_lt_list = [lt_a[i] for i in group_a[s_a, :]]
                a_lt.extend(a_lt_list)

            for s_b in range(s_t_b[0]):
                proba_b = probability_of_state(group_b[s_b, :], n_clusters,
                                               output)
                b_proba_list = [proba_b[i] for i in group_b[s_b, :]]
                b_probas.extend(b_proba_list)
                lt_b = mean_lifetime_of_state(group_b[s_b, :], n_clusters,
                                              output, TR)
                b_lt_list = [lt_b[i] for i in group_b[s_b, :]]
                b_lt.extend(b_lt_list)

            cond_a = [a_name for i in range(len(a_labels))]
            cond_b = [b_name for z in range(len(b_labels))]

            dict_prob = {'probability': a_probas + b_probas,
                         'lifetime': a_lt + b_lt,
                         'condition': cond_a + cond_b,
                         'cluster': a_labels.tolist() + b_labels.tolist()}
            df = pd.DataFrame(data=dict_prob)
            df.to_csv(os.path.join(output, 'probas_lt_dataframe.csv'))
            plot_probabilities_barplots(df, output)
            plot_lifetimes_barplots(df, output)
            for c in tqdm(range(n_clusters)):
                df_n = df[df['cluster'] == c]
                con_a_df = df_n[df_n['condition'] == a_name]
                con_b_df = df_n[df_n['condition'] == b_name]
                create_dir(os.path.join(
                    output_path, a_name, str(c)))
                create_dir(os.path.join(
                    output_path, b_name, str(c)))
                entropy_of_states(con_a_df['probability'], os.path.join(
                    output_path, a_name, str(c)), c)
                entropy_of_states(con_b_df['probability'],os.path.join(
                    output_path, b_name, str(c)), c)
                p_prob, t_prob = permutation_t_test(con_a_df['probability'], con_b_df['probability'],
                                                 os.path.join(output, str(c), 'probability'))
                p_lt, t_lt = permutation_t_test(con_a_df['lifetime'],
                                             con_b_df['lifetime'],
                                             os.path.join(output, str(c), 'lifetime'))
                probas_p_values.append(p_prob)
                lifetimes_p_values.append(p_lt)
                conditions.append(a_name + '_' + b_name + '_' + str(c))
        probas_p_adjusted = multipletests(probas_p_values, alpha=0.05, method='bonferroni')
        lt_p_adjusted = multipletests(lifetimes_p_values, alpha=0.05, method='bonferroni')
        p_values = pd.DataFrame({'probabilities_p': probas_p_values,
                                 'bonferroni_probans_p': probas_p_adjusted[1].tolist(),
                                 'bonferroni_lt_p': lt_p_adjusted[1].tolist(),
                                 'lifetimes_p': lifetimes_p_values,
                                 'conditions': conditions})
        p_values.to_csv(os.path.join(output_path, 'p_values_{}.csv'.format(n_clusters)))


if __name__ == '__main__':
    main()
