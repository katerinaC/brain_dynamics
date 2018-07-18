"""
Script that covers states post-processing like comparing the lifetimes and
probabilities of states estimating the p-value. It does it for the whole
concatenated array or separately. It also computes the variance and entropy of
each state. Lastly, visualizations are performed.

Katerina Capouskova 2018, kcapouskova@hotmail.com
"""

import argparse
import itertools
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from states_features import distribution_probability_lifetime, \
    variance_of_states, entropy_of_states, students_t_test, \
    mean_lifetime_of_state, probability_of_state
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
    parser.add_argument('--separate', action='store_true', default=False,
                        help='Separate tasks from in the concatenated matrix',
                        required=False)
    parser.add_argument('--clusters', type=str,
                        help='Path to clusters file clustered_matrix', required=True)
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

    create_dir(output_path)
    reduced_components = np.load(input_path)['arr_0'][:, :-1]
    variance = variance_of_states(reduced_components, output_path)
    labels = np.load(clusters)['arr_0']
    plot_variance(labels, variance, output_path)
    probabilities, lifets = distribution_probability_lifetime(labels, output_path, n_clusters)
    entropy_of_states(probabilities, output_path, n_clusters)

    if separate:
        probas_p_values = []
        lifetimes_p_values = []
        new_paths = separate_concat_array(input_path, starts_json, output_path,
                                          n_clusters)
        for path in tqdm(new_paths):
            output_p = os.path.join(output_path,
                                    os.path.basename(os.path.dirname(path)))
            create_dir(output_p)
            matrix = np.load(path)['arr_0']
            clusters = matrix[:, -1]
            reduced_task = matrix[:, :-1]
            probas, lifetimes = distribution_probability_lifetime(clusters, output_p, n_clusters)
            task_var = variance_of_states(reduced_task, output_p)
            plot_variance(clusters, task_var, output_p)
            entropy_of_states(probas, output_p, n_clusters)

        for a, b in itertools.combinations(new_paths, 2):
            group_a = np.load(a)['arr_0'][:, -1]
            group_b = np.load(b)['arr_0'][:, -1]
            a_name = os.path.basename(os.path.dirname(a))
            b_name = os.path.basename(os.path.dirname(b))
            output = os.path.join(output_path, a_name + '_' + b_name)
            create_dir(output)
            proba_a = probability_of_state(group_a, n_clusters, output)
            proba_a = {int(k): v for k,v in proba_a.items()}
            proba_b = probability_of_state(group_b, n_clusters, output)
            proba_b = {int(k): v for k, v in proba_b.items()}
            lt_a = mean_lifetime_of_state(group_a, n_clusters, output)
            lt_a = {int(k): v for k, v in lt_a.items()}
            lt_b = mean_lifetime_of_state(group_b, n_clusters, output)
            lt_b = {int(k): v for k, v in lt_b.items()}
            probas_a = [proba_a[i] for i in group_a]
            probas_b = [proba_b[y] for y in group_b]
            lts_a = [lt_a[m] for m in group_a]
            lts_b = [lt_b[n] for n in group_b]
            cond_a = [a_name for i in range(len(group_a))]
            cond_b = [b_name for z in range(len(group_b))]
            dict_prob = {'probability': probas_a + probas_b,
                         'lifetime': lts_a + lts_b,
                         'condition': cond_a + cond_b,
                         'cluster': group_a.tolist() + group_b.tolist()}
            df = pd.DataFrame(data=dict_prob)
            df.to_csv(os.path.join(output, 'probas_lt_dataframe.csv'))
            plot_probabilities_barplots(df, output)
            plot_lifetimes_barplots(df, output)
            for c in tqdm(range(n_clusters)):
                df_n = df[df['cluster'] == c]
                con_a = df_n[df_n['condition'] == a_name]
                con_b = df_n[df_n['condition'] == b_name]
                t_prob, p_prob = students_t_test(con_a['probability'], con_b['probability'],
                                                 os.path.join(output, str(c), 'probability'))
                t_lt, p_lt = students_t_test(con_a['lifetime'],
                                             con_b['lifetime'],
                                             os.path.join(output, str(c), 'lifetime'))
                probas_p_values.append(p_prob)
                lifetimes_p_values.append(p_lt)
        p_values = pd.DataFrame({'probabilities_p': probas_p_values,
                                 'lifetimes_p': lifetimes_p_values})
        p_values.to_csv(os.path.join(output_path, 'p_values_{}.csv'.format(n_clusters)))


if __name__ == '__main__':
    main()
