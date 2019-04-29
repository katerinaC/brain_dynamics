"""
Script that performs graph analysis on dFC matrices for different states.
It also performs permutation test to estimate p-values between different tasks.

Katerina Capouskova 2019, kcapouskova@hotmail.com
"""

import argparse
import itertools
import os

import community
import networkx as nx
import numpy as np
import pandas as pd
from statsmodels.sandbox.stats.multicomp import multipletests

from states_features import permutation_t_test
from utilities import create_dir, return_paths_list


def parse_args():
    """
    Argument parser.
    """

    parser = argparse.ArgumentParser('Run graph analysis.')

    parser.add_argument('--input', type=str, help='Path to the dfc folder, '
                                                  'after the path should have form:'
                                                  '/task/dFC_out/nr_of_state',
                        required=True)
    parser.add_argument('--output', type=str,
                        help='Path to output folder', required=True)
    parser.add_argument('--n_clusters', type=int,
                        help='Number of clusters', required=True)
    parser.add_argument('--tasks', nargs='+', required=True,
                        help='Name of tasks')
    return parser.parse_args()


def main():
    """
    Graph analysis, permutation tests
    """
    args = parse_args()
    input_path = args.input
    tasks = args.tasks
    output_path = args.output
    n_clusters = args.n_clusters

    create_dir(output_path)
    states = [str(float(i)) for i in range(n_clusters)]

    for task in tasks:
        for state in states:
            input_p = '{}/{}/dFC_out/{}'.format(input_path, task, str(state))
            in_paths = return_paths_list(input_p, output_path, '.npz')
            clustering = []
            modularity = []
            avg_path = []

            for in_path in in_paths:
                path, file = os.path.split(in_path)
                if file == 'averaged_dfc.npz':
                    continue
                if file == 'averaged_dfc_thesholded.npz':
                    continue
                input = np.load(in_path)['arr_0']
                input = np.absolute(input)
                np.fill_diagonal(input, 0)
                A = np.asmatrix(input)
                G = nx.from_numpy_matrix(A)
                clust_coef = nx.clustering(G, weight='weight')
                clust_list = [v for _, v in clust_coef.items()]
                avg_clust_coef = sum(clust_list) / len(clust_list)
                clustering.append(avg_clust_coef)
                short_path = nx.shortest_path(G)
                s_p_dicts = [v for _, v in short_path.items()]
                s_p = []
                for i in s_p_dicts:
                    s_p_ = [len(v) for k, v in i.iteritems()]
                    s_p.extend(s_p_)
                avg_s_p = sum(s_p) / len(s_p)
                avg_path.append(avg_s_p)
                # rc = nx.rich_club_coefficient(G)
                part = community.best_partition(G)
                values = [part.get(node) for node in G.nodes()]
                mod = community.modularity(part, G)
                modularity.append(mod)
            task_name_len = len(modularity)
            task_name = [task for i in range(task_name_len)]
            dict_graphs = {'condition': task_name, 'clustering': clustering,
                           'modularity': modularity,
                           'avg_shortest_path': avg_path}
            df = pd.DataFrame(data=dict_graphs)
            df.to_csv(os.path.join(output_path,
                                   'graph_analysis_{}_{}_{}.csv'.format(n_clusters,
                                       str(state), task)))

    conditions = []
    p_clust = []
    mod = []
    for a, b in itertools.combinations(tasks, 2):
        for c in range(n_clusters):
            input_a = pd.read_csv(
                '{}/graph_analysis_7_{}_{}.csv'.format(output_path, str(float(c)), a))
            input_b = pd.read_csv(
                '{}/graph_analysis_7_{}_{}.csv'.format(output_path, str(float(c)), b))
            p_cl, t_clust = permutation_t_test(input_a['clustering'],
                                               input_b['clustering'],
                                               os.path.join(output_path, str(
                                                   c) + '_' + 'clustering' + '_' + a + '_' + b))
            p_clust.append(p_cl)
            p_mod, t_mod = permutation_t_test(input_a['modularity'],
                                              input_b['modularity'],
                                              os.path.join(output_path, str(
                                                  c) + '_' + 'modularity' + '_' + a + '_' + b))
            mod.append(p_mod)
            conditions.append(a + '_' + b + '_' + str(c))

    p_adjusted_clust = multipletests(p_clust, alpha=0.05, method='bonferroni')
    p_adjusted_mod = multipletests(mod, alpha=0.05, method='bonferroni')

    p_values = pd.DataFrame({'p_clustering': p_clust,
                             'modularity': mod,
                             'bonferrroni_mod': p_adjusted_mod[1].tolist(),
                             'bonferrroni_clust': p_adjusted_clust[1].tolist(),
                             'conditions': conditions})

    p_values.to_csv(
        os.path.join(output_path, 'p_values_graph_analysis_{}.csv'.format(n_clusters)))


if __name__ == '__main__':
    main()
