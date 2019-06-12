"""
Script that processes dynamic functional connectivity(DFC) according to identified
clusters.

Katerina Capouskova 2018, kcapouskova@hotmail.com
"""
import argparse
import json
import os
import shutil

import community
import networkx as nx
import pandas as pd
import numpy as np
from numpy.linalg import linalg
from tqdm import tqdm

from utilities import create_dir, return_paths_list
from visualizations import plot_dfc_areas_correlation, \
    plot_averaged_dfc_clustermap


def parse_args():
    """
    Argument parser.
    """

    parser = argparse.ArgumentParser('Run DFC features.')

    parser.add_argument('--input', type=str,
                        help='Path to the main data folder',
                        required=True)
    parser.add_argument('--output', type=str,
                        help='Path to output folder', required=True)
    parser.add_argument('--starts', type=str,
                        help='Path starts json file', required=True)
    parser.add_argument('--clusters', type=str,
                        help='Path to clusters file concatentated_matrix_clusters',
                        required=True)
    parser.add_argument('--features', type=int,
                        help='Number of features (brain areas) - for PCA: BA * 2,'
                             'for autoencoder: number of features in encoded',
                        required=True)
    parser.add_argument('--names', type=str,
                        help='Path to a file with brain areas names (.npy file)',
                        required=False)
    return parser.parse_args()


def main():
    """
    DFC features
    """
    args = parse_args()
    input_path = args.input
    output_path = args.output
    starts_json = args.starts
    clusters = args.clusters
    brain_areas = args.features
    names = args.names

    if names is not None:
        names_array = np.load(names)
    else:
        names_array = None

    # Load labels and starts json and divide labels by tasks into separate
    # folders
    labels = np.load(clusters)['arr_0']
    with open(starts_json) as s:
        starts = json.load(s)
    clusters = []
    cluster_paths = []
    all_cluster_paths = []

    for key, values in tqdm(starts.iteritems()):
        output_p = os.path.join(output_path, key)
        create_dir(os.path.join(output_p, 'dFC_out'))
        new_array = labels[values[0]: values[1], :]
        dFC_paths = return_paths_list(os.path.join(input_path, key, 'dFC'),
                                      output_path, '.npz')
        np.savez(os.path.join(output_path, key, 'dFC_out', 'labels_{}'.format(key)),
                 new_array)

        # Tasks labels divide dFCs according to labels into states folders
        for n in dFC_paths:
            cluster = new_array[dFC_paths.index(n)][-1]
            clusters.append(cluster)
            cluster_output = os.path.join(output_p, 'dFC_out', str(cluster))
            create_dir(cluster_output)
            cluster_paths.append(cluster_output)
            file_name = os.path.basename(n)
            all_clusters_out = os.path.join(output_path, str(cluster))
            create_dir(all_clusters_out)
            all_cluster_paths.append(all_clusters_out)
            shutil.copyfile(n, os.path.join(all_clusters_out, key + '_' + file_name))
            shutil.copyfile(n, os.path.join(cluster_output, file_name))
    
    # Average for clusters and visualise
    n_clusters = max(clusters)  # Number of clusters
    # Remove duplicates from path lists
    cluster_paths = list(set(cluster_paths))
    all_cluster_paths = list(set(all_cluster_paths))

    # Divided by tasks
    for c in tqdm(cluster_paths):
        matrix_paths = return_paths_list(c, output_path, '.npz')
        n_matrix = len(matrix_paths)
        avg_dfc = np.full((n_matrix, brain_areas, brain_areas), fill_value=0).astype(
            np.float64)
        for i in range(n_matrix):
            matrix = np.load(matrix_paths[i])['arr_0']
            avg_dfc[i, :, :] = matrix
        averaged = np.average(avg_dfc, 0)  # Average over all matrices in a cluster

        np.savez(os.path.join(c, 'averaged_dfc'), averaged)
        plot_dfc_areas_correlation(averaged, c)
        plot_averaged_dfc_clustermap(averaged, c)

    modularity = {}
    clust_coeffs = {}

    # Not divided by tasks
    for m in tqdm(all_cluster_paths):
        print m
        matrix_paths = return_paths_list(m, output_path, '.npz')
        n_matrix = len(matrix_paths)
        print n_matrix
        avg_dfc = np.memmap('merged.buffer', dtype=np.float64, mode='w+',
                            shape=(n_matrix, brain_areas, brain_areas))
        for i in range(n_matrix):
            matrix = np.load(matrix_paths[i])['arr_0']
            avg_dfc[i, :, :] = matrix
        averaged = np.average(avg_dfc, 0)  # Average over all matrices in a cluster

        state = os.path.split(m)[1]
        # modularity
        input = np.absolute(averaged)
        np.fill_diagonal(input, 0)
        A = np.asmatrix(input)
        G = nx.from_numpy_matrix(A)
        part = community.best_partition(G)
        mod = community.modularity(part, G)
        modularity.update({state: mod})

        # clust coeff
        clust_coef = nx.clustering(G, weight='weight')
        clust_list = [v for _, v in clust_coef.items()]
        avg_clust_coef = sum(clust_list) / len(clust_list)
        clust_coeffs.update({state: avg_clust_coef})

        np.savez(os.path.join(m, 'averaged_dfc'), averaged)

        eigen_vals, eigen_vects = linalg.eig(averaged)
        leading_eig = eigen_vects[:, eigen_vals.argmax()]
        np.savez(
            os.path.join(m, 'averaged_dfc_lead_eig'),
            leading_eig)

        # import brain area names if known
        if names_array is not None:
            idx = eigen_vals.argsort()[::-1]
            names_eigs = names_array[idx]
            np.savetxt(
                os.path.join(m, 'leading_eigs_sorted_names.csv'),
                names_eigs, fmt='%s')
            df = pd.DataFrame(averaged, index=names_array, columns=names_array)
            plot_dfc_areas_correlation(df, m)
            plot_averaged_dfc_clustermap(df, m)
        else:
            plot_dfc_areas_correlation(averaged, m)
            plot_averaged_dfc_clustermap(averaged, m)

    with open(os.path.join(output_path, 'modularity_avg_dfcs.json'), 'w') as fp:
            json.dump(modularity, fp)

    with open(os.path.join(output_path, 'avg_clust_coeff_avg_dfcs.json'), 'w') as fp:
            json.dump(clust_coeffs, fp)


if __name__ == '__main__':
    main()
