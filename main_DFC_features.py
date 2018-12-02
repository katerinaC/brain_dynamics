"""
Script that processes dynamic functional connectivity(DFC) according to identified
clusters.

Katerina Capouskova 2018, kcapouskova@hotmail.com
"""
import argparse
import json
import os
import shutil
import pandas as pd
import numpy as np
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

    # Load labels and starts json and divide labels by tasks into separate
    # folders
    labels = np.load(clusters)['arr_0']
    with open(starts_json) as s:
        starts = json.load(s)
    clusters = []
    cluster_paths = []
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
            shutil.copyfile(n, os.path.join(cluster_output, file_name))

    # Average for clusters and visualise
    n_clusters = max(clusters)  # Number of clusters
    cluster_paths = list(set(cluster_paths))
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

        # import brain area names if known
        if names is not None:
            names_array = np.load(names)
            df = pd.DataFrame(averaged, index=names_array, columns=names_array)
            plot_dfc_areas_correlation(df, c)
            plot_averaged_dfc_clustermap(df, c)
        else:
            plot_dfc_areas_correlation(averaged, c)
            plot_averaged_dfc_clustermap(averaged, c)


if __name__ == '__main__':
    main()
