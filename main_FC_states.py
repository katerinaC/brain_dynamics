"""
Script that computes functional connectivity dynamics and does clustering of
brain states.
Note: check for number of components in dim. reduction and TR in mean
lifetime of states (default 2 for both).
Check for delimiter when using .csv

Katerina Capouskova 2018, kcapouskova@hotmail.com
"""

import argparse
import json
import os
import numpy as np
from sklearn import metrics
from sklearn.cluster import KMeans

from data_processing_functional_connectivity import \
    preform_lle_on_dynamic_connectivity, preform_pca_on_dynamic_connectivity, \
    functional_connectivity_dynamics, dynamic_functional_connectivity, \
    preform_lead_eig_on_dynamic_connectivity
from modeling_FC_states import kmeans_clustering, kmeans_clustering_mean_score, \
    dbscan, autoencoder
from utilities import convert_components, \
    create_new_output_path, create_dir, preprocess_autoencoder, \
    return_paths_list
from visualizations import plot_functional_connectivity_matrix, \
    plot_states_line, plot_see_against_n_clusters


def parse_args():
    """
    Argument parser.
    """

    parser = argparse.ArgumentParser('Run FC states clustering module.')

    parser.add_argument('--input', nargs='+', help='Path to the input directory',
                        required=True)
    parser.add_argument('--pattern', type=str, default='.csv',
                        help='Type of the data file (.csv or .mat.)',
                        required=False)
    parser.add_argument('--output', type=str,
                        help='Path to output folder', required=True)
    parser.add_argument('--areas', type=int,
                        help='Number of brain areas', required=True)
    parser.add_argument('--phases', type=int,
                        help='Number of time phases', required=False)
    parser.add_argument('--tr', type=float,
                        help='TR of imaging method', required=True)
    parser.add_argument('--pca', action='store_true', default=False,
                        help='Perform PCA data dimension reduction', required=False)
    parser.add_argument('--lle', action='store_true', default=False,
                        help='Perform Locally Linear Embedding data dimension '
                             'reduction', required=False)
    parser.add_argument('--lead_eig', action='store_true', default=False,
                        help='Get leading eigenvector as dim. reduction',
                        required=False)
    parser.add_argument('--autoen', action='store_true', default=False,
                        help='Perform autoencoder data dimension reduction', required=False)
    parser.add_argument('--clusters', type=int, default=None,
                        help='Number of clusters', required=False)
    parser.add_argument('--db', action='store_true', default=False,
                        help='Perform DBSCAN clustering algorithm',
                        required=False)
    parser.add_argument('--imb', action='store_true', default=False,
                        help='Imbalanced dataset', required=False)
    return parser.parse_args()


def main():
    """
    Dynamic functional connectivity states clustering
    """
    args = parse_args()
    input_paths = args.input
    pattern = args.pattern
    output_path = os.path.normpath(args.output)
    brain_areas = args.areas
    pca = args.pca
    lle = args.lle
    lead_eig = args.lead_eig
    n_clusters = args.clusters
    t_phases = args.phases
    db = args.db
    autoen = args.autoen
    imbalanced = args.imb
    TR = args.tr

    create_dir(output_path)

    new_outputs = []
    dfc_paths = []
    output_paths = []
    dict = {}

    for input_path in input_paths:
        name = os.path.basename(input_path)
        paths_list = return_paths_list(input_path, output_path, pattern=pattern)
        n_subjects = len(paths_list)
        array = np.genfromtxt(paths_list[0], delimiter=';')
        t_phases = array.shape[0]
        dict.update({name: [n_subjects, t_phases]})
        new_output = create_new_output_path(input_path, output_path)
        new_outputs.append(new_output)
        create_dir(new_output)
        output_paths.append(os.path.join(new_output, 'components_matrix.npz'))

        if pca:
            components, shape = preform_pca_on_dynamic_connectivity(
                paths_list, new_output, brain_areas, pattern, t_phases, n_subjects, TR)
            fcd_matrix = functional_connectivity_dynamics(components, new_output)
            plot_functional_connectivity_matrix(fcd_matrix, new_output)

        if lle:
            components, shape = preform_lle_on_dynamic_connectivity(
                paths_list, new_output, brain_areas, pattern, t_phases, n_subjects, TR)
            fcd_matrix = functional_connectivity_dynamics(components,
                                                          new_output)
            plot_functional_connectivity_matrix(fcd_matrix, output_path)

        if lead_eig:
            components, shape = preform_lead_eig_on_dynamic_connectivity(
                paths_list, new_output, brain_areas, pattern, t_phases, n_subjects, TR)
            fcd_matrix = functional_connectivity_dynamics(components,
                                                          new_output)
            plot_functional_connectivity_matrix(fcd_matrix, output_path)

        if autoen:
            dfc_path = dynamic_functional_connectivity(
                paths_list, new_output, brain_areas, pattern, t_phases, n_subjects, TR)
            dfc_paths.append(dfc_path)

    if autoen:
        dfc_all, n_samples, y = preprocess_autoencoder(dfc_paths, output_path,
                                                       brain_areas)
        encoded = autoencoder(dfc_all, output_path, y, imbalanced=imbalanced)

    if n_clusters is not None and autoen is False:
        # concatenate all data
        concatenated = convert_components(output_paths, output_path)
        kmeans_clustering_mean_score(concatenated, output_path, n_clusters, TR)

    elif n_clusters is not None and autoen is True:
        kmeans_clustering_mean_score(encoded, output_path, n_clusters, TR)
        sse = []
        silhouette = []
        list_k = list(range(2, 20))

        for k in list_k:
            km = KMeans(n_clusters=k)
            km.fit(encoded)
            sse.append(km.inertia_)
            labels = km.labels_
            silhouette.append(metrics.silhouette_score(encoded, labels,
                                                       metric='euclidean',
                                                       sample_size=500))

        plot_see_against_n_clusters(list_k, sse, silhouette, output_path)

    elif db:
        concatenated = convert_components(output_paths, output_path)
        dbscan(concatenated, output_path)

    else:
        # perform clustering on data separately
        clusters = kmeans_clustering(components, output_path)
        plot_states_line(clusters, t_phases, output_path)

    with open(os.path.join(output_path, 'subjects_times_dict.json'), 'w') as fp:
        json.dump(dict, fp)


if __name__ == '__main__':
    main()
