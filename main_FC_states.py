"""
Script that computes functional connectivity dynamics and does clustering of
brain states.
Note: check for number of components in dim. reduction and TR in mean
lifetime of states (default 2 for both).

Katerina Capouskova 2018, kcapouskova@hotmail.com
"""

import argparse
import os

from data_processing_functional_connectivity import \
    preform_lle_on_dynamic_connectivity, preform_pca_on_dynamic_connectivity, \
    functional_connectivity_dynamics
from modeling_FC_states import kmeans_clustering, kmeans_clustering_mean_score
from utilities import convert_components, \
    create_new_output_path, create_dir
from visualizations import plot_functional_connectivity_matrix, plot_states_line


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
    parser.add_argument('--pca', action='store_true', default=False,
                        help='Perform PCA data dimension reduction', required=False)
    parser.add_argument('--lle', action='store_true', default=False,
                        help='Perform Locally Linear Embedding data dimension '
                             'reduction', required=False)
    parser.add_argument('--clusters', type=int, default=None,
                        help='Number of clusters', required=False)
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
    t_phases = args.phases
    clusters = args.clusters

    create_dir(output_path)

    output_paths = []
    for input_path in input_paths:
        new_output = create_new_output_path(input_path, output_path)
        create_dir(new_output)
        output_paths.append(os.path.join(new_output, 'components_matrix.npz'))
        if pca:
            components, shape = preform_pca_on_dynamic_connectivity(input_path, new_output,
                                                                 brain_areas,
                                                                 pattern)
            fcd_matrix = functional_connectivity_dynamics(components, new_output)
            plot_functional_connectivity_matrix(fcd_matrix, new_output)

        if lle:
            components, shape = preform_lle_on_dynamic_connectivity(input_path, new_output,
                                                                    brain_areas,
                                                                    pattern)
            fcd_matrix = functional_connectivity_dynamics(components,
                                                          new_output)
            plot_functional_connectivity_matrix(fcd_matrix, output_path)

    if clusters is not None:
        # concatenate all data
        concatenated = convert_components(output_paths, output_path)
        kmeans_clustering_mean_score(concatenated, output_path,
                                     clusters)
    else:
        # perform clustering on data separately
        clusters = kmeans_clustering(components, output_path)
        plot_states_line(clusters, t_phases, output_path)

    # hidden_states, predict_proba, n_components, markov_array = hidden_markov_model_pomegranate(
    # pca_components, output_path)
    # plot_hidden_states(hidden_states, n_components, markov_array, output_path)


if __name__ == '__main__':
    main()
