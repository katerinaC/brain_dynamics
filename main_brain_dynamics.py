"""
Script that computes functional connectivity dynamics and does clustering of
brain states.

Katerina Capouskova 2018, kcapouskova@hotmail.com
"""

import argparse
import os

from data_processing_functional_connectivity import \
    convert_to_phases, \
    preform_lle_on_dynamic_connectivity, preform_pca_on_dynamic_connectivity, \
    functional_connectivity_dynamics
from modeling_FC_states import kmeans_clustering
from utilities import return_paths_list
from visualizations import plot_functional_connectivity_matrix


def parse_args():
    """
    Argument parser.
    """

    parser = argparse.ArgumentParser('Run brain dynamics module.')

    parser.add_argument('--input', type=str, help='Path to the input directory',
                        required=True)
    parser.add_argument('--pattern', type=str, default='.csv',
                        help='Type of the data file (.csv or .mat.)',
                        required=False)
    parser.add_argument('--output', type=str,
                        help='Path to output folder', required=True)
    parser.add_argument('--areas', type=int,
                        help='Number of brain areas', required=True)
    parser.add_argument('--phases', type=int,
                        help='Number of time phases', required=True)
    parser.add_argument('--subjects', type=int,
                        help='Number of participating subjects', required=True)
    parser.add_argument('--pca', action='store_true', default=False,
                        help='Perform PCA data dimension reduction', required=False)
    parser.add_argument('--lle', action='store_true', default=False,
                        help='Perform Locally Linear Embedding data dimension '
                             'reduction', required=False)
    return parser.parse_args()


def main():
    """
    Dynamic functional connectivity
    """
    args = parse_args()
    input_path = args.input
    pattern = args.pattern
    output_path = os.path.normpath(args.output)
    brain_areas = args.areas
    t_phases = args.phases
    n_subjects = args.subjects
    pca = args.pca
    lle = args.lle

    if os.path.isdir(output_path):
        pass
    else:
        os.makedirs(output_path)

    if pca:
        paths_list = return_paths_list(input_path, output_path, pattern)
        phases = convert_to_phases(paths_list, output_path, brain_areas, t_phases)
        pca_components = preform_pca_on_dynamic_connectivity(phases, output_path,
                                                             brain_areas,
                                                             t_phases, n_subjects)
        fcd_matrix = functional_connectivity_dynamics(pca_components, output_path,
                                                     t_phases, n_subjects)
        clusters = kmeans_clustering(pca_components, output_path)
        #hidden_states, predict_proba, n_components, markov_array = hidden_markov_model_pomegranate(
            #pca_components, output_path)
        #plot_hidden_states(hidden_states, n_components, markov_array, output_path)
        plot_functional_connectivity_matrix(fcd_matrix, output_path)

    if lle:
        paths_list = return_paths_list(input_path, output_path, pattern)
        phases = convert_to_phases(paths_list, output_path, brain_areas, t_phases)
        lle_components = preform_lle_on_dynamic_connectivity(phases, output_path,
                                                             brain_areas,
                                                             t_phases,
                                                             n_subjects)
        fcd_matrix = functional_connectivity_dynamics(lle_components,
                                                     output_path,
                                                     t_phases, n_subjects)
        clusters = kmeans_clustering(lle_components, output_path)
        #hidden_states, predict_proba, n_components, markov_array = hidden_markov_model_pomegranate(
            #lle_components, output_path)
        #plot_hidden_states(hidden_states, n_components, markov_array, output_path)
        plot_functional_connectivity_matrix(fcd_matrix, output_path)

if __name__ == '__main__':
    main()
