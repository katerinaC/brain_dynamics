"""
Script that computes functional connectivity dynamics

Katerina Capouskova 2018, kcapouskova@hotmail.com
"""

import argparse
import os

from modeling_FC_states import kmeans_clustering
from data_processing_functional_connectivity import \
    preform_pca_on_instant_connectivity, dynamic_functional_connectivity, \
    preform_lle_on_instant_connectivity
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
        pca_components = preform_pca_on_instant_connectivity(paths_list, output_path,
                                                             brain_areas,
                                                             t_phases, n_subjects)
        fcd_matrix = dynamic_functional_connectivity(pca_components, output_path,
                                                     t_phases, n_subjects)
        clusters = kmeans_clustering(pca_components, output_path)
        plot_functional_connectivity_matrix(fcd_matrix, output_path)

    if lle:
        paths_list = return_paths_list(input_path, output_path, pattern)
        lle_components = preform_lle_on_instant_connectivity(paths_list,
                                                             output_path,
                                                             brain_areas,
                                                             t_phases,
                                                             n_subjects)
        fcd_matrix = dynamic_functional_connectivity(lle_components,
                                                     output_path,
                                                     t_phases, n_subjects)
        clusters = kmeans_clustering(lle_components, output_path)
        plot_functional_connectivity_matrix(fcd_matrix, output_path)

if __name__ == '__main__':
    main()
