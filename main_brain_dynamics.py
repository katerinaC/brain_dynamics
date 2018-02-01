"""
Script that computes functional connectivity dynamics

Katerina Capouskova 2018, kcapouskova@hotmail.com
"""

import argparse
import os

from clustering_FC_states import kmeans_clustering
from data_processing_functional_connectivity import \
    preform_pca_on_instant_connectivity, dynamic_functional_connectivity
from visualizations import plot_functional_connectivity_matrix


def parse_args():
    """
    Argument parser.
    """

    parser = argparse.ArgumentParser('Run brain dynamics module.')

    parser.add_argument('--input', type=str, nargs='+',
                        help='Path to the input directory.',
                        required=True)
    parser.add_argument('--pattern', type=str, default='.csv',
                        help='Type of the data file (.csv or .mat.',
                        required=False)
    parser.add_argument('--output', type=str,
                        help='Path to output folder.', required=True)

    return parser.parse_args()


def main():
    """
    Dynamic functional connectivity
    """
    args = parse_args()
    input_path = args.input_path
    pattern = args.pattern
    output_path = os.path.normpath(args.output)

    os.makedirs(output_path)

    pca_components = preform_pca_on_instant_connectivity(input_path, output_path,
                                                         pattern)
    fcd_matrix = dynamic_functional_connectivity(pca_components, output_path)
    clusters = kmeans_clustering(pca_components, output_path)
    plot_functional_connectivity_matrix(fcd_matrix, output_path)

if __name__ == '__main__':
    main()
