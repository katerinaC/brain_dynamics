"""
Visualization tools for functional connectivity dynamics. 

Katerina Capouskova 2018, kcapouskova@hotmail.com
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set()


def plot_functional_connectivity_matrix(fcd_matrix, output_path):
    """
    Plots the heatmap of functional connectivity dynamics (TxT)

    :param fcd_matrix: functional connectivity matrix
    :type fcd_matrix: np.ndarray
    :param output_path: path to output directory 
    :type output_path: str
    """
    fig, ax = plt.subplots(1)
    heat_map = sns.heatmap(fcd_matrix[1, :, :], cmap="rainbow", ax=ax,
                           square=True)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Time (seconds)')
    heat_map.set_yticklabels(heat_map.get_yticklabels(), rotation=45, fontsize=4)
    heat_map.set_xticklabels(heat_map.get_xticklabels(), rotation=0, fontsize=4)
    plt.show()

    plt.savefig(os.path.join(output_path, 'FCD_matrix_heatmap.png'))


def plot_clustering():
    """
    Plots the clusters of states

    :param fcd_matrix: functional connectivity matrix
    :type fcd_matrix: np.ndarray
    :param output_path: path to output directory 
    :type output_path: str
    """