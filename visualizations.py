"""
Visualization tools for functional connectivity dynamics. 

Katerina Capouskova 2018, kcapouskova@hotmail.com
"""
import os

import matplotlib.pyplot as plt
import pandas as pd
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
    heat_map = sns.heatmap(fcd_matrix[0, :, :], cmap="rainbow", ax=ax)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Time (seconds)')

    heat_map.savefig(os.path.join(output_path, 'FCD_matrix_heatmap'))


