"""
Visualization tools for functional connectivity dynamics. 

Katerina Capouskova 2018, kcapouskova@hotmail.com
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import nitime.timeseries as ts
import numpy as np
import seaborn as sns
from matplotlib import cm
from nitime.analysis import CoherenceAnalyzer

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
    heat_map = sns.heatmap(fcd_matrix[1, :, :], cmap='hsv', ax=ax,
                           square=True)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Time (seconds)')
    heat_map.set_yticklabels(heat_map.get_yticklabels(), rotation=0, fontsize=4)
    heat_map.set_xticklabels(heat_map.get_xticklabels(), rotation=90, fontsize=4)

    plt.savefig(os.path.join(output_path, 'FCD_matrix_heatmap.png'))
    plt.show()


def plot_dfc_areas_correlation(instant_connectivity, output_path):
    """
    Plots brain areas correlations of one subject at one time point (NxN).

    :param instant_connectivity: array representing PCA or LLE matrix
    :type instant_connectivity: np.ndarray
    :param output_path: path to output directory
    :type output_path: str
    """
    brain_areas = instant_connectivity.shape[2]
    iFC = np.full((brain_areas, brain_areas), fill_value=0).astype(np.float64)
    for i in range(0, brain_areas):
        for z in range(0, brain_areas):
            iFC[i, z] = instant_connectivity[1, 3, z]
            print instant_connectivity[1, 3, z]
    df = pd.DataFrame(data=iFC)
    corr = df.corr()
    fig, ax = plt.subplots(1)
    #cmap = sns.diverging_palette(250, 15, as_cmap=True, center="dark")
    heat_map = sns.heatmap(corr, cmap='RdYlGn', ax=ax,
                           square=True)
    plt.xlabel('Brain area')
    plt.ylabel('Brain area')
    heat_map.set_yticklabels(heat_map.get_yticklabels(), rotation=0, fontsize=4)
    heat_map.set_xticklabels(heat_map.get_xticklabels(), rotation=90, fontsize=4)

    plt.savefig(os.path.join(output_path, 'Area_correlation_heatmap.png'))
    plt.show()


def plot_hidden_states(hidden_states, n_components, markov_array, output_path):
    """
    Plots the hidden states of Hidden Markov Model

    :param hidden_states: matrix with predicted hidden states
    :type hidden_states: np.ndarray
    :param n_components: number of components in the model
    :type n_components: int
    :param markov_array: array that was used for the model
    :type markov_array: np.ndarray
    :param output_path: path to output directory 
    :type output_path: str    
    """
    df = pd.DataFrame(markov_array)
    col = [n for n in range(0, len(hidden_states))]
    sns.set(font_scale=1.25)
    style_kwds = {'xtick.major.size': 3, 'ytick.major.size': 3,
                  'legend.frameon': True}
    sns.set_style('white', style_kwds)

    fig, axs = plt.subplots(n_components, sharex=True, sharey=True,
                            figsize=(12, 9))
    colors = cm.rainbow(np.linspace(0, 1, n_components))

    for i, (ax, color) in enumerate(zip(axs, colors)):
        # Use fancy indexing to plot data in each state.
        mask = hidden_states == i
        ax.plot(df.index.values[mask], df[col].values[mask], 'o', c=color, )
        ax.set_title("{}th hidden state".format(i), fontsize=14,
                     fontweight='demi')

        # Format the ticks.
        sns.despine(offset=10)

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'Hidden Markov Model Different States.png'))
    plt.show()

    sns.set(font_scale=1.5)
    df.rename(columns={df.columns[-1]: 'states'},
              inplace=True)
    df['time points'] = [n for n in range(len(hidden_states))]
    print df.head()
    sns.set_style('white', style_kwds)
    fg = sns.FacetGrid(data=df, hue='states', palette=colors, aspect=1.31,
                       size=12)
    fg.map(plt.scatter, 'time points', 0, alpha=0.8).add_legend()
    sns.despine(offset=10)
    fg.fig.suptitle('Different brain states according to HMM', fontsize=24,
                    fontweight='demi')
    fg.savefig(os.path.join(output_path, 'Hidden Markov Model States.png'))
    sns.plt.show()


def dash_plot_timeseries(instant_connectivity, output_path):
    """
    Plots the timeseries of one subject.

    :param instant_connectivity: array representing PCA or LLE matrix
    :type instant_connectivity: np.ndarray
    :param output_path: path to output directory
    :type output_path: str
    """
    array = instant_connectivity[1, 2, :]
    df = pd.DataFrame(array)
    df.plot()
    plt.savefig(os.path.join(output_path, 'timeseries_plot.png'))
    plt.show()


def plot_states_line(cluster_states, output_path):
    """
    Plots the states in one line.

    :param cluster_states: array representing clustered states
    :type cluster_states: np.ndarray
    :param output_path: path to output directory
    :type output_path: str
    """
    plt.style.use('seaborn-whitegrid')
    fig = plt.figure()
    ax = plt.axes()
    ax.plot(cluster_states, '-y')
    plt.savefig(os.path.join(output_path, 'clustered_states_plot.png'))
    plt.show()


