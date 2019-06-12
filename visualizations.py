"""
Visualization tools for functional connectivity dynamics. 

Katerina Capouskova 2018, kcapouskova@hotmail.com
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import cm

sns.set_context("paper")


def plot_functional_connectivity_matrix(fcd_matrix, output_path):
    """
    Plots the heatmap of functional connectivity dynamics (TxT)

    :param fcd_matrix: functional connectivity matrix
    :type fcd_matrix: np.ndarray
    :param output_path: path to output directory 
    :type output_path: str
    """
    fig, ax = plt.subplots(1)
    heat_map = sns.heatmap(fcd_matrix[1, :, :], cmap='jet', ax=ax,
                           square=True)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Time (seconds)')
    heat_map.set_yticklabels(heat_map.get_yticklabels(), rotation=0, fontsize=4)
    heat_map.set_xticklabels(heat_map.get_xticklabels(), rotation=90, fontsize=4)

    plt.savefig(os.path.join(output_path, 'FCD_matrix_heatmap.png'))
    # plt.show()


def plot_dfc_areas_correlation(connectivity, output_path):
    """
    Plots brain areas correlations (NxN).

    :param connectivity: array representing dynamical functional conn.
    :type connectivity: np.ndarray, pd.DataFrame
    :param output_path: path to output directory
    :type output_path: str
    """
    if isinstance(connectivity, pd.DataFrame):
        pass
    else:
        connectivity = pd.DataFrame(data=connectivity)
    plt.subplots(figsize=(20, 15))
    #fig, ax = plt.subplots(1)
    #cmap = sns.diverging_palette(250, 15, as_cmap=True, center="dark")
    heat_map = sns.heatmap(connectivity, cmap='RdYlGn',
                           square=True, vmin=-1, vmax=1)
    plt.xlabel('Brain area')
    plt.ylabel('Brain area')
    heat_map.set_yticklabels(heat_map.get_yticklabels(), rotation=0, fontsize=10)
    heat_map.set_xticklabels(heat_map.get_xticklabels(), rotation=90, fontsize=10)

    plt.savefig(os.path.join(output_path, 'Area_correlation_heatmap_averaged.png'))
    #plt.show()


def plot_averaged_dfc_clustermap(data, output_path):
    """
    Plots brain areas correlations (NxN).

    :param data: array representing average dfc of a state
    :type data: pd.DataFrame
    :param output_path: path to output directory
    :type output_path: str
    """
    # cmap = sns.diverging_palette(250, 15, as_cmap=True, center="dark")
    c_map = sns.clustermap(data, cmap='RdYlGn', yticklabels=True,
                           xticklabels=True, figsize=(20, 20))
    plt.xlabel('Brain area')
    plt.ylabel('Brain area')

    plt.savefig(
        os.path.join(output_path, 'Averaged_dfc_clustered.png'))
    #plt.show()


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
    # sns.plt.show()


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
    # plt.show()


def plot_states_line(cluster_states, t_phases, output_path):
    """
    Plots the states in one line of one subejct.

    :param cluster_states: array representing clustered states
    :type cluster_states: np.ndarray
    :param t_phases: number of time phases
    :type t_phases: int
    :param output_path: path to output directory
    :type output_path: str
    """
    plt.style.use('seaborn-whitegrid')
    fig = plt.figure()
    ax = plt.axes()
    ax.plot(cluster_states[0:t_phases], '-y')
    plt.savefig(os.path.join(output_path, 'clustered_states_plot.png'))
    # plt.show()


def plot_variance(labels, variance, output_path):
    """
    Plots the states' variances.

    :param labels: array representing labels
    :type labels: np.ndarray
    :param variance: array representing variance
    :type variance: np.ndarray
    :param output_path: path to output directory
    :type output_path: str
    """
    fig, ax = plt.subplots(1)
    violin = sns.violinplot(x=labels, y=variance, inner='quartile')
    plt.xlabel('States')
    plt.ylabel('Variance')
    plt.savefig(os.path.join(output_path, 'States variance.png'))
    # plt.show()


def plot_probabilities_barplots(df, output_path):
    """
    Plots the states' probabilities histogram.

    :param df: probabilities dataframe
    :type df: pd.Dataframe
    :param output_path: path to output directory
    :type output_path: str
    """
    sns.set(style="whitegrid")

    # Draw a nested barplot
    g = sns.factorplot(x='cluster', y='probability', hue='condition',
                       size=6, kind='bar', palette='PRGn', data=df)
    g.despine(left=True)
    g.set_ylabels('Probability')
    plt.savefig(os.path.join(output_path, 'States probabilities cond.png'))
    # plt.show()


def plot_probabilities_boxplots(df, output_path):
    """
    Plots the states' probabilities boxplot.

    :param df: probabilities dataframe
    :type df: pd.Dataframe
    :param output_path: path to output directory
    :type output_path: str
    """
    sns.set(style="whitegrid")

    # Draw a nested barplot
    g = sns.boxplot(x='cluster', y='probability', data=df, hue='condition',
                palette="PRGn")
    plt.ylabel('Probability')
    plt.savefig(os.path.join(output_path, 'States probabilities boxplot.png'))
    # plt.show()


def plot_lifetimes_boxplots(df, output_path):
    """
    Plots the states' lifetimes boxplot.

    :param df: lifetimes dataframe
    :type df: pd.DataFrame
    :param output_path: path to output directory
    :type output_path: str
    """
    sns.set(style="whitegrid")

    # Draw a nested barplot
    g = sns.boxplot(x='cluster', y='lifetime', data=df, hue='condition',
                palette="PRGn")
    plt.ylabel('Lifetime')
    plt.savefig(os.path.join(output_path, 'States lifetimes boxplot.png'))
    # plt.show()


def plot_lifetimes_barplots(df, output_path):
    """
    Plots the states' lifetimes histogram.

    :param df: lifetimes dataframe
    :type df: pd.DataFrame
    :param output_path: path to output directory
    :type output_path: str
    """
    sns.set(style="whitegrid")

    # Draw a nested barplot
    g = sns.factorplot(x='cluster', y='lifetime', hue='condition',
                       size=6, kind='bar', palette='PRGn', data=df)
    g.despine(left=True)
    g.set_ylabels('Mean lifetime of a state (seconds)')
    plt.savefig(os.path.join(output_path, 'States lifetimes.png'))
    # plt.show()


def plot_silhouette_analysis(X, output_path, n_clusters, silhouette_avg,
                             sample_silhouette_values, cluster_labels, centers):
    """
    Plots the Silhouette analysis for clustering algorithm.

    :param X: clustering input features
    :type X: np.ndarray
    :param output_path: path to output directory
    :type output_path: str
    :param n_clusters: number of clusters
    :type n_clusters: int
    :param silhouette_avg: silhouette average score
    :type silhouette_avg: float
    :param sample_silhouette_values: silhouette scores for each sample
    :type sample_silhouette_values: float
    :param cluster_labels: cluster labels
    :type cluster_labels: int
    :param centers: coordinates of cluster centers
    :type centers: array, [n_clusters, n_features]
    """
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = ['darkorange', 'mediumslateblue', 'mediumaquamarine', 'orchid',
                 'steelblue', 'lightgreen', 'lightslategrey', 'darksalmon',
                 'tomato', 'turquoise']
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color[i], edgecolor=color[i], alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="orangered", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    colors = plt.get_cmap('Spectral')(np.linspace(0, 1, 10))
    ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                c=colors, edgecolor='k')
    # Draw white circles at cluster centers
    ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                c="white", alpha=1, s=200, edgecolor='k')

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                    s=50, edgecolor='k')

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                  "with n_clusters = %d" % n_clusters),
                 fontsize=14, fontweight='bold')
    plt.savefig(os.path.join(output_path, 'Clustering_{}.png'.format(n_clusters)))
    # plt.show()


def plot_autoe_vs_pca(pca_a, enc_a, output_path):
    """
    Plots the PCA vs autoencoder dimensionality reduction.

    :param pca_a: PCA output array
    :type pca_a: np.ndarray
    :param enc_a: autoencoder output array
    :type enc_a: np.ndarray
    :param output_path: path to output directory
    :type output_path: str
    """
    plt.figure(figsize=(8, 4))
    plt.subplot(121)
    plt.title('PCA')
    plt.scatter(pca_a[:5000, 0], pca_a[:5000, 1], s=8,
                cmap='tab10')
    plt.gca().get_xaxis().set_ticklabels([])
    plt.gca().get_yaxis().set_ticklabels([])

    plt.subplot(122)
    plt.title('Autoencoder')
    plt.scatter(enc_a[:5000, 0], enc_a[:5000, 1], s=8,
                cmap='tab10')
    plt.gca().get_xaxis().set_ticklabels([])
    plt.gca().get_yaxis().set_ticklabels([])

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'PCA_vs_autoencoder.png'))


def plot_val_los_autoe(val, loss, output_path):
    """
    Plots the training and validation loss function of an autoencoder.

    :param val: validation loss values
    :type val: []
    :param loss: training loss values
    :type loss: []
    :param output_path: path to output directory
    :type output_path: str
    """
    sns.set(style="whitegrid")
    dict = {'validation': val, 'training': loss}
    data = pd.DataFrame(data=dict)
    sns.lineplot(data=data, palette="tab10", linewidth=2.5)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and validation loss')
    plt.savefig(os.path.join(output_path, 'Val_loss_autoencoder.png'))


def plot_see_against_n_clusters(list_k, sse, silhouette, output_path):
    """
    Plots the sum of squared distances against number of clusters.

    :param list_k: list of number of clusters
    :type list_k: []
    :param sse: list of sum of squared distances
    :type sse: []
    :param silhouette: list of mean silhouette scores
    :type silhouette: []
    :param output_path: path to output directory
    :type output_path: str
    """
    # Plot sse against k
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    sil = [1-s for s in silhouette]
    ax1.plot(list_k, sse, '-o')
    ax2.plot(list_k, sil, 'ro-')
    ax1.set_xlabel(r'Number of clusters *k*')
    ax1.set_ylabel('Sum of squared distance')
    ax2.set_ylabel('Silhouette score')
    plt.savefig(os.path.join(output_path, 'sse_sil_n_clusters.png'))
