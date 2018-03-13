"""
Function for data pre-processing to get dynamic function connectivity 
multidimensional matrix.
Takes data as a numpy array and performs a PCA 
and then compares by cosine similarity all time points to return a phase-lag 
matrix of dynamic functional connectivity.

Katerina Capouskova 2018, kcapouskova@hotmail.com
"""

import json
import os

import numpy as np
import pylab

from scipy import signal
from sklearn import manifold
from sklearn.decomposition import PCA
from tqdm import tqdm


def preform_pca_on_instant_connectivity(paths_list, output_path, brain_areas,
                                        t_phases, n_subjects):
    """
    Computes the instant connectivity of brain areas with performing 
    a PCA returning its matrix.

    :param paths_list: list of all paths 
    :type paths_list: []
    :param output_path: path to output directory 
    :type output_path: str
    :param brain_areas: number of brain areas
    :type brain_areas: int
    :param t_phases: number of time phases
    :type t_phases: int
    :param n_subjects: number of subjects 
    :type n_subjects: int
    :return: PCA matrix
    :rtype: np.ndarray
    """
    phases = np.full((brain_areas, t_phases), fill_value=0).astype(np.float64)
    iFC = np.full((brain_areas, brain_areas), fill_value=0).astype(np.float64)
    pca_components = np.full((n_subjects, t_phases, brain_areas), fill_value=0)\
        .astype(np.float64)

    for path in tqdm(paths_list):
        array = np.genfromtxt(path, delimiter=',')
        for area in tqdm(range(0, brain_areas)):
            # select by columns, transform to phase
            time_series = pylab.demean(signal.detrend(array[:, area]))
            phases[area, :] = np.angle(signal.hilbert(time_series))
            for t in tqdm(range(0, t_phases)):
                for i in tqdm(range(0, brain_areas)):
                    for z in tqdm(range(0, brain_areas)):
                        if np.absolute(phases[i, t] - phases[z, t]) > np.pi:
                            iFC[i, z] = np.cos(2 * np.pi - np.absolute(
                                phases[i, t] - phases[z, t]))
                        else:
                            iFC[i, z] = np.absolute(phases[i, t] - phases[z, t])
                pca = PCA(n_components=1)
                pca.fit(iFC)
                pca_dict = {
                    'components': pca.components_.tolist(),
                    'explained variance': pca.explained_variance_.tolist(),
                    'explained variance ratio': pca.
                    explained_variance_ratio_.tolist(),
                    'mean': pca.mean_.tolist(),
                    'n components': pca.n_components_,
                    'noise variance': pca.noise_variance_.tolist()
                }
                with open(os.path.join(output_path, 'PCA_results_{}'.format(t)),
                          'w') as output:
                    json.dump(pca_dict, output)
                pca_components[paths_list.index(path), t, :] = \
                    pca_dict['components'][0]
    # save the PCA matrix into a .csv file
    np.savez(os.path.join(output_path, 'PCA_components_matrix'), pca_components)
    return pca_components


def preform_lle_on_instant_connectivity(paths_list, output_path, brain_areas,
                                        t_phases, n_subjects):
    """
    Computes the instant connectivity of brain areas with performing 
    a locally linear embedding returning its matrix.

    :param paths_list: path to directory with all .csv files 
    :type paths_list: str
    :param output_path: path to output directory 
    :type output_path: str
    :param brain_areas: number of brain areas
    :type brain_areas: int
    :param t_phases: number of time phases
    :type t_phases: int
    :param n_subjects: number of subjects 
    :type n_subjects: int
    :return: LLE matrix
    :rtype: np.ndarray
    """
    phases = np.full((brain_areas, t_phases), fill_value=0).astype(np.float64)
    iFC = np.full((brain_areas, brain_areas), fill_value=0).astype(np.float64)
    lle_components = np.full((n_subjects, t_phases, brain_areas), fill_value=0) \
        .astype(np.float64)

    for path in tqdm(paths_list):
        array = np.genfromtxt(path, delimiter=',')
        for area in tqdm(range(0, brain_areas)):
            # select by columns, transform to phase
            time_series = pylab.demean(signal.detrend(array[:, area]))
            phases[area, :] = np.angle(signal.hilbert(time_series))
            for t in tqdm(range(0, t_phases)):
                for i in tqdm(range(0, brain_areas)):
                    for z in tqdm(range(0, brain_areas)):
                        if np.absolute(phases[i, t] - phases[z, t]) > np.pi:
                            iFC[i, z] = np.cos(2 * np.pi - np.absolute(
                                phases[i, t] - phases[z, t]))
                        else:
                            iFC[i, z] = np.absolute(phases[i, t] - phases[z, t])
                lle, err = manifold.locally_linear_embedding(iFC, n_neighbors=12,
                                                             n_components=1)
                with open(os.path.join(output_path, 'LLE_error_{}'.format(t)),
                          'w') as output:
                    json.dump(err, output)
                lle_components[paths_list.index(path), t, :] = np.squeeze(lle)

    # save the PCA matrix into a .csv file
    np.savez(os.path.join(output_path, 'LLE_components_matrix'), lle_components)
    return lle_components


def dynamic_functional_connectivity(reduced_components, output_path, t_phases,
                                    n_subjects):
    """
    Computes the functional connectivity dynamics of brain areas.

    :param reduced_components: reduced components matrix
    :type reduced_components: np.ndarray
    :param output_path: path to output directory 
    :type output_path: str
    :param t_phases: number of time phases
    :type t_phases: int
    :param n_subjects: number of subjects 
    :type n_subjects: int
    :return: FCD matrix
    :rtype: np.ndarray
    """
    FCD = np.full((n_subjects, t_phases, t_phases), fill_value=0)\
        .astype(np.float64)
    # Compute the FCD matrix for each subject as cosine similarity over time
    for subject in tqdm(range(0, n_subjects)):
        for t1 in tqdm(range(0, t_phases)):
            vec_1 = np.squeeze(reduced_components[subject, t1, :])
            for t2 in tqdm(range(0, t_phases)):
                vec_2 = np.squeeze(reduced_components[subject, t2, :])
                FCD[subject, t1, t2] = np.dot(vec_1, vec_2) / np.linalg.norm(
                    vec_1) / np.linalg.norm(vec_2)
    np.savez(os.path.join(output_path, 'FCD_matrix'), FCD)
    return FCD
