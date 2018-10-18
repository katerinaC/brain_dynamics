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
from sklearn import manifold, preprocessing
from sklearn.decomposition import PCA
from tqdm import tqdm

from utilities import return_paths_list, create_dir


def convert_to_phases(input_path, output_path, brain_areas, t_phases, subject):
    """
    Converts raw data into phases by Hilbert Transform

    :param input_path: path to input file
    :type input_path: str
    :param output_path: path to output directory
    :type output_path: str
    :param brain_areas: number of brain areas
    :type brain_areas: int
    :param t_phases: number of time phases
    :type t_phases: int
    :param subject: subject number
    :type subject: int
    :return: phases matrix
    :rtype: np.ndarray
    """
    phases = np.full((brain_areas, t_phases), fill_value=0).astype(np.float64)
    array = np.genfromtxt(input_path, delimiter=',')
    for area in tqdm(range(0, brain_areas)):
        # select by columns, transform to phase
        time_series = pylab.demean(signal.detrend(array[:, area]))
        phases[area, :] = np.angle(signal.hilbert(time_series))
    np.savez(os.path.join(output_path, 'phases_{}'.format(subject)), phases)
    return phases


def dynamic_functional_connectivity(input_path, output_path, brain_areas,
                                    pattern):
    """
    Computes the dynamic functional connectivity of brain areas.

    :param input_path: path to input dir
    :type input_path: str
    :param output_path: path to output directory
    :type output_path: str
    :param brain_areas: number of brain areas
    :type brain_areas: int
    :param pattern: pattern of input files
    :type pattern: str
    :return: dFC output path
    :rtype: str
    """
    paths = return_paths_list(input_path, output_path, pattern=pattern)
    n_subjects = len(paths)
    array = np.genfromtxt(paths[0], delimiter=',')
    t_phases = array.shape[0]
    dFC = np.full((brain_areas, brain_areas), fill_value=0).astype(np.float64)

    for n in tqdm(range(n_subjects)):
        phases = convert_to_phases(paths[n], output_path, brain_areas, t_phases, n)
        for t in range(0, t_phases):
            for i in range(0, brain_areas):
                for z in range(0, brain_areas):
                    if np.absolute(phases[i, t] - phases[z, t]) > np.pi:
                        dFC[i, z] = np.cos(2 * np.pi - np.absolute(
                            phases[i, t] - phases[z, t]))
                    else:
                        dFC[i, z] = np.cos(np.absolute(phases[i, t] -
                                                       phases[z, t]))
            dfc_output = os.path.join(output_path, 'dFC')
            create_dir(dfc_output)
            np.savez(os.path.join(dfc_output, 'subject_{}_time_{}'.format(n, t)), dFC)

    return dfc_output


def preform_pca_on_dynamic_connectivity(input_path, output_path, brain_areas,
                                        pattern):
    """
    Computes the dynamic connectivity of brain areas with performing
    a PCA returning its matrix.

    :param input_path: path to input dir
    :type input_path: str
    :param output_path: path to output directory 
    :type output_path: str
    :param brain_areas: number of brain areas
    :type brain_areas: int
    :param pattern: pattern of input files
    :type pattern: str
    :return: PCA matrix, PCA matrix shape
    :rtype: np.ndarray, tuple
    """
    paths = return_paths_list(input_path, output_path, pattern=pattern)
    n_subjects = len(paths)
    array = np.genfromtxt(paths[0], delimiter=',')
    t_phases = array.shape[0]
    dFC = np.full((brain_areas, brain_areas), fill_value=0).astype(np.float64)
    pca_components = np.full((n_subjects, t_phases, (brain_areas * 2)),
                             fill_value=0).astype(np.float64)
    for n in tqdm(range(n_subjects)):
        phases = convert_to_phases(paths[n], output_path, brain_areas, t_phases, n)
        for t in range(0, t_phases):
            for i in range(0, brain_areas):
                for z in range(0, brain_areas):
                    if np.absolute(phases[i, t] - phases[z, t]) > np.pi:
                        dFC[i, z] = np.cos(2 * np.pi - np.absolute(
                            phases[i, t] - phases[z, t]))
                    else:
                        dFC[i, z] = np.cos(np.absolute(phases[i, t] -
                                                       phases[z, t]))
            dfc_output = os.path.join(output_path, 'dFC')
            create_dir(dfc_output)
            np.savez(os.path.join(dfc_output, 'subject_{}_time_{}'.format(n, t)), dFC)
            pca = PCA(n_components=2)
            # normalize
            # dFC = preprocessing.normalize(dFC, norm='l2')
            pca.fit(dFC)
            pca_dict = {
                'components': pca.components_.tolist(),
                'explained variance': pca.explained_variance_.tolist(),
                'explained mean variance': np.mean(pca.explained_variance_.tolist()),
                'explained variance ratio': pca.explained_variance_ratio_.tolist(),
                'mean': pca.mean_.tolist(),
                'n components': pca.n_components_,
                'noise variance': pca.noise_variance_.tolist()
            }
            with open(os.path.join(output_path, 'PCA_results_{}_{}'.format(n, t)),
                      'w') as output:
                json.dump(pca_dict, output)
            pca_components[n, t, :] = \
                pca_dict['components'][0] + pca_dict['components'][1]
    # save the PCA matrix into a .npz file
    np.savez(os.path.join(output_path, 'components_matrix'), pca_components)
    return pca_components, pca_components.shape


def preform_lle_on_dynamic_connectivity(input_path, output_path, brain_areas,
                                        pattern):
    """
    Computes the dynamic connectivity of brain areas with performing
    a locally linear embedding returning its matrix.

    :param input_path: path to input dir
    :type input_path: str
    :param output_path: path to output directory 
    :type output_path: str
    :param brain_areas: number of brain areas
    :type brain_areas: int
    :param pattern: pattern of input files
    :type pattern: str
    :return: LLE matrix, LLE matrix shape
    :rtype: np.ndarray, tuple
    """
    paths = return_paths_list(input_path, output_path, pattern=pattern)
    n_subjects = len(paths)
    array = np.genfromtxt(paths[0], delimiter=',')
    t_phases = array.shape[0]
    dFC = np.full((brain_areas, brain_areas), fill_value=0).astype(np.float64)
    lle_components = np.full((n_subjects, t_phases, (brain_areas * 2)),
                             fill_value=0).astype(np.float64)
    for n in tqdm(range(0, n_subjects)):
        phases = convert_to_phases(paths[n], output_path, brain_areas, t_phases, n)
        for t in range(0, t_phases):
            for i in range(0, brain_areas):
                for z in range(0, brain_areas):
                    if np.absolute(phases[i, t] - phases[z, t]) > np.pi:
                        dFC[i, z] = np.cos(2 * np.pi - np.absolute(
                            phases[i, t] - phases[z, t]))
                    else:
                        dFC[i, z] = np.cos(np.absolute(phases[i, t] -
                                                       phases[z, t]))
            dfc_output = os.path.join(output_path, 'dFC')
            create_dir(dfc_output)
            np.savez(os.path.join(dfc_output, 'subject_{}_time_{}'.format(n, t)),
                dFC)
            lle, err = manifold.locally_linear_embedding(dFC, n_neighbors=12,
                                                         n_components=2)
            with open(os.path.join(output_path, 'LLE_error_{}_{}'.format(n, t)),
                      'w') as output:
                json.dump(err, output)
            lle_components[n, t, :] = np.squeeze(lle.flatten())
    # save the LLE matrix into a .npz file
    np.savez(os.path.join(output_path, 'components_matrix'), lle_components)
    return lle_components, lle_components.shape


def functional_connectivity_dynamics(reduced_components, output_path):
    """
    Computes the functional connectivity dynamics of brain areas.

    :param reduced_components: reduced components matrix
    :type reduced_components: np.ndarray
    :param output_path: path to output directory 
    :type output_path: str
    :return: FCD matrix
    :rtype: np.ndarray
    """
    n_subjects, t_phases, brain_areas_2 = reduced_components.shape
    FCD = np.full((n_subjects, t_phases, t_phases), fill_value=0)\
        .astype(np.float64)
    # Compute the FCD matrix for each subject as cosine similarity over time
    for subject in tqdm(range(0, n_subjects)):
        for t1 in range(0, t_phases):
            vec_1 = np.squeeze(reduced_components[subject, t1, :])
            for t2 in range(0, t_phases):
                vec_2 = np.squeeze(reduced_components[subject, t2, :])
                # cosine similarity
                FCD[subject, t1, t2] = np.dot(vec_1, vec_2) / (np.linalg.norm(
                    vec_1) * np.linalg.norm(vec_2))
    np.savez(os.path.join(output_path, 'FCD_matrix'), FCD)
    return FCD
