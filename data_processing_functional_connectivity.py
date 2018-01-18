"""
Function for data pre-processing to get dynamic function connectivity 
multidimensional matrix.
Takes data as a numpy array and performs a PCA 
and then compares by cosine similarity all time points to return a phase-lag 
matrix of dynamic functional connectivity.

Katerina Capouskova 2018, kcapouskova@hotmail.com
"""
import fnmatch
import json
import os

import numpy as np
import pylab

from scipy import signal
from sklearn.decomposition import PCA

# number of brain areas
brain_areas = 90
# number of time phases
t_phases = 175
# number of subjects
n_subjects = 98


def preform_pca_on_functional_connectivity(input_path, output_path,
                                           pattern='.csv'):
    """
    Computes the functional connectivity of brain areas with performing 
    a PCA returning its matrix.

    :param input_path: path to directory with all .csv files 
    :type input_path: str
    :param output_path: path to output directory 
    :type output_path: str
    :param pattern: the pattern of files to include
    :type pattern: str
    :return: PCA matrix
    :rtype: np.ndarray
    """
    # list of all .csv files in the directory
    paths_list = []
    for dir, _, files in os.walk(input_path):
        paths_list += [os.path.join(dir, name) for name in files
                          if fnmatch.fnmatch(name, pattern)]

    phases = np.zeros((brain_areas, t_phases))
    iFC = np.zeros((brain_areas, brain_areas))
    pca_components = np.zeros((n_subjects, t_phases, brain_areas))

    for path in paths_list:
        array = np.genfromtxt(path, delimiter=',')
        for area in range(0, brain_areas):
            time_series = pylab.demean(signal.detrend(array[:, area]))
            phases[area, :] = np.angle(signal.hilbert(time_series))
            for t in range(0, t_phases):
                for i in range(0, brain_areas):
                    for z in range(0, brain_areas):
                        if np.absolute(phases[i, t] - phases[z, t]) > np.pi:
                            iFC[i, z] = np.cos(2 * np.pi - np.absolute(
                                phases[i, t] - phases[z, t]))
                        else:
                            iFC[i, z] = np.absolute(phases[i, t] - phases[z, t])
                pca = PCA(n_components=2)
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
                with open(os.path.join(output_path, 'PCA_results_{}'.format(t)))\
                        as output:
                    json.dump(pca_dict, output)

                pca_components[paths_list.index(path), t, :] = \
                    pca_dict['components']
    # save the PCA matrix into a .csv file
    np.savetxt(os.path.join(output_path, 'PCA_components_matrix'),
               pca_components, delimiter=',')
    return pca_components


def dynamic_functional_connectivity(input_path, output_path):
    """
    Computes the functional connectivity dynamics of brain areas.

    :param input_path: path to directory with PCA matrix
    :type input_path: str
    :param output_path: path to output directory 
    :type output_path: str
    :return: FCD matrix
    :rtype: np.ndarray
    """
    FCD = np.zeros((n_subjects, t_phases - 2, t_phases - 2))
    pca_components = np.genfromtxt(input_path, delimiter=',')

    # Compute the FCD matrix for each subject as cosine similarity over time
    for subject in range(0, n_subjects):
        for t1 in range(0, t_phases):
            vec_1 = np.squeeze(pca_components[subject, t1, :])
            for t2 in range(0, t_phases):
                vec_2 = np.squeeze(pca_components[subject, t2, :])
                FCD[subject, t1, t2] = np.dot(vec_1, vec_2) / np.linalg.norm(
                    vec_1) / np.linalg.norm(vec_2)
    np.savetxt(os.path.join(output_path, 'FCD_matrix'),
               FCD, delimiter=',')
    return FCD
