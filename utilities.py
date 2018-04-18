"""
Utilities file for diffenet operations to be used in the processing scripts.

Katerina Capouskova 2018, kcapouskova@hotmail.com
"""

import os
import scipy.io
import numpy as np
from tqdm import tqdm


def load_mat_save_as_csv(input_path, output_path):
    """
    Loads the .mat file and saves it as a .csv file or files.

    :param input_path: path to the .mat file
    :type input_path: str
    :param output_path: path where to save the .csv file/s
    :type output_path: str
    :return: full output path of the .csv file/s
    :rtype: str
    """
    if os.path.isdir(output_path):
        pass
    else:
        os.makedirs(output_path)
    data = scipy.io.loadmat(input_path)
    for key in tqdm(data):
        if '__' not in key and 'readme' not in key and 'Subjects' not in key:
            np.savetxt((os.path.join(output_path, 'data', key + '.csv')), data[key],
                       delimiter=',')
    return os.path.join(output_path, 'data')


def return_paths_list(input_path, output_path, pattern):
    """
    Loads the .mat file and saves it as a .csv file or files.

    :param input_path: path to the files directory
    :type input_path: str
    :param output_path: path where to save the .csv file/s
    :type output_path: str
    :param pattern: the pattern of files to include
    :type pattern: str
    :return: list of all paths in a directory
    :rtype: []
    """
    # list of all .csv files in the directory
    paths_list = []
    if pattern == '.csv':
        for directory, _, files in os.walk(input_path):
            paths_list += [os.path.join(directory, file) for file in files
                           if file.endswith(pattern)]
    elif pattern == '.mat':
        csv_path = load_mat_save_as_csv(input_path, output_path)
        for directory, _, files in os.walk(csv_path):
            paths_list += [os.path.join(directory, file) for file in files
                           if file.endswith(pattern)]
    return paths_list


def trasform_data(input_path, output_path, n_subjects, n_tasks):
    """
    Loads data in npy format and outputs them in a desired format.
    For each subject get Time X Brain Areas matrix as a .csv file

    :param input_path: path to the file 
    :type input_path: str
    :param output_path: path where to save the .csv file/s
    :type output_path: str
    :param n_subjects: number of subjects
    :type n_subjects: int
    :param n_tasks: number of tasks 
    :type n_tasks: int
    """
    if os.path.isdir(output_path):
        pass
    else:
        os.makedirs(output_path)
    data = np.load(input_path)
    data = np.swapaxes(data, 2, 3)
    print data.shape
    for subject in range(n_subjects):
        for task in range(n_tasks):
            np.savetxt(os.path.join(output_path, 'subject{}_task{}.csv'.format
            (subject, task)), data[subject, task, :, :], delimiter=',')
