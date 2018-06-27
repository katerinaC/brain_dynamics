# Functional Connectivity (FC) States

This repository contains Python code for calculating fMRI's
brain's functional connectivity states as identified by clustering model.

Running should be done via `main_FC_states.py` script for dynamical FC, FC dynamics,
FC states (kmeans or hidden markov model), and states features.

To get specific results for the states characteristics, running should be done
via `main_states_features.py`. This will generate lifetimes and
probabilities of states and it estimate the p-value between two conditions.


### Running the script

To run a main script, open your command line, navigate to the directory and run
it according to the arguments.

#### Example for FC states:

`python main_FC_states.py --input users/name/data/task1 users/name/data/task2
--output users/name/data/tasks_output --areas 66 --pca --clusters 4`

#### Example for states features:

`python main_states_features.py --input users/name/data/concatendated_clusters.npz
--output users/name/data/tasks_output/states --n_clusters 4
--starts users/name/data/tasks_output/starts.json --separate
--clusetrs users/name/data/tasks_output/clusters.npz`

### Data format

Data should be stored either in `.csv` or `.mat` file formats. For each subject
there should be a separate file (time x brain areas).

### Notes

Two parameters are in the code as they are usually the same, however, check for
for number of components in dim. reduction and TR in mean
lifetime of states (default 2 for both).

Also, if running for different datasets, check the number of brain areas. Ideally,
the number of brain areas and the brain areas themselves should correspond.


Questions about the code can be addressed at: kcapouskova[at]hotmail.com
