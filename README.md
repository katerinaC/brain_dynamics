### Script for a published paper:

Capouskova, K ., Kringelbach, M.L. & Deco, G. Modes of cognition: Evidence from 
metastable brain dynamics. Neuroimage 260, 119489 (2022). doi: 10.1016/J.NEUROIMAGE.2022.119489


# Functional Connectivity (FC) States

This repository contains Python code for calculating fMRI's
brain's functional connectivity states as identified by clustering model.

Running should be done via `main_FC_states.py` script for dynamical FC, FC dynamics,
FC states (kmeans or hidden markov model), and states features.

To get specific results for the states characteristics, running should be done
via `main_states_features.py`. This will generate lifetimes and
probabilities of states and it estimate the p-value between two conditions.

To get averaged DFCs and also separate dfcs according to states for further graph
analysis, run `main_DFC_features.py`. 

For graph analysis on dfcs in different states(clusters) run 
`main_graph_analysis.py`.


### Running the script

To run a main script, open your command line, navigate to the directory and run
it according to the arguments. Note that you have to have installed all required
packages as indicated in requirements.txt, preferably within a designated virtual
environment. 

#### Example for FC states:

`python main_FC_states.py --input users/name/data/task1 users/name/data/task2
--output users/name/data/tasks_output --areas 66 --pca --clusters 4`

#### Example for states features:

`python main_states_features.py --input users/name/data/concatenated_clusters.npz
--output users/name/data/tasks_output/states --n_clusters 4
--starts users/name/data/tasks_output/starts.json --separate
--clusters users/name/data/tasks_output/clusters.npz`

#### Example for DFC features:

`python main_DFC_features.py --input users/name/data/
--output users/name/data/tasks_output/dfc_out --n_clusters 4
--starts users/name/data/tasks_output/starts.json 
--clusters users/name/data/tasks_output/clusters.npz
--features 66 --names users/name/data/brain_areas.npy`


#### Example for graph analysis:

`python main_graph_analysis.py --input users/name/data/dfc_out
--output users/name/data/tasks_output/graph_analysis --n_clusters 4
--tasks flanker, rest, n_back`

### Data format

Data should be stored either in `.csv` or `.mat` file formats. For each subject
there should be a separate file (time x brain areas).

### Notes
Check for the number of components in dim. reduction.

Also, if running for different data sets, check the number of brain areas. Ideally,
the number of brain areas and the brain areas themselves should correspond.


Questions about the code can be addressed at: kcapouskova[at]hotmail.com
