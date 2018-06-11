# Functional Connectivity (FC) States

This repository contains Python code for calculating 
brain's functional connectivity states as identified by clustering model.

Running should be done via `main_FC_states.py` script for dynamical FC, FC dynamics,
FC states (kmeans or hidden markov model), and states features.

Two parameters are in the code as they are usually the same, however, check for
for number of components in dim. reduction and TR in mean
lifetime of states (default 2 for both).

Also, if running for different datasets, check the number of brain areas. Ideally,
the number of brain areas and the brain areas themselves should correspond.

Questions about the code can be addressed at: kcapouskova@hotmail.com
