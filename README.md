# GRU-TV
Implementation of Time and Velocity Perception GRU

# Dataset
Two databse used are public available
PhysioNet 2012: https://physionet.org/content/challenge-2012/1.0.0/
MIMIC-III: https://physionet.org/content/mimiciii/1.4/

The prrprocessing pipeline of MIMIC-III references to that provided in 
Harutyunyan, Hrayr, et al. "Multitask learning and benchmarking with clinical time series data." Scientific data 6.1 (2019): 96.
Also available in https://doi.org/10.5281/zenodo.1306527.

# Reproduce
Reproducing the result requires replacing the path placeholder in the code with the local path.
In scripts/data_process.py, utils for the statistical datasets features and dadatset sampling used in the paper are provided
