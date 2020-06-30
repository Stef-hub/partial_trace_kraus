# partial_trace_kraus

This repository contains the code for the paper "Partial Trace Regression and Low-Rank Kraus Decomposition" published at ICML 2020.

This repository is still in construction.

# Required environment
  - Anaconda3
  - TensorFlow 1.13.1 
  - Keras 2.2.4
  - torch 1.1.0

# Code to build models
  - KrausLayer.py
  - PSDReluLayer.py
  - spd2spd_kraus.py (PSD to PSD matrix regression, section 3.1)
  - completion_kraus.py (PSD matrix completion, section 3.2)

# To generate simulated data
  - make_data_spd2spd.py generates toy data for PSD to PSD matrix regression
  - make_data_completion.py generates toy data for PSD matrix completion
  - util.py is required for previous scripts to build kraus model

# To execute the code
* run for example (see the code files for args): 
   - python spd2spd_kraus.py 10000_20_10_5 0 5 2 15 1 0.1
   - python completion_kraus.py 90_20_10_0_4 0 50 2 50 1 0.1

