# Direct Extraction of Signal and Noise Correlations from Two-Photon Calcium Imaging of Ensemble Neuronal Activity

This repository is the official implementation of 'Direct Extraction of Signal and Noise Correlations from Two-Photon Calcium Imaging of Ensemble Neuronal Activity'.

## Requirements

This code was originally implemented in MATLAB R2017B. However, it should run in most versions without a problem.

## Instructions

1. Download all the codes in a single directory
2. Run the script main.m 

## Results

Following these steps will regenerate the Figures (Figure 2) and results outlined in the Simulation study 1, in the Results section. 

## Contents

The tasks performed by each MATLAB function are explained in detail in their headers and comments. In summary:

1. main.m : Master script
2. generate_signals.m: Generates the simulated data
3. proposed_estimation_procedure.m: The implementation of the proposed iterative procedure
4. performance_evaluation.m: Used for Hyper-Parameter tuning
5. redblue.m: colormap for correlation plots
6. spike_deconvolution_FCSS.m: Generates the deconvolved spikes for two-stage estimates


