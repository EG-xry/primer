# Primer — Neural Spike Train Analysis Tutorial

## Overview

This project is a MATLAB-based neural spike train analysis tutorial codebase, designed to accompany related academic publications. Through three real neuroscience datasets, the project systematically demonstrates the complete analysis pipeline from raw data to model fitting and prediction validation, covering multiple classical neural coding analysis methods.

## Datasets

The project contains three experimental datasets from different levels of the nervous system:

| Dataset | Series | Neural System Region | Data Directory | Description |
|---------|--------|---------------------|----------------|-------------|
| **Retina** | 100 series | Retinal ganglion cells | `RetinaData/` | White noise checkerboard stimuli and spike trains from 53 cells |
| **Whisker/Thalamus** | 200 series | Rat thalamic VPM region | `WhiskerData/` | Whisker position and spike train recordings from 7 thalamic cells |
| **Motor Cortex** | 300 series | Motor cortex | `MotorData/` | Motor cortex neuronal population activity data |

## Analysis Methods

The project covers the following neural coding analysis methods:

- **STA (Spike-Triggered Average)** — Estimates the neuron's linear receptive field
- **STC (Spike-Triggered Covariance)** — Captures second-order features
- **MNE (Maximally Noise Entropy)** — Maximum noise entropy model for nonlinear feature extraction
- **GLM (Generalized Linear Model)** — Including:
  - Uncoupled GLM
  - Coupled GLM for modeling neural network interactions
- **Prediction & Validation** — Model prediction performance evaluation and cross-validation

## Project Structure

```
primer-master/
├── RetinaData/          # Retina experiment raw data
├── RetinaScripts/       # Retina data analysis scripts (100 series)
│   └── RetinaMNEGLM/    # MNE/GLM helper functions
├── WhiskerData/         # Whisker experiment raw data
├── WhiskerScripts/      # Whisker data analysis scripts (200 series)
│   └── WhiskerMNEGLM/   # MNE/GLM helper functions
├── MotorData/           # Motor cortex experiment raw data
├── MotorScripts/        # Motor cortex data analysis scripts (300 series)
│   ├── MotorGLM/        # GLM helper functions
│   └── L1Group/         # L1 regularization optimization tools
├── OtherScripts/        # Other general helper functions
├── docs/                # Documentation directory
└── readme.txt           # Original readme file
```

## Script Organization

Scripts for each dataset are divided into two categories:

- **`script_*`** — Data processing and analysis scripts: perform raw data preprocessing, model fitting, and other computations, saving results
- **`makefig_*`** — Plotting scripts: load saved analysis results and generate figures from the paper

> ⚠️ Execution order: You must first run the corresponding `script_*` to complete the analysis, then run `makefig_*` to generate figures. Some plotting scripts require results from multiple analysis scripts.

## Dependencies and Installation

1. Extract the project files to the target directory
2. Place data files in the corresponding data directories (`RetinaData/`, `WhiskerData/`, `MotorData/`)
3. Add helper function directories to the MATLAB search path (including `MotorGLM/`, `L1Group/`, `OtherScripts/`, etc.)
4. Compile L1Group MEX files: run `MotorScripts/L1Group/mexAll.m`
5. Install [Jonathan Pillow's GLM toolkit](https://github.com/pillowlab/GLMspiketools)
6. Install [Mark Schmidt's L1 optimization tools](https://www.cs.ubc.ca/~schmidtm/Software/minFunc.html)

> 💡 It is recommended to use MATLAB's **Cell Mode** or **Debug Mode** to step through scripts, allowing you to follow the data transformations alongside the formulas in the paper.

## Cross-Validation Scripts

Scripts in the 300 series with the `_crossval` suffix are used for five-fold cross-validation to select optimal regularization parameters for coupled models and estimate standard errors. Note that `script_303_MotorData_SimCoupledGLM_crossval.m` is computationally intensive and users should modify it to run across multiple CPUs/machines based on available computational resources. These scripts are provided as templates and are not guaranteed to work out of the box.

## License and Citation

This project is companion code for an academic paper. Please cite the relevant paper when using it. For detailed information about the whisker data, see:

> Moore JD, Mercer Lindsay N, Deschênes M, Kleinfeld D (2015) Vibrissa Self-Motion and Touch Are Reliably Encoded along the Same Somatosensory Pathway from Brainstem through Thalamus. *PLoS Biol* 13(9): e1002253.
