# Tumor Growth Model Comparison: ODE vs IDE

This project implements a comprehensive comparison between classical Ordinary Differential Equation (ODE) models and Impulsive Differential Equation (IDE) models for tumor growth with radiation therapy, based on the methodology from Laleh et al. (2022).

## IMPORTANT NOTE

As we now have the real data we are using, keep these things in mind:
- Treatment is given on days 1, 22, 43, 64, etc. (day 1, then again at the beginning of every 21 day cycle)
    * Thus, this is when impulses must be represented!
- There are two sources of data
    * One is for advanced/metastatic NSCLC after failed platinum based chemo
    * One is for advanced/metastatic squamous NSCLC after failed platinum based chemo
- Both of them: 
    * Are on the 21 day cycles with treatment on day 1 of each cycle (i.e. 1,, 22, 43, etc.) as stated above
    * Treat with 75mg Docetaxel (a Taxane chemotherapeutic agent) per square meter, IV
    * Treatment continues until the first one of these events occurs:
        + Disease Progression
        + Death
        + Unacceptable toxicity
        + Withdrawal of consent
        + Study termination


## Helpful Links

Information about RECIST data: https://radiologyassistant.nl/more/recist-1-1/recist-1-1-1  
Laleh Paper: https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1009822  
Laleh Paper Code: https://github.com/KatherLab/ImmunotherapyModels (Latest commit is broken)
Laleh Paper Code Forked Repo: https://github.com/mstracci/ImmunotherapyModels.git

## Overview

The project compares six classical tumor growth models:
- Exponential
- Logistic
- Classic Bertalanffy
- General Bertalanffy
- Classic Gompertz
- General Gompertz
Each model is implemented in both ODE (continuous) and IDE (with discrete treatment impulses) versions to assess whether treating the therapy as having a discrete impulsive effect on the tumor volume improves model accuracy and predictive power.


## Key Features

- **Synthetic Data Generation**: Creates realistic tumor volume data with treatment schedules
- **Model Implementation**: Complete implementation of all six classical models
- **IDE Conversion**: Conversion of ODE models to IDE versions with radiation impulses
- **Parameter Fitting**: Robust parameter estimation using differential evolution
- **Early Prediction**: Tests model performance with limited early data
- **Statistical Analysis**: MAE, MSE, RMSE for fit (sort out more details later, collect all for now)
- **Visualization**: Publication-ready figures and plots


## Quick Start

> python two_experiments_comparison.py

This runs both experiments from the paper:
- **Experiment 1**: Goodness of fit using all available data
- **Experiment 2**: Early prediction using only first half of data points


## Model Details

### ODE Models
Classical continuous models where treatment effects are modeled as continuous modifications of growth parameters (e.g., reduced growth rate).

### IDE Models
Impulsive differential equation versions that incorporate discrete therapy sessions (assumed to be one day following the time of the RECIST scan) with immediate tumor volume reduction at treatment times.


## Key Results

The analysis provides:
1. **Model Comparison**: All 6 classical tumor growth models fitted to real patient data
2. **ODE vs IDE Comparison**: Both models fitted to same patient data during treatment
3. **Two Experiments**: Goodness of fit and early prediction analysis
4. **Clinical Validation**: Do IDE models outperform ODE models??


### Key Findings:

- unknown at present


## Methodology

Based on the methodology from:
> Laleh, N. G., et al. (2022). "Classical mathematical models for prediction of response to chemotherapy and immunotherapy." PLOS Computational Biology.

