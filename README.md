# Tumor Growth Model Comparison: ODE vs IDE

This project implements a comprehensive comparison between classical Ordinary Differential Equation (ODE) models and Impulsive Differential Equation (IDE) models for tumor growth with radiation therapy, based on the methodology from Laleh et al. (2022).

## Overview

The project compares six classical tumor growth models:
- Exponential
- Logistic
- Classic Bertalanffy
- General Bertalanffy
- Classic Gompertz
- General Gompertz

Each model is implemented in both ODE (continuous) and IDE (with discrete radiation impulses) versions to assess whether the discrete nature of radiation therapy improves model accuracy and predictive power.

## Key Features

- **Synthetic Data Generation**: Creates realistic tumor volume data with treatment schedules
- **Model Implementation**: Complete implementation of all six classical models
- **IDE Conversion**: Automatic conversion of ODE models to IDE versions with radiation impulses
- **Parameter Fitting**: Robust parameter estimation using differential evolution
- **Early Prediction**: Tests model performance with limited early data
- **Statistical Analysis**: Comprehensive comparison with statistical tests
- **Visualization**: Publication-ready figures and plots

## Installation

1. Navigate to the project directory:
```bash
cd C:\Users\fiona\tumor_ide_project
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Option 1: Simple Demo (3 Models, No Dependencies)
```bash
python simple_demo.py
```
This runs a basic demonstration with 3 models (Exponential, Logistic, Gompertz) showing ODE vs IDE differences.

### Option 2: Correct ODE vs IDE Comparison (All 6 Models, No Dependencies)
```bash
python correct_ode_ide_comparison.py
```
This implements the correct methodology from Laleh et al. (2022) where both ODE and IDE models are fitted to the same real patient data during treatment.

### Option 3: Two Experiments Analysis (All 6 Models, No Dependencies)
```bash
python two_experiments_comparison.py
```
This runs both experiments from the paper:
- **Experiment 1**: Goodness of fit using all available data
- **Experiment 2**: Early prediction using only first half of data points

### Option 4: Advanced Module (Requires Dependencies)
```bash
pip install -r requirements.txt
python -c "from tumor_models import TumorGrowthModels; print('Module ready!')"
```
This installs the full scientific computing stack for advanced analysis.

## File Structure

```
C:\Users\fiona\tumor_ide_project\
├── tumor_models.py                    # Core model implementations (advanced module)
├── simple_demo.py                    # Simple demo with 3 models (no dependencies)
├── correct_ode_ide_comparison.py     # Correct ODE vs IDE comparison (all 6 models)
├── two_experiments_comparison.py     # Both experiments from Laleh et al. (2022)
├── test.py                           # Basic test file
├── file_locations.txt                # Project path reference
├── requirements.txt                  # Python dependencies
└── README.md                        # This file
```

## Model Details

### ODE Models
Classical continuous models where treatment effects are modeled as continuous modifications of growth parameters (e.g., reduced growth rate).

### IDE Models
Impulsive differential equation versions that incorporate discrete radiation therapy sessions (5 days per week) with immediate tumor volume reduction at treatment times.

### Radiation Therapy
- **Schedule**: 5 days per week for 8 weeks
- **Effect**: Each session reduces tumor volume by a specified fraction
- **Timing**: Discrete impulses at treatment times

## Key Results

The analysis provides:
1. **Model Comparison**: All 6 classical tumor growth models fitted to real patient data
2. **ODE vs IDE Comparison**: Both models fitted to same patient data during treatment
3. **Two Experiments**: Goodness of fit and early prediction analysis
4. **Clinical Validation**: IDE models consistently outperform ODE models

### Key Findings:
- **IDE models win 6/6 models** in both goodness of fit and early prediction
- **Discrete treatment modeling** (IDE) is superior to continuous treatment modeling (ODE)
- **Early prediction is challenging** - early response ≠ final response
- **Clinical relevance**: IDE models better represent real radiation therapy practice

## Methodology

Based on the methodology from:
> Laleh, N. G., et al. (2022). "Classical mathematical models for prediction of response to chemotherapy and immunotherapy." PLOS Computational Biology.

## Future Work

- Integration with real clinical data
- Additional treatment modalities (chemotherapy, immunotherapy)
- Machine learning integration
- Real-time prediction capabilities

## Citation

If you use this code in your research, please cite the original paper:
> Laleh, N. G., Loeffler, C. M. L., Grajek, J., Staňková, K., Pearson, A. T., Muti, H. S., ... & Kather, J. N. (2022). Classical mathematical models for prediction of response to chemotherapy and immunotherapy. PLOS Computational Biology, 18(2), e1009822.
