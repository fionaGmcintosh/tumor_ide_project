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

### Simple Demo (No Dependencies Required)
```bash
python simple_demo.py
```

This will run a basic demonstration showing the difference between ODE and IDE models.

### Full Analysis (Requires Dependencies)
```bash
python example_analysis.py
```

## File Structure

```
C:\Users\fiona\tumor_ide_project\
├── tumor_models.py          # Core model implementations
├── data_generator.py        # Synthetic data generation
├── model_fitting.py         # Parameter fitting and optimization
├── comparison_analysis.py   # Analysis and visualization tools
├── example_analysis.py      # Complete workflow example
├── simple_demo.py          # Simple demonstration (no dependencies)
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Model Details

### ODE Models
Classical continuous models for tumor growth without treatment effects.

### IDE Models
Impulsive differential equation versions that incorporate discrete radiation therapy sessions (5 days per week) with immediate tumor volume reduction.

### Radiation Therapy
- **Schedule**: 5 days per week for 8 weeks
- **Effect**: Each session reduces tumor volume by a specified fraction
- **Timing**: Discrete impulses at treatment times

## Key Results

The analysis provides:
1. **Fit Quality Comparison**: RMSE, MAE, and R² metrics
2. **Statistical Tests**: Paired t-tests and Wilcoxon tests
3. **Early Prediction**: Performance with limited data points
4. **Model Ranking**: Best performing models for different scenarios

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
