import os

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, curve_fit
import matplotlib.pyplot as plt

from growth_models import (GrowthModelODE, GrowthModelIDE, exponential, logistic, classic_gompertz, general_gompertz,
                           classic_bertalanffy, general_bertalanffy)


def get_model(base_ode, model_type):
    """Construct a DE-based model given a base ODE and the type of model to construct.

    :param base_ode: The base ODE of the model in function form
    :param model_type: String describing the type of DE model to return
    :return: :class:'~growth_models.GrowthModel' object constructed to specifications
    """
    if model_type == 'ode':
        return GrowthModelODE(base_ode)
    elif model_type == 'ide':
        return GrowthModelIDE(21, base_ode)
    else:
        raise ValueError('Unknown model type')


def find_model_params(model, t_array, y_array):
    """Estimate the DE parameters that result in the best fit of the model to the given data.

    :param model: Model to be fit to the given actual data
    :param t_array: Array of time values at which data points were collected
    :param y_array: Array of actual data points
    :return: Tuple of DE parameters that result in the best fit (minimum sum of squared errors)
    """
    # Create a bounds object for each parameter to be used for differential evolution
    parameter_bounds = []
    for i in range(model.num_params):
        parameter_bounds.append((0, 1))

    # Stochastically estimate initial parameter guesses with and iteratively improve parameter fit with scipy
    params = differential_evolution(model.sse, tuple(parameter_bounds), args=(t_array, y_array),
                                    seed=42, strategy='best1bin').x
    params, _ = curve_fit(model.solution, t_array, y_array, params, maxfev=1000, method='trf')
    return params


# TODO: This function is for arbitrary models on one plot, do we want specific like ODE vs IDE or fit vs prediction?
def plot_models(models, param_sets, names, t_array, y_array, save_dir=None, titles=None):
    """Takes a set of models and their parameters and plots their solutions against a scatter plot of actual data.

    :param models: List of models to be plotted over the interval described by the bounds of the given time data
    :param param_sets: List of DE parameter tuples to be passed to each respective model for fitting
    :param names: List of strings describing the names of the models to be plotted
    :param t_array: Array of time values at which data points were collected
    :param y_array: Array of actual data points
    :param save_dir: Path to directory where plot is saved, plot will be shown if none is provided
    :param titles: Tuple of titles for the plot, the x-axis, the y-axis respectively, will be generated if none provided
    """
    # Plot experimental data and create time interval based on bounds of actual data
    fig, ax = plt.subplots()
    ax.plot(t_array, y_array, marker='.', linestyle='None', label='actual data')
    t_array_pred = np.linspace(t_array[0], t_array[-1], 100)

    # Plot model predictions
    for model, params, name in zip(models, param_sets, names):
        rmse = model.rmse(params, t_array, y_array)
        y_array_pred = model.solution(t_array_pred, *params)
        ax.plot(t_array_pred, y_array_pred, label='%s (RMSE: %s)' % (name, rmse))

    # Format plot
    plt.tight_layout()
    ax.set_ylim(bottom=0)
    if titles is None:
        title = 'experimental data'
        for name in names:
            title += ' vs %s' % name
        x_title = 't'
        y_title = 'y'
    else:
        title, x_title, y_title = titles
    ax.set_title(title)
    ax.set_xlabel(x_title)
    ax.set_ylabel(y_title)
    ax.grid(True, alpha=0.5)
    ax.legend()

    # Save or show plot as image
    if save_dir:
        plt.savefig(os.path.join(save_dir, '%s.png' % title), bbox_inches='tight')
    else:
        plt.show()


def perform_data_analysis(data_path, model_properties, num_plots=5, save_dir=None):
    """Perform main data analysis task including the best fit experiment and the predictive power experiment.

    :param data_path: Path to CSV file containing actual dataset
    :param model_properties: Dictionary mapping the name of each model to its base ODE function and its model type
    :param num_plots: Number of patient plots to display with best fit and prediction results
    :param save_dir: Path to directory where plot is saved, plot will be shown if none is provided
    :return: Nested dictionary of MAE results per patient, per model, per experiment
    """
    # Load datasheet and break down into table for identifying patient treatment arm, table for patient data points
    data_frame = pd.read_csv(data_path)
    # TODO: 2NF database normalization, patient ID is key to data table and arm table, do we need to preserve arm?
    patient_info = data_frame[['patient_id', 'study_arm']].drop_duplicates().reset_index(drop=True)
    patient_data = data_frame[['patient_id', 'treatment_day', 'longest_diameter_mm']]

    # Set 'TOO SMALL TO MEASURE' entries to 0, remove any other data points with string entries
    patient_data.loc[patient_data['longest_diameter_mm'] == "TOO SMALL TO MEASURE", 'longest_diameter_mm'] = str(0)
    patient_data['longest_diameter_mm'] = pd.to_numeric(patient_data['longest_diameter_mm'], errors='coerce')
    patient_data = patient_data.dropna(subset=['longest_diameter_mm'])

    # Initialize nested dictionary of results and iterate through each patient in the datasheet
    results = {}
    for i, patient_id in enumerate(patient_info['patient_id']):
        # Initialize dictionary of results for this patient and extract their time and normalized tumor volume data
        results[patient_id] = {}
        patient_data_subset = patient_data[patient_data['patient_id'] == patient_id]
        t_array = patient_data_subset['treatment_day'].to_numpy()
        vol_array = 0.5 * (patient_data_subset['longest_diameter_mm'].to_numpy()) ** 3
        # TODO: Should we be normalizing per patient baseline? In study use full dataset max, here we use patient max
        # vol_array /= vol_array[0]
        vol_array /= vol_array.max()

        for model_name, model_dict in model_properties.items():
            # Iterate through and construct each type of model and check number of data points
            model = get_model(model_dict['base_ode'], model_dict['model_type'])
            param_sets = []

            # TODO: Should we be pre-splitting datasets here? This seems more efficient but a little sloppier
            if len(patient_data_subset) >= 3:
                # If this patient has more than 3 data points, fit parameters to all data and store MAE
                params_fit = find_model_params(model, t_array, vol_array)
                param_sets.append(params_fit)
                results[patient_id][model_name] = {'mae_fit': model.mae(params_fit, t_array, vol_array)}

                if len(patient_data_subset) >= 6:
                    # If more than 6 data points, hold out last 3 points, fit to remaining data, and store MAE
                    t_array_fit = t_array[:-3]
                    vol_array_fit = vol_array[:-3]
                    params_pred = find_model_params(model, t_array_fit, vol_array_fit)
                    param_sets.append(params_pred)
                    results[patient_id][model_name]['mae_prediction'] = model.mae(params_pred, t_array, vol_array)

                    if i < num_plots:
                        plot_models([model] * 2, param_sets, ['fit', 'prediction'], t_array, vol_array,
                                    save_dir=save_dir,
                                    titles=(str(patient_id), 'time (days)', 'volume (baseline norm)'))

            print('Model %s complete' % model_name)
        print('Patient %s complete' % patient_id)
    return results


if __name__ == "__main__":

    DATA_PATH = r"resources/real_data.csv"
    MODEL_PROPERTIES = {
        'exponential':
            {'base_ode': exponential, 'model_type': 'ode'},
        'exponential_impulsive':
            {'base_ode': exponential, 'model_type': 'ide'},
        'logistic':
            {'base_ode': logistic, 'model_type': 'ode'},
        'logistic_impulsive':
            {'base_ode': logistic, 'model_type': 'ide'},
        'classic_gompertz':
            {'base_ode': classic_gompertz, 'model_type': 'ode'},
        'classic_gompertz_impulsive':
            {'base_ode': classic_gompertz, 'model_type': 'ide'},
        'general_gompertz':
            {'base_ode': general_gompertz, 'model_type': 'ode'},
        'general_gompertz_impulsive':
            {'base_ode': general_gompertz, 'model_type': 'ide'},
        'classic_bertalanffy':
            {'base_ode': classic_bertalanffy, 'model_type': 'ode'},
        'classic_bertalanffy_impulsive':
            {'base_ode': classic_bertalanffy, 'model_type': 'ide'},
        'general_bertalanffy':
            {'base_ode': general_bertalanffy, 'model_type': 'ode'},
        'general_bertalanffy_impulsive':
            {'base_ode': general_bertalanffy, 'model_type': 'ide'}
    }
    NUM_PLOT = 5
    SAVE_DIR = 'results'

    if os.path.exists(SAVE_DIR):
        os.rmdir(SAVE_DIR)
    os.mkdir(SAVE_DIR)

    RESULTS = perform_data_analysis(DATA_PATH, MODEL_PROPERTIES, NUM_PLOT, SAVE_DIR)
