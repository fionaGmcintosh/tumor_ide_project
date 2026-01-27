import pandas as pd
from scipy.optimize import differential_evolution, curve_fit

from growth_models import (GrowthModelODE, GrowthModelIDE, exponential, logistic, classic_gompertz, general_gompertz,
                           classic_bertalanffy, general_bertalanffy)


def find_model_params(model, t_array, y_array):

    parameter_bounds = []
    for i in range(model.num_params):
        parameter_bounds.append((0, 1))
    params = differential_evolution(model.sse, tuple(parameter_bounds), args=(t_array, y_array), seed=42, strategy='best1bin').x

    params, _ = curve_fit(model.solution, t_array, y_array, params, maxfev=200000, method='trf')

    return params


def get_model(base_ode, model_type):
    if model_type == 'ode':
        return GrowthModelODE(base_ode)
    elif model_type == 'ide':
        return GrowthModelIDE(21, base_ode)
    else:
        raise ValueError('Unknown model type')


def perform_data_analysis(model_properties, data_path):

    data_frame = pd.read_csv(data_path)

    patient_info = data_frame[['patient_id', 'treatment_arm']].drop_duplicates().reset_index(drop=True)
    patient_data = data_frame[['patient_id', 'treatment_day', 'longest_diameter_mm']]

    patient_data = patient_data[patient_data['longest_diameter_mm'] != "NOT EVALUABLE"]
    patient_data[patient_data['longest_diameter_mm'] == "TOO SMALL TO MEASURE"] = 0

    results = {}
    for patient_id in patient_info['patient_id']:

        results[patient_id] = {}
        patient_data_subset = patient_data[patient_data['patient_id'] == patient_id]

        if len(patient_data_subset) >= 3:

            t_array = patient_data_subset['treatment_day'].to_numpy()
            y_array = patient_data_subset['longest_diameter_mm'].to_numpy()

            for model_name, model_dict in model_properties.items():
                model = get_model(model_dict['base_ode'], model_dict['model_type'])
                params = find_model_params(model, t_array, y_array)
                mae_best_fit = model.mae(params, t_array, y_array)

                results[patient_id][model_name] = {}
                results[patient_id][model_name]['mae_best_fit'] = mae_best_fit

        if len(patient_data_subset) >= 6:

            t_array = patient_data_subset['treatment_day'].to_numpy()
            y_array = patient_data_subset['longest_diameter_mm'].to_numpy()

            t_array_fit = t_array[:-3]
            y_array_fit = y_array[:-3]
            t_array_ext = t_array[-3:]
            y_array_ext = y_array[-3:]

            for model_name, model_dict in model_properties.items():

                model = get_model(model_dict['base_ode'], model_dict['model_type'])
                params = find_model_params(model, t_array_fit, y_array_fit)
                mae_prediction = model.mae(params, t_array_ext, y_array_ext)

                results[patient_id][model_name]['mae_prediction'] = mae_prediction


if __name__ == "__main__":

    MODEL_PROPERTIES = {
        'exponential':
            {'base_ode': exponential, 'model_type': 'ode'},
        'exponential_impulsive':
            {'base_ode': exponential, 'model_type': 'ide'},
        'logistic':
            {'base_ode': logistic, 'model_type': 'ode'},
        'logistic_impulsive':
            {'base_ode': logistic, 'model_type': 'ide'},
        'classic_bertalanffy':
            {'base_ode': classic_bertalanffy, 'model_type': 'ode'},
        'classic_bertalanffy_impulsive':
            {'base_ode': classic_bertalanffy, 'model_type': 'ide'},
        'general_bertalanffy':
            {'base_ode': general_bertalanffy, 'model_type': 'ode'},
        'general_bertalanffy_impulsive':
            {'base_ode': general_bertalanffy, 'model_type': 'ide'},
        'classic_gompertz':
            {'base_ode': classic_gompertz, 'model_type': 'ode'},
        'classic_gompertz_impulsive':
            {'base_ode': classic_gompertz, 'model_type': 'ide'},
        'general_gompertz':
            {'base_ode': general_gompertz, 'model_type': 'ode'},
        'general_gompertz_impulsive':
            {'base_ode': general_gompertz, 'model_type': 'ide'},
    }

    DATA_PATH = r"resources/real_data.csv"

    perform_data_analysis(MODEL_PROPERTIES, DATA_PATH)
