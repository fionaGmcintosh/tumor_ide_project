"""
Tumor Growth Models: ODE vs IDE Comparison
==========================================

This module implements classical ODE tumor growth models and their 
Impulsive Differential Equation (IDE) counterparts for radiation therapy.
"""

import numpy as np
from scipy.integrate import solve_ivp
from typing import List, Tuple, Dict


# ODE Model Definitions
def exponential(self, t: float, y: float, params: Dict) -> float:
    """Exponential growth: dy/dt = r*y"""
    r = params['r']
    return r * y

def logistic(self, t: float, y: float, params: Dict) -> float:
    """Logistic growth: dy/dt = r*y*(1 - y/K)"""
    r, K = params['r'], params['K']
    return r * y * (1 - y / K)

def classic_bertalanffy(self, t: float, y: float, params: Dict) -> float:
    """Classic Bertalanffy: dy/dt = a*y^(2/3) - b*y"""
    a, b = params['a'], params['b']
    return a * (y ** (2/3)) - b * y

def general_bertalanffy(self, t: float, y: float, params: Dict) -> float:
    """General Bertalanffy: dy/dt = a*y^m - b*y^n"""
    a, b, m, n = params['a'], params['b'], params['m'], params['n']
    return a * (y ** m) - b * (y ** n)

def gompertz(self, t: float, y: float, params: Dict) -> float:
    """Classic Gompertz: dy/dt = r*y*ln(K/y)"""
    r, K = params['r'], params['K']
    if y <= 0:
        return 0
    return r * y * np.log(K / y)

def general_gompertz(self, t: float, y: float, params: Dict) -> float:
    """General Gompertz: dy/dt = r*y*ln(K/y)^(1/m)"""
    r, K, m = params['r'], params['K'], params['m']
    if y <= 0:
        return 0
    return r * y * (np.log(K / y) ** (1/m))

def solve_ode(model_func, params: Dict, t_span: Tuple[float, float],
              y0: float, t_eval: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve ODE model numerically.

    Args:
        model_func: Model function handle
        params: Model parameters
        t_span: Time span (t_start, t_end)
        y0: Initial condition
        t_eval: Time points for evaluation

    Returns:
        t: Time points
        y: Solution values
    """

    # Handle special cases for parameters
    if model_name in ['general_bertalanffy', 'general_gompertz']:
        # Ensure m, n are positive
        if 'm' in params and params['m'] <= 0:
            params['m'] = 0.1
        if 'n' in params and params['n'] <= 0:
            params['n'] = 0.1

    sol = solve_ivp(
        lambda t, y: model_func(t, y, params),
        t_span,
        [y0],
        t_eval=t_eval,
        method='RK45',
        rtol=1e-6,
        atol=1e-8
    )

    return sol.t, sol.y[0]

def solve_ide(model_func, params: Dict, t_span: Tuple[float, float],
              y0: float, treatment_times: List[float], impulse_strength: float = 0.1,
              t_eval: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve IDE model with radiation impulses.

    :param model_func: Model function handle
    :param params: Model parameters
    :param t_span: Time span (t_start, t_end)
    :param y0: Initial condition
    :param treatment_times: Times when radiation is administered
    :param impulse_strength: Fraction of tumor killed per treatment (0-1)
    :param t_eval: Time points for evaluation

    :return: Time points, Solution values
    """

    # Create time points including treatment times
    if t_eval is None:
        t_eval = np.linspace(t_span[0], t_span[1], 1000)

    # Add treatment times to evaluation points
    all_times = np.sort(np.unique(np.concatenate([t_eval, treatment_times])))

    # Solve piecewise between treatment times
    t_solution = []
    y_solution = []

    current_y = y0
    current_t = t_span[0]

    for i, t_end in enumerate(all_times):
        if t_end <= current_t:
            continue

        # Solve ODE between current_t and t_end
        if t_end in treatment_times:
            # Solve up to treatment time
            t_temp = np.linspace(current_t, t_end, 100)
            _, y_temp = solve_ode(model_func, params, (current_t, t_end), current_y, t_temp)

            # Apply impulse (radiation kills fraction of tumor)
            current_y = current_y * (1 - impulse_strength)

            t_solution.extend(t_temp)
            y_solution.extend(y_temp)
        else:
            # Solve normally
            t_temp = np.linspace(current_t, t_end, 100)
            _, y_temp = solve_ode(model_func, params, (current_t, t_end), current_y, t_temp)

            t_solution.extend(t_temp)
            y_solution.extend(y_temp)
            current_y = y_temp[-1]

        current_t = t_end

    return np.array(t_solution), np.array(y_solution)

def generate_treatment_schedule(start_day: int = 0, duration_weeks: int = 8,
                               days_per_week: int = 5) -> List[float]:
    """
    Generate radiation treatment schedule (5 days per week).

    Args:
        start_day: Day to start treatment
        duration_weeks: Number of weeks of treatment
        days_per_week: Number of treatment days per week (default 5)

    Returns:
        List of treatment times in days
    """
    treatment_times = []

    for week in range(duration_weeks):
        for day in range(days_per_week):
            treatment_day = start_day + week * 7 + day
            treatment_times.append(treatment_day)

    return treatment_times


if __name__ == '__main__':
    models = [exponential, logistic, classic_bertalanffy, general_bertalanffy, gompertz, general_gompertz]

    for model in models:
        solve_ide(model, {'r': 0.05, 'K': 0.05, 'm': 0.05, 'n': 0.05})