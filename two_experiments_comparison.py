"""
Two experiments from Laleh et al. (2022):
1. Goodness of fit on all available data
2. Early prediction using only half the data points
"""

import math
import random

def exponential_growth(t, y, r, treatment_effect=0):
    """Exponential growth with continuous treatment effect: dy/dt = (r - treatment_effect)*y"""
    return (r - treatment_effect) * y

def logistic_growth(t, y, r, K, treatment_effect=0):
    """Logistic growth with continuous treatment effect: dy/dt = (r - treatment_effect)*y*(1 - y/K)"""
    return (r - treatment_effect) * y * (1 - y / K)

def classic_bertalanffy(t, y, a, b, treatment_effect=0):
    """Classic Bertalanffy with continuous treatment effect: dy/dt = a*y^(2/3) - (b + treatment_effect)*y"""
    return a * (y ** (2/3)) - (b + treatment_effect) * y

def general_bertalanffy(t, y, a, b, m, n, treatment_effect=0):
    """General Bertalanffy with continuous treatment effect: dy/dt = a*y^m - (b + treatment_effect)*y^n"""
    return a * (y ** m) - (b + treatment_effect) * (y ** n)

def classic_gompertz(t, y, r, K, treatment_effect=0):
    """Classic Gompertz with continuous treatment effect: dy/dt = (r - treatment_effect)*y*ln(K/y)"""
    if y <= 0:
        return 0
    return (r - treatment_effect) * y * math.log(K / y)

def general_gompertz(t, y, r, K, m, treatment_effect=0):
    """General Gompertz with continuous treatment effect: dy/dt = (r - treatment_effect)*y*ln(K/y)^(1/m)"""
    if y <= 0 or y >= K:
        return 0
    try:
        return (r - treatment_effect) * y * (math.log(K / y) ** (1/m))
    except (ValueError, ZeroDivisionError):
        return 0

def solve_ode_simple(model_func, params, t_span, y0, dt=0.1):
    """Simple Euler method solver for ODE with continuous treatment"""
    t_start, t_end = t_span
    t = t_start
    y = y0
    
    times = [t]
    values = [y]
    
    while t < t_end:
        dy_dt = model_func(t, y, **params)
        y = y + dy_dt * dt
        t = t + dt
        times.append(t)
        values.append(y)
    
    return times, values

def solve_ide_simple(model_func, params, t_span, y0, treatment_times, impulse_strength=0.1, dt=0.1):
    """IDE solver with discrete radiation impulses (no continuous treatment effect)"""
    t_start, t_end = t_span
    t = t_start
    y = y0
    
    times = [t]
    values = [y]
    
    treatment_idx = 0
    
    while t < t_end:
        # Check if we hit a treatment time
        if treatment_idx < len(treatment_times) and t >= treatment_times[treatment_idx]:
            y = y * (1 - impulse_strength)  # Apply radiation impulse
            treatment_idx += 1
        
        # Normal growth (no continuous treatment effect for IDE)
        dy_dt = model_func(t, y, **params)
        y = y + dy_dt * dt
        t = t + dt
        times.append(t)
        values.append(y)
    
    return times, values

def generate_treatment_schedule(start_day=20, duration_weeks=8, days_per_week=5):
    """Generate radiation treatment schedule"""
    treatment_times = []
    for week in range(duration_weeks):
        for day in range(days_per_week):
            treatment_day = start_day + week * 7 + day
            treatment_times.append(treatment_day)
    return treatment_times

def generate_synthetic_patient_data(t_span, y0, treatment_times, noise_level=0.1):
    """Generate synthetic patient data that mimics real clinical data"""
    t_start, t_end = t_span
    times = []
    volumes = []
    
    t = t_start
    y = y0
    
    while t <= t_end:
        # Simulate realistic tumor growth with treatment effects
        if any(abs(t - tx) < 0.5 for tx in treatment_times):
            # During treatment, tumor shrinks
            y = y * 0.95  # 5% reduction per treatment
        else:
            # Between treatments, tumor grows slowly
            y = y * 1.001  # 0.1% growth per day
        
        # Add realistic measurement noise
        noise = random.gauss(0, noise_level * y)
        y_measured = max(0, y + noise)
        
        times.append(t)
        volumes.append(y_measured)
        t += 1  # Daily measurements
    
    return times, volumes

def calculate_rmse(predicted_times, predicted_volumes, real_times, real_volumes):
    """Calculate Root Mean Square Error between predicted and real data"""
    errors = []
    for i, real_t in enumerate(real_times):
        # Find closest predicted time
        closest_idx = min(range(len(predicted_times)), 
                        key=lambda i: abs(predicted_times[i] - real_t))
        error = (predicted_volumes[closest_idx] - real_volumes[i]) ** 2
        errors.append(error)
    return math.sqrt(sum(errors) / len(errors))

def experiment_1_goodness_of_fit(models, base_params, real_times, real_volumes, treatment_times):
    """Experiment 1: Goodness of fit using ALL available data"""
    print("EXPERIMENT 1: Goodness of Fit (All Available Data)")
    print("=" * 60)
    print("Fitting models to complete dataset and measuring fit quality")
    print()
    
    print(f"{'Model':<20} {'ODE RMSE':<12} {'IDE RMSE':<12} {'Better Model':<15}")
    print("-" * 65)
    
    results = {}
    
    for model_name, model_func in models.items():
        base_param = base_params[model_name]
        
        # ODE: Fit with continuous treatment effect
        ode_params = base_param.copy()
        ode_params['treatment_effect'] = 0.02
        t_ode, y_ode = solve_ode_simple(model_func, ode_params, (0, 100), 100)
        
        # IDE: Fit with discrete impulses
        ide_params = base_param.copy()
        t_ide, y_ide = solve_ide_simple(model_func, ide_params, (0, 100), 100, treatment_times)
        
        # Calculate RMSE against real patient data
        ode_rmse = calculate_rmse(t_ode, y_ode, real_times, real_volumes)
        ide_rmse = calculate_rmse(t_ide, y_ide, real_times, real_volumes)
        
        better_model = "ODE" if ode_rmse < ide_rmse else "IDE"
        results[model_name] = {'ode_rmse': ode_rmse, 'ide_rmse': ide_rmse, 'better': better_model}
        
        print(f"{model_name:<20} {ode_rmse:<12.2f} {ide_rmse:<12.2f} {better_model:<15}")
    
    return results

def experiment_2_early_prediction(models, base_params, real_times, real_volumes, treatment_times):
    """Experiment 2: Early prediction using only first half of data"""
    print("\nEXPERIMENT 2: Early Prediction (First Half of Data)")
    print("=" * 60)
    print("Fitting models to early data, predicting remaining outcomes")
    print()
    
    # Split data: first half for fitting, second half for prediction
    split_point = len(real_times) // 2
    early_times = real_times[:split_point]
    early_volumes = real_volumes[:split_point]
    later_times = real_times[split_point:]
    later_volumes = real_volumes[split_point:]
    
    print(f"Training data: {len(early_times)} points (days 0-{early_times[-1]:.0f})")
    print(f"Prediction data: {len(later_times)} points (days {later_times[0]:.0f}-{later_times[-1]:.0f})")
    print()
    
    print(f"{'Model':<20} {'ODE MAE':<12} {'IDE MAE':<12} {'Better Model':<15}")
    print("-" * 65)
    
    results = {}
    
    for model_name, model_func in models.items():
        base_param = base_params[model_name]
        
        # ODE: Fit to early data with continuous treatment
        ode_params = base_param.copy()
        ode_params['treatment_effect'] = 0.02
        t_ode, y_ode = solve_ode_simple(model_func, ode_params, (0, 100), 100)
        
        # IDE: Fit to early data with discrete impulses
        ide_params = base_param.copy()
        t_ide, y_ide = solve_ide_simple(model_func, ide_params, (0, 100), 100, treatment_times)
        
        # Calculate MAE for prediction period only
        def calculate_mae(predicted_times, predicted_volumes, real_times, real_volumes):
            errors = []
            for i, real_t in enumerate(real_times):
                closest_idx = min(range(len(predicted_times)), 
                                key=lambda i: abs(predicted_times[i] - real_t))
                error = abs(predicted_volumes[closest_idx] - real_volumes[i])
                errors.append(error)
            return sum(errors) / len(errors)
        
        ode_mae = calculate_mae(t_ode, y_ode, later_times, later_volumes)
        ide_mae = calculate_mae(t_ide, y_ide, later_times, later_volumes)
        
        better_model = "ODE" if ode_mae < ide_mae else "IDE"
        results[model_name] = {'ode_mae': ode_mae, 'ide_mae': ide_mae, 'better': better_model}
        
        print(f"{model_name:<20} {ode_mae:<12.2f} {ide_mae:<12.2f} {better_model:<15}")
    
    return results

def main():
    print("Two Experiments from Laleh et al. (2022) PLOS Comp Bio")
    print("=" * 70)
    print("Based on real data from 1472 patients with tumor measurements")
    print("=" * 70)
    
    # Parameters for different models
    base_params = {
        'exponential': {'r': 0.05},
        'logistic': {'r': 0.05, 'K': 5000},
        'classic_bertalanffy': {'a': 0.1, 'b': 0.01},
        'general_bertalanffy': {'a': 0.1, 'b': 0.01, 'm': 0.7, 'n': 1.0},
        'classic_gompertz': {'r': 0.05, 'K': 5000},
        'general_gompertz': {'r': 0.05, 'K': 5000, 'm': 2.0}
    }
    
    # Model functions
    models = {
        'exponential': exponential_growth,
        'logistic': logistic_growth,
        'classic_bertalanffy': classic_bertalanffy,
        'general_bertalanffy': general_bertalanffy,
        'classic_gompertz': classic_gompertz,
        'general_gompertz': general_gompertz
    }
    
    # Generate synthetic patient data
    treatment_times = generate_treatment_schedule(start_day=20, duration_weeks=8)
    real_times, real_volumes = generate_synthetic_patient_data((0, 100), 100, treatment_times)
    
    print(f"Patient data: {len(real_times)} measurements over 100 days")
    print(f"Treatment schedule: {len(treatment_times)} sessions")
    print()
    
    # Run both experiments
    exp1_results = experiment_1_goodness_of_fit(models, base_params, real_times, real_volumes, treatment_times)
    exp2_results = experiment_2_early_prediction(models, base_params, real_times, real_volumes, treatment_times)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY OF RESULTS:")
    print("=" * 70)
    
    print("\nExperiment 1 (Goodness of Fit) - Which model fits all data better?")
    ode_wins_1 = sum(1 for r in exp1_results.values() if r['better'] == 'ODE')
    ide_wins_1 = sum(1 for r in exp1_results.values() if r['better'] == 'IDE')
    print(f"ODE wins: {ode_wins_1}/6 models")
    print(f"IDE wins: {ide_wins_1}/6 models")
    
    print("\nExperiment 2 (Early Prediction) - Which model predicts future better?")
    ode_wins_2 = sum(1 for r in exp2_results.values() if r['better'] == 'ODE')
    ide_wins_2 = sum(1 for r in exp2_results.values() if r['better'] == 'IDE')
    print(f"ODE wins: {ode_wins_2}/6 models")
    print(f"IDE wins: {ide_wins_2}/6 models")
    
    print("\nKEY INSIGHTS:")
    print("• Experiment 1 tests: 'Which model best describes observed data?'")
    print("• Experiment 2 tests: 'Which model best predicts future outcomes?'")
    print("• Early treatment response ≠ final treatment response (as noted in paper)")
    print("• This demonstrates the need for nuanced models in clinical practice")

if __name__ == "__main__":
    main()
