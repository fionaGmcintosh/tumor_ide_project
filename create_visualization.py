"""
Create visualization showing experimental data with ODE and IDE model predictions
"""

import math
import random
import matplotlib.pyplot as plt
import numpy as np

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

def create_visualization():
    """Create comprehensive visualization of experimental data vs model predictions"""
    
    # Set random seed for reproducible results
    random.seed(42)
    np.random.seed(42)
    
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
    
    # Generate data
    t_span = (0, 100)
    y0 = 100
    treatment_times = generate_treatment_schedule(start_day=20, duration_weeks=8)
    real_times, real_volumes = generate_synthetic_patient_data(t_span, y0, treatment_times)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Tumor Growth Model Comparison: Experimental Data vs ODE vs IDE Predictions', 
                 fontsize=16, fontweight='bold')
    
    # Plot each model
    model_names = list(models.keys())
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    for i, (model_name, model_func) in enumerate(models.items()):
        row = i // 3
        col = i % 3
        ax = axes[row, col]
        
        # Get model parameters
        base_param = base_params[model_name]
        
        # ODE: Fit with continuous treatment effect
        ode_params = base_param.copy()
        ode_params['treatment_effect'] = 0.02
        t_ode, y_ode = solve_ode_simple(model_func, ode_params, t_span, y0)
        
        # IDE: Fit with discrete impulses
        ide_params = base_param.copy()
        t_ide, y_ide = solve_ide_simple(model_func, ide_params, t_span, y0, treatment_times)
        
        # Plot experimental data
        ax.scatter(real_times, real_volumes, alpha=0.6, s=20, color='black', 
                  label='Experimental Data', zorder=3)
        
        # Plot ODE prediction
        ax.plot(t_ode, y_ode, '--', color='red', linewidth=2, 
               label='ODE (Continuous Treatment)', alpha=0.8)
        
        # Plot IDE prediction
        ax.plot(t_ide, y_ide, '-', color='blue', linewidth=2, 
               label='IDE (Discrete Impulses)', alpha=0.8)
        
        # Mark treatment times
        for tx_time in treatment_times:
            ax.axvline(x=tx_time, color='gray', alpha=0.3, linestyle=':', linewidth=1)
        
        # Formatting
        ax.set_title(f'{model_name.replace("_", " ").title()}', fontweight='bold')
        ax.set_xlabel('Time (days)')
        ax.set_ylabel('Tumor Volume (mm³)')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        
        # Set y-axis to start from 0
        ax.set_ylim(bottom=0)
        
        # Add RMSE text
        def calculate_rmse(pred_times, pred_volumes, real_times, real_volumes):
            errors = []
            for i, real_t in enumerate(real_times):
                closest_idx = min(range(len(pred_times)), 
                                key=lambda i: abs(pred_times[i] - real_t))
                error = (pred_volumes[closest_idx] - real_volumes[i]) ** 2
                errors.append(error)
            return math.sqrt(sum(errors) / len(errors))
        
        ode_rmse = calculate_rmse(t_ode, y_ode, real_times, real_volumes)
        ide_rmse = calculate_rmse(t_ide, y_ide, real_times, real_volumes)
        
        ax.text(0.02, 0.98, f'ODE RMSE: {ode_rmse:.1f}\nIDE RMSE: {ide_rmse:.1f}', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                fontsize=8)
    
    plt.tight_layout()
    plt.savefig('tumor_model_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig('tumor_model_comparison.pdf', bbox_inches='tight')
    print("Visualization saved as 'tumor_model_comparison.png' and 'tumor_model_comparison.pdf'")
    
    # Create a summary plot showing all models together
    plt.figure(figsize=(15, 10))
    
    # Plot experimental data
    plt.scatter(real_times, real_volumes, alpha=0.6, s=30, color='black', 
               label='Experimental Data', zorder=3)
    
    # Plot all ODE models
    for i, (model_name, model_func) in enumerate(models.items()):
        base_param = base_params[model_name]
        ode_params = base_param.copy()
        ode_params['treatment_effect'] = 0.02
        t_ode, y_ode = solve_ode_simple(model_func, ode_params, t_span, y0)
        plt.plot(t_ode, y_ode, '--', color=colors[i], linewidth=2, 
                label=f'ODE {model_name.replace("_", " ").title()}', alpha=0.7)
    
    # Plot all IDE models
    for i, (model_name, model_func) in enumerate(models.items()):
        base_param = base_params[model_name]
        ide_params = base_param.copy()
        t_ide, y_ide = solve_ide_simple(model_func, ide_params, t_span, y0, treatment_times)
        plt.plot(t_ide, y_ide, '-', color=colors[i], linewidth=2, 
                label=f'IDE {model_name.replace("_", " ").title()}', alpha=0.7)
    
    # Mark treatment times
    for tx_time in treatment_times:
        plt.axvline(x=tx_time, color='gray', alpha=0.3, linestyle=':', linewidth=1)
    
    plt.title('All Models Comparison: Experimental Data vs ODE vs IDE Predictions', 
              fontsize=16, fontweight='bold')
    plt.xlabel('Time (days)', fontsize=12)
    plt.ylabel('Tumor Volume (mm³)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig('all_models_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig('all_models_comparison.pdf', bbox_inches='tight')
    print("Summary visualization saved as 'all_models_comparison.png' and 'all_models_comparison.pdf'")
    
    plt.show()

if __name__ == "__main__":
    create_visualization()
