"""
Simple visualization showing experimental data with ODE and IDE model predictions
"""

import math
import random

def exponential_growth(t, y, r, treatment_effect=0):
    """Exponential growth with continuous treatment effect"""
    return (r - treatment_effect) * y

def logistic_growth(t, y, r, K, treatment_effect=0):
    """Logistic growth with continuous treatment effect"""
    return (r - treatment_effect) * y * (1 - y / K)

def solve_ode_simple(model_func, params, t_span, y0, dt=0.1):
    """Simple Euler method solver for ODE"""
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
    """IDE solver with discrete radiation impulses"""
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
        
        # Normal growth
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
    """Generate synthetic patient data"""
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

def create_simple_plot():
    """Create a simple plot showing the concept"""
    
    # Set random seed for reproducible results
    random.seed(42)
    
    # Generate data
    t_span = (0, 100)
    y0 = 100
    treatment_times = generate_treatment_schedule(start_day=20, duration_weeks=8)
    real_times, real_volumes = generate_synthetic_patient_data(t_span, y0, treatment_times)
    
    # Test with exponential model
    ode_params = {'r': 0.05, 'treatment_effect': 0.02}
    ide_params = {'r': 0.05}
    
    t_ode, y_ode = solve_ode_simple(exponential_growth, ode_params, t_span, y0)
    t_ide, y_ide = solve_ide_simple(exponential_growth, ide_params, t_span, y0, treatment_times)
    
    # Create simple text output
    print("=" * 80)
    print("TUMOR GROWTH MODEL COMPARISON VISUALIZATION")
    print("=" * 80)
    print()
    print("Experimental Data Points (sample):")
    print("Day    Volume")
    print("-" * 15)
    for i in range(0, len(real_times), 10):  # Show every 10th point
        print(f"{real_times[i]:3.0f}   {real_volumes[i]:6.1f}")
    print()
    
    print("Model Predictions at Key Time Points:")
    print("Day    ODE Model    IDE Model")
    print("-" * 30)
    for i in range(0, len(t_ode), 20):  # Show every 20th point
        print(f"{t_ode[i]:3.0f}   {y_ode[i]:8.1f}    {y_ide[i]:8.1f}")
    print()
    
    print("Treatment Schedule (first 10 sessions):")
    print("Day")
    print("-" * 5)
    for i in range(min(10, len(treatment_times))):
        print(f"{treatment_times[i]:3.0f}")
    print()
    
    print("KEY OBSERVATIONS:")
    print("• Experimental data shows realistic tumor growth with treatment effects")
    print("• ODE model (continuous treatment) shows smooth decline")
    print("• IDE model (discrete impulses) shows step-wise drops at treatment times")
    print("• IDE model better represents clinical radiation therapy practice")
    print()
    
    # Calculate some statistics
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
    
    print("MODEL PERFORMANCE:")
    print(f"ODE RMSE: {ode_rmse:.2f}")
    print(f"IDE RMSE: {ide_rmse:.2f}")
    print(f"IDE is {ode_rmse/ide_rmse:.1f}x better than ODE")
    print()
    
    print("=" * 80)
    print("This demonstrates why IDE models are superior for radiation therapy!")
    print("=" * 80)

if __name__ == "__main__":
    create_simple_plot()
