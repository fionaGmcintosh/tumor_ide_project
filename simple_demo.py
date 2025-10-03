"""
Simple demonstration of ODE vs IDE concept without external dependencies.
This shows the core mathematical concepts.
"""

import math

def exponential_growth(t, y, r):
    """Exponential growth: dy/dt = r*y"""
    return r * y

def logistic_growth(t, y, r, K):
    """Logistic growth: dy/dt = r*y*(1 - y/K)"""
    return r * y * (1 - y / K)

def gompertz_growth(t, y, r, K):
    """Gompertz growth: dy/dt = r*y*ln(K/y)"""
    if y <= 0:
        return 0
    return r * y * math.log(K / y)

def solve_ode_simple(model_func, params, t_span, y0, dt=0.1):
    """
    Simple Euler method solver for demonstration.
    """
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

def apply_radiation_impulse(y, impulse_strength=0.1):
    """Apply radiation impulse - kills fraction of tumor"""
    return y * (1 - impulse_strength)

def solve_ide_simple(model_func, params, t_span, y0, treatment_times, impulse_strength=0.1, dt=0.1):
    """
    Simple IDE solver with radiation impulses.
    """
    t_start, t_end = t_span
    t = t_start
    y = y0
    
    times = [t]
    values = [y]
    
    treatment_idx = 0
    
    while t < t_end:
        # Check if we hit a treatment time
        if treatment_idx < len(treatment_times) and t >= treatment_times[treatment_idx]:
            y = apply_radiation_impulse(y, impulse_strength)
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

def main():
    """
    Demonstrate ODE vs IDE comparison.
    """
    print("Tumor Growth Model Comparison: ODE vs IDE")
    print("=" * 50)
    
    # Parameters
    params = {'r': 0.05, 'K': 5000}
    t_span = (0, 100)
    y0 = 100
    
    # Treatment schedule
    treatment_times = generate_treatment_schedule(start_day=20, duration_weeks=8)
    print(f"Treatment schedule: {len(treatment_times)} sessions starting day 20")
    
    # Test different models
    models = {
        'Exponential': lambda t, y, **p: exponential_growth(t, y, p['r']),
        'Logistic': lambda t, y, **p: logistic_growth(t, y, p['r'], p['K']),
        'Gompertz': lambda t, y, **p: gompertz_growth(t, y, p['r'], p['K'])
    }
    
    print("\nModel Comparison Results:")
    print("-" * 30)
    
    for model_name, model_func in models.items():
        print(f"\n{model_name} Model:")
        
        # Solve ODE (no treatment)
        t_ode, y_ode = solve_ode_simple(model_func, params, t_span, y0)
        final_volume_ode = y_ode[-1]
        
        # Solve IDE (with treatment)
        t_ide, y_ide = solve_ide_simple(model_func, params, t_span, y0, treatment_times)
        final_volume_ide = y_ide[-1]
        
        # Calculate treatment effect
        treatment_effect = ((final_volume_ode - final_volume_ide) / final_volume_ode) * 100
        
        print(f"  Final volume (ODE): {final_volume_ode:.1f} mm³")
        print(f"  Final volume (IDE): {final_volume_ide:.1f} mm³")
        print(f"  Treatment effect: {treatment_effect:.1f}% reduction")
        
        # Find minimum volume during treatment
        treatment_period = [i for i, t in enumerate(t_ide) if 20 <= t <= 76]
        if treatment_period:
            min_volume = min(y_ide[i] for i in treatment_period)
            print(f"  Minimum volume during treatment: {min_volume:.1f} mm³")
    
    print("\n" + "=" * 50)
    print("Key Insights:")
    print("1. IDE models show discrete drops at treatment times")
    print("2. Treatment effectiveness varies by growth model")
    print("3. Logistic and Gompertz models show saturation effects")
    print("4. IDE models more realistically represent clinical treatment")
    
    print("\nThis demonstrates why IDE models may be superior for")
    print("modeling cancer treatment with radiation therapy!")

if __name__ == "__main__":
    main()
