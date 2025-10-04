"""
Enhanced analysis of all 6 tumor growth models with realistic context
Shows ODE vs IDE vs hypothetical clinical targets
"""

import math

def exponential_growth(t, y, r):
    """Exponential growth: dy/dt = r*y"""
    return r * y

def logistic_growth(t, y, r, K):
    """Logistic growth: dy/dt = r*y*(1 - y/K)"""
    return r * y * (1 - y / K)

def classic_bertalanffy(t, y, a, b):
    """Classic Bertalanffy: dy/dt = a*y^(2/3) - b*y"""
    return a * (y ** (2/3)) - b * y

def general_bertalanffy(t, y, a, b, m, n):
    """General Bertalanffy: dy/dt = a*y^m - b*y^n"""
    return a * (y ** m) - b * (y ** n)

def classic_gompertz(t, y, r, K):
    """Classic Gompertz: dy/dt = r*y*ln(K/y)"""
    if y <= 0:
        return 0
    return r * y * math.log(K / y)

def general_gompertz(t, y, r, K, m):
    """General Gompertz: dy/dt = r*y*ln(K/y)^(1/m)"""
    if y <= 0 or y >= K:
        return 0
    try:
        return r * y * (math.log(K / y) ** (1/m))
    except (ValueError, ZeroDivisionError):
        return 0

def solve_ode_simple(model_func, params, t_span, y0, dt=0.1):
    """Simple Euler method solver"""
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
    """IDE solver with radiation impulses"""
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

def main():
    print("Enhanced Tumor Growth Analysis: ODE vs IDE vs Clinical Context")
    print("=" * 80)
    
    # Parameters for different models
    params_sets = {
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
    
    # Common parameters
    t_span = (0, 100)
    y0 = 100  # Initial tumor volume (mm³)
    treatment_times = generate_treatment_schedule(start_day=20, duration_weeks=8)
    
    # Hypothetical clinical targets (these would come from real patient data)
    clinical_targets = {
        'exponential': 200,    # Target: <200 mm³
        'logistic': 150,       # Target: <150 mm³
        'classic_bertalanffy': 20,   # Target: <20 mm³
        'general_bertalanffy': 25,   # Target: <25 mm³
        'classic_gompertz': 1000,    # Target: <1000 mm³ (harder to treat)
        'general_gompertz': 500      # Target: <500 mm³
    }
    
    print(f"Initial tumor volume: {y0} mm³")
    print(f"Treatment schedule: {len(treatment_times)} sessions (5 days/week for 8 weeks)")
    print(f"Treatment starts: Day 20, ends: Day 76")
    print()
    
    print(f"{'Model':<20} {'No Treatment':<12} {'With Treatment':<15} {'Clinical Target':<15} {'Success?':<8} {'Effect %':<10}")
    print("-" * 90)
    
    for model_name, model_func in models.items():
        params = params_sets[model_name]
        
        # Solve ODE (no treatment)
        t_ode, y_ode = solve_ode_simple(model_func, params, t_span, y0)
        
        # Solve IDE (with treatment)
        t_ide, y_ide = solve_ide_simple(model_func, params, t_span, y0, treatment_times)
        
        # Calculate treatment effect
        effect = ((y_ode[-1] - y_ide[-1]) / y_ode[-1]) * 100
        
        # Check if treatment meets clinical target
        target = clinical_targets[model_name]
        success = "✓ YES" if y_ide[-1] < target else "✗ NO"
        
        print(f"{model_name:<20} {y_ode[-1]:<12.1f} {y_ide[-1]:<15.1f} {target:<15.1f} {success:<8} {effect:<10.1f}")
    
    print("\n" + "=" * 80)
    print("EXPLANATION OF TERMS:")
    print("-" * 20)
    print("• No Treatment (ODE): Tumor volume after 100 days with NO radiation therapy")
    print("• With Treatment (IDE): Tumor volume after 100 days WITH radiation therapy")
    print("• Clinical Target: Hypothetical treatment goal (would come from real patient data)")
    print("• Success?: Whether the treatment achieved the clinical target")
    print("• Effect %: How much radiation therapy reduced the tumor size")
    print()
    print("KEY INSIGHTS:")
    print("• This is SYNTHETIC data - real clinical targets would come from patient studies")
    print("• IDE models show discrete drops at each radiation session")
    print("• Some models (Exponential, Logistic) are very sensitive to treatment")
    print("• Others (Gompertz) show saturation effects and are harder to treat")
    print("• The 'real' comparison would require actual patient data!")

if __name__ == "__main__":
    main()
