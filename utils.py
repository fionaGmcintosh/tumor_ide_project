import numpy as np
import matplotlib.pyplot as plt

# ######################## STILL MOSTLY AI CODE, WILL VET AND IMPLEMENT EASY VISUALIZER ################################


def create_visualization(t_array, y_array, t_array_pred, y_array_pred):

    fig, ax = plt.subplots()

    # Plot experimental data
    ax.scatter(t_array, y_array, alpha=0.6, s=20, color='black',
               label='Experimental Data', zorder=3)

    # Plot ODE prediction
    ax.plot(t_array_pred, y_array_pred, '--', color='red', linewidth=2,
            label='ODE (Continuous Treatment)', alpha=0.8)

    # Plot IDE prediction
    ax.plot(t_ide, y_ide, '-', color='blue', linewidth=2,
            label='IDE (Discrete Impulses)', alpha=0.8)

    # Mark treatment times
    for tx_time in t_array:
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