import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import plotly.graph_objects as go
from pathlib import Path
import numpy as np
from pandas.plotting import parallel_coordinates

def read_param_bounds(param_file, params_to_calibrate):
    bounds = {}
    try:
        with open(param_file, 'r') as f:
            for line in f:
                parts = line.strip().split('|')
                if len(parts) >= 4 and parts[0].strip() in params_to_calibrate:
                    param = parts[0].strip()
                    lower = float(parts[2].strip().replace('d', 'e'))
                    upper = float(parts[3].strip().replace('d', 'e'))
                    bounds[param] = (lower, upper)
    except FileNotFoundError:
        print(f"Warning: Parameter file not found: {param_file}")
    except Exception as e:
        print(f"Error reading parameter file {param_file}: {str(e)}")
    return bounds

# Control file handling
controlFolder = Path('../0_control_files')
controlFile = 'control_active.txt'

def read_from_control(file, setting):
    """Extract a given setting from the control file."""
    with open(file) as contents:
        for line in contents:
            if setting in line and not line.startswith('#'):
                return line.split('|', 1)[1].split('#', 1)[0].strip()
    return None

def make_default_path(suffix):
    """Specify a default path based on the control file settings."""
    rootPath = Path(read_from_control(controlFolder/controlFile, 'root_path'))
    domainName = read_from_control(controlFolder/controlFile, 'domain_name')
    return rootPath / f'domain_{domainName}' / suffix

def plot_objective_surface(results, metric, output_file, n_levels=20):
    parameter_columns = [col for col in results.columns if col not in ['Iteration', 'RMSE', 'KGE', 'KGEp', 'NSE', 'MAE']]
    sensitivities = [(param, abs(results[param].corr(results[metric]))) for param in parameter_columns]
    top_two_params = sorted(sensitivities, key=lambda x: x[1], reverse=True)[:2]
    
    param1, param2 = top_two_params[0][0], top_two_params[1][0]
    
    plt.figure(figsize=(10, 8))
    contour = plt.tricontourf(results[param1], results[param2], results[metric], levels=n_levels, cmap='viridis')
    plt.colorbar(contour, label=metric)
    plt.xlabel(param1)
    plt.ylabel(param2)
    plt.title(f'Objective Function Surface for Top 2 Sensitive Parameters')
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def plot_parallel_coordinates(results, metric, output_file, n_best=10):
    parameter_columns = [col for col in results.columns if col not in ['Iteration', 'RMSE', 'KGE', 'KGEp', 'NSE', 'MAE']]
    best_results = results.nsmallest(n_best, metric)
    
    plt.figure(figsize=(12, 6))
    parallel_coordinates(best_results[parameter_columns + [metric]], metric, colormap=plt.get_cmap("viridis"))
    plt.title(f'Parallel Coordinates Plot of Top {n_best} Parameter Sets')
    plt.xlabel('Parameters')
    plt.ylabel('Normalized Parameter Values')
    plt.legend(title=metric, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def plot_parameter_sensitivity(results, metric, output_file):
    parameter_columns = [col for col in results.columns if col not in ['Iteration', 'RMSE', 'KGE', 'KGEp', 'NSE', 'MAE']]
    sensitivities = []
    
    for param in parameter_columns:
        param_range = results[param].max() - results[param].min()
        metric_range = results[metric].max() - results[metric].min()
        sensitivity = (metric_range / results[metric].mean()) / (param_range / results[param].mean())
        sensitivities.append(sensitivity)
    
    plt.figure(figsize=(10, 6))
    plt.bar(parameter_columns, sensitivities)
    plt.xlabel('Parameters')
    plt.ylabel(f'Sensitivity to {metric}')
    plt.title(f'Parameter Sensitivity Analysis based on {metric}')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def plot_parameter_correlation(results, metric, output_file):
    corr_columns = [col for col in results.columns if col not in ['Iteration']]
    corr_matrix = results[corr_columns].corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
    plt.title('Parameter and Metric Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def plot_objective_evolution(results, output_file):
    metrics = ['RMSE', 'KGE', 'KGEp', 'NSE', 'MAE']
    
    fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 4*len(metrics)), sharex=True)
    
    for ax, metric in zip(axes, metrics):
        if metric in ['RMSE', 'MAE']:
            # For these metrics, lower is better
            ax.plot(results['Iteration'], results[metric], 'b-', alpha=0.5, label='All iterations')
            ax.plot(results['Iteration'], results[metric].cummin(), 'r-', label='Best so far')
        else:
            # For KGE, KGEp, NSE, higher is better
            ax.plot(results['Iteration'], results[metric], 'b-', alpha=0.5, label='All iterations')
            ax.plot(results['Iteration'], results[metric].cummax(), 'r-', label='Best so far')
        
        ax.set_ylabel(metric)
        ax.grid(True)
        ax.legend()
        
        # Add text with the best value achieved
        if metric in ['RMSE', 'MAE']:
            best_value = results[metric].min()
        else:
            best_value = results[metric].max()
        ax.text(0.02, 0.95, f'Best: {best_value:.4f}', transform=ax.transAxes, 
                verticalalignment='top', fontweight='bold')
    
    axes[-1].set_xlabel('Iteration')
    plt.suptitle('Evolution of Objective Metrics Over Iterations')
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def plot_parameter_convergence(results, output_file, bounds):
    parameter_columns = [col for col in results.columns if col not in ['Iteration', 'RMSE', 'KGE', 'KGEp', 'NSE', 'MAE']]
    
    fig, axes = plt.subplots(len(parameter_columns), 1, figsize=(12, 4*len(parameter_columns)), sharex=True)
    for ax, param in zip(axes, parameter_columns):
        ax.plot(results['Iteration'], results[param], 'b.')
        ax.set_ylabel(param)
        ax.grid(True)
        
        # Add bounds as horizontal lines if available
        if param in bounds:
            lower, upper = bounds[param]
            ax.axhline(y=lower, color='r', linestyle='--', alpha=0.5)
            ax.axhline(y=upper, color='r', linestyle='--', alpha=0.5)
    
    axes[-1].set_xlabel('Iteration')
    plt.suptitle('Parameter Values Over Iterations')
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def plot_parallel_coordinates(results, metric, output_file):
    parameter_columns = [col for col in results.columns if col not in ['Iteration', 'RMSE', 'KGE', 'KGEp', 'NSE', 'MAE']]
    
    fig = go.Figure(data=
        go.Parcoords(
            line = dict(color = results[metric],
                        colorscale = 'Viridis',
                        showscale = True),
            dimensions = [dict(label = col, values = results[col]) for col in parameter_columns + [metric]]
        )
    )
    
    fig.update_layout(
        title='Parallel Coordinates Plot of Parameters and Objective Function',
        width=1200,
        height=600
    )
    
    fig.write_html(output_file)


def plot_pairwise_scatter(results, metric, output_file):
    parameter_columns = [col for col in results.columns if col not in ['Iteration', 'RMSE', 'KGE', 'KGEp', 'NSE', 'MAE']]
    plot_columns = parameter_columns + [metric]
    
    sns.pairplot(results[plot_columns], corner=True, diag_kind='kde', plot_kws={'alpha': 0.5})
    plt.suptitle('Pairwise Relationships Between Parameters and Objective Function')
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def plot_best_vs_worst(results, metric, output_file, bounds):
    parameter_columns = [col for col in results.columns if col not in ['Iteration', 'RMSE', 'KGE', 'KGEp', 'NSE', 'MAE']]
    
    best_solution = results.loc[results[metric].idxmin()]
    worst_solution = results.loc[results[metric].idxmax()]
    
    n_params = len(parameter_columns)
    n_cols = 3  # You can adjust this to change the number of columns in the plot
    n_rows = (n_params + n_cols - 1) // n_cols  # Calculate number of rows needed
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    axes = axes.flatten()  # Flatten the 2D array of axes for easier indexing
    
    for i, (param, ax) in enumerate(zip(parameter_columns, axes)):
        best_val = best_solution[param]
        worst_val = worst_solution[param]
        
        ax.bar([0, 1], [best_val, worst_val], color=['g', 'r'], alpha=0.7)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Best', 'Worst'])
        ax.set_title(param)
        ax.tick_params(axis='x', rotation=45)
        
        # Add bounds as horizontal lines if available
        if param in bounds:
            lower, upper = bounds[param]
            ax.axhline(y=lower, color='k', linestyle='--', alpha=0.5)
            ax.axhline(y=upper, color='k', linestyle='--', alpha=0.5)
        
        # Use scientific notation for y-axis if the values are very small or large
        ax.ticklabel_format(style='sci', axis='y', scilimits=(-2, 2))
    
    # Remove any unused subplots
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    fig.suptitle(f'Comparison of Best and Worst Solutions (based on {metric})', y=1.02)
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()

def main():
    local_parameters_file = read_from_control(controlFolder/controlFile, 'local_parameters_file')
    basin_parameters_file = read_from_control(controlFolder/controlFile, 'basin_parameters_file')

    # Read parameters to calibrate from control file
    params_to_calibrate = read_from_control(controlFolder/controlFile, 'params_to_calibrate').split(',')
    basin_params_to_calibrate = read_from_control(controlFolder/controlFile, 'basin_params_to_calibrate').split(',')

    # Read parameter files
    local_parameters_file = read_from_control(controlFolder/controlFile, 'local_parameters_file')
    if local_parameters_file == 'default':
        local_parameters_file = make_default_path('settings/SUMMA/localParamInfo.txt')
    else:
        local_parameters_file = Path(local_parameters_file)

    basin_parameters_file = read_from_control(controlFolder/controlFile, 'basin_parameters_file')
    if basin_parameters_file == 'default':
        basin_parameters_file = make_default_path('settings/SUMMA/basinParamInfo.txt')
    else:
        basin_parameters_file = Path(basin_parameters_file)

    # Read bounds
    local_bounds_dict = read_param_bounds(local_parameters_file, params_to_calibrate)
    basin_bounds_dict = read_param_bounds(basin_parameters_file, basin_params_to_calibrate)
    all_bounds = {**local_bounds_dict, **basin_bounds_dict}

    # Read the path to the iteration results file from the control file
    results_file = read_from_control(controlFolder/controlFile, 'iteration_results_file')
    if results_file is None or results_file == 'default':
        results_file = make_default_path('optimisation/iteration_results_sample.csv')
    else:
        results_file = Path(results_file)

    # Read the CSV file into a DataFrame
    results = pd.read_csv(results_file)

    # Define output folder
    output_folder = read_from_control(controlFolder/controlFile, 'diagnostics_output_folder')
    if output_folder is None or output_folder == 'default':
        output_folder = make_default_path('plots/calibration_diagnostics/')
    else:
        output_folder = Path(output_folder)
    
    output_folder.mkdir(parents=True, exist_ok=True)

    performance_metric = 'KGE'

    # Create diagnostic plots
    plot_objective_evolution(results, output_file=output_folder / 'objective_evolution.png')
    plot_parameter_convergence(results, output_file=output_folder / 'parameter_convergence.png', bounds=all_bounds)
    plot_parallel_coordinates(results, metric=performance_metric, output_file=output_folder / 'parallel_coordinates.html')
    plot_pairwise_scatter(results, metric=performance_metric, output_file=output_folder / 'pairwise_scatter.png')
    plot_best_vs_worst(results, metric=performance_metric, output_file=output_folder / 'best_vs_worst.png', bounds=all_bounds)
    plot_parameter_correlation(results, metric=performance_metric, output_file=output_folder / 'parameter_correlation.png')
    plot_objective_surface(results, metric=performance_metric, output_file=output_folder / 'objective_surface.png')

    print(f"Diagnostic plots have been saved to: {output_folder}")



if __name__ == "__main__":
    main()
