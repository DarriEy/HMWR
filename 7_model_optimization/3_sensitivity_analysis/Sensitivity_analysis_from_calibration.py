import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from pyviscous import viscous
from SALib.analyze import sobol
from tqdm import tqdm
from SALib.sample import sobol as sobol_sample
from scipy.stats import spearmanr
from SALib.analyze import delta, rbd_fast
import logging

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Control file handling
controlFolder = Path('../../0_control_files')
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

def read_calibration_results(file_path):
    """Read calibration results from the CSV file and remove rows with missing data."""
    df = pd.read_csv(file_path)
    return df.dropna()

def setup_logging():
    """Set up logging configuration."""
    log_path = make_default_path('optimisation/_workflow_log/sensitivity_analysis.log')
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def preprocess_data(samples, metric='RMSE'):
    # Remove duplicates
    samples_unique = samples.drop_duplicates(subset=[col for col in samples.columns if col != 'Iteration'])
    
    # Add small noise to break ties (use with caution)
    #samples_unique[metric] += np.random.normal(0, 1e-6, size=len(samples_unique))
    
    return samples_unique

def perform_delta_analysis(samples, metric='RMSE'):
    logger.info(f"Performing delta analysis using {metric} metric")
    parameter_columns = [col for col in samples.columns if col not in ['Iteration', 'RMSE', 'KGE', 'KGEp', 'NSE', 'MAE']]
    
    problem = {
        'num_vars': len(parameter_columns),
        'names': parameter_columns,
        'bounds': [[samples[col].min(), samples[col].max()] for col in parameter_columns]
    }
    
    X = samples[parameter_columns].values
    Y = samples[metric].values
    
    delta_results = delta.analyze(problem, X, Y)
    logger.info("Delta analysis completed")
    return pd.Series(delta_results['delta'], index=parameter_columns)

def perform_rbd_fast_analysis(samples, metric='RMSE'):
    logger.info(f"Performing rbd fast analysis using {metric} metric")
    parameter_columns = [col for col in samples.columns if col not in ['Iteration', 'RMSE', 'KGE', 'KGEp', 'NSE', 'MAE']]
    
    problem = {
        'num_vars': len(parameter_columns),
        'names': parameter_columns,
        'bounds': [[samples[col].min(), samples[col].max()] for col in parameter_columns]
    }
    
    X = samples[parameter_columns].values
    Y = samples[metric].values
    
    rbd_results = rbd_fast.analyze(problem, X, Y)
    logger.info("RBD fast analysis completed")
    return pd.Series(rbd_results['S1'], index=parameter_columns)

def perform_sensitivity_analysis(samples, metric='RMSE', min_samples=60):
    logger.info(f"Performing sensitivity analysis using {metric} metric")
    parameter_columns = [col for col in samples.columns if col not in ['Iteration', 'RMSE', 'KGE', 'KGEp', 'NSE', 'MAE']]
    
    if len(samples) < min_samples:
        logger.warning(f"Insufficient data for reliable sensitivity analysis. "
                       f"Have {len(samples)} samples, recommend at least {min_samples}.")
        return pd.Series([-999] * len(parameter_columns), index=parameter_columns)
    
    x = samples[parameter_columns].values
    y = samples[metric].values.reshape(-1, 1)
    
    sensitivities = []
    
    for i, param in tqdm(enumerate(parameter_columns), total=len(parameter_columns), desc="Calculating sensitivities"):
        try:
            # Try 'total' first, then 'single' if 'total' fails
            try:
                sensitivity_result = viscous(x, y, i, sensType='total')
            except ValueError:
                sensitivity_result = viscous(x, y, i, sensType='single')
            
            if isinstance(sensitivity_result, tuple):
                sensitivity = sensitivity_result[0]
            else:
                sensitivity = sensitivity_result
            
            sensitivities.append(sensitivity)
            print(f"Successfully calculated sensitivity for {param}")
        except Exception as e:
            print(f"Error in sensitivity analysis for parameter {param}: {str(e)}")
            sensitivities.append(-999)
    
    logger.info("Sensitivity analysis completed")
    return pd.Series(sensitivities, index=parameter_columns)

def perform_sobol_analysis(samples, metric='RMSE'):
    logger.info(f"Performing sobol analysis using {metric} metric")
    parameter_columns = [col for col in samples.columns if col not in ['Iteration', 'RMSE', 'KGE', 'KGEp', 'NSE', 'MAE']]
    
    problem = {
        'num_vars': len(parameter_columns),
        'names': parameter_columns,
        'bounds': [[samples[col].min(), samples[col].max()] for col in parameter_columns]
    }
    
    param_values = sobol_sample.sample(problem, 1024)
    
    Y = np.zeros(param_values.shape[0])
    for i in range(param_values.shape[0]):
        interpolated_values = []
        for j, col in enumerate(parameter_columns):
            interpolated_values.append(np.interp(param_values[i, j], 
                                                 samples[col].sort_values().values, 
                                                 samples[metric].values[samples[col].argsort()]))
        Y[i] = np.mean(interpolated_values)
    
    Si = sobol.analyze(problem, Y)

    logger.info("Sobol analysis completed")
    return pd.Series(Si['ST'], index=parameter_columns)

def plot_sensitivity(sensitivity, output_file):
    plt.figure(figsize=(10, 6))
    sensitivity.plot(kind='bar')
    plt.title("Parameter Sensitivity Analysis")
    plt.xlabel("Parameters")
    plt.ylabel("Sensitivity")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def perform_correlation_analysis(samples, metric='RMSE'):
    logger.info(f"Performing correlation analysis using {metric} metric")
    parameter_columns = [col for col in samples.columns if col not in ['Iteration', 'RMSE', 'KGE', 'KGEp', 'NSE', 'MAE']]
    correlations = []
    for param in parameter_columns:
        corr, _ = spearmanr(samples[param], samples[metric])
        correlations.append(abs(corr))  # Use absolute value for sensitivity
    logger.info("Correlation analysis completed") 
    return pd.Series(correlations, index=parameter_columns)

def plot_sensitivity_comparison(all_results, output_file):
    plt.figure(figsize=(12, 8))
    all_results.plot(kind='bar')
    plt.title("Sensitivity Analysis Comparison")
    plt.xlabel("Parameters")
    plt.ylabel("Sensitivity")
    plt.legend(title="Method", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

logger = setup_logging()

def main():
    logger.info("Starting sensitivity analysis script")

    # Read the path to the iteration results file from the control file
    results_file = read_from_control(controlFolder/controlFile, 'iteration_results_file')
    if results_file is None or results_file == 'default':
        results_file = make_default_path('optimisation/iteration_results_sample.csv')
    else:
        results_file = Path(results_file)
    logger.info(f"Using results file: {results_file}")

    # Define output folder
    output_folder = read_from_control(controlFolder/controlFile, 'sensitivity_output_folder')
    if output_folder is None or output_folder == 'default':
        output_folder = make_default_path('plots/sensitivity_analysis')
    else:
        output_folder = Path(output_folder)
    
    output_folder.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output folder set to: {output_folder}")

    # Read calibration results
    results = read_calibration_results(results_file)
    logger.info(f"Read {len(results)} calibration results")
    
    parameter_columns = [col for col in results.columns if col not in ['Iteration', 'RMSE', 'KGE', 'KGEp', 'NSE', 'MAE']]
    logger.info(f"Identified {len(parameter_columns)} parameter columns")
    
    if len(results) < 10:  # Arbitrary small number, adjust as needed
        logger.error("Error: Not enough data points for sensitivity analysis.")
        return
    
    # Preprocess the data
    results_preprocessed = preprocess_data(results, metric='RMSE')
    logger.info("Data preprocessing completed")

    logger.info("Starting sensitivity analysis...")
    
    # Perform multiple sensitivity analyses
    pyviscous_sensitivity = perform_sensitivity_analysis(results_preprocessed, metric='RMSE')
    logger.info("pyViscous sensitivity analysis completed")
    
    sobol_sensitivity = perform_sobol_analysis(results_preprocessed, metric='RMSE')
    logger.info("Sobol sensitivity analysis completed")
    
    delta_sensitivity = perform_delta_analysis(results_preprocessed, metric='RMSE')
    logger.info("Delta sensitivity analysis completed")
    
    rbd_fast_sensitivity = perform_rbd_fast_analysis(results_preprocessed, metric='RMSE')
    logger.info("RBD-FAST sensitivity analysis completed")
    
    correlation_sensitivity = perform_correlation_analysis(results_preprocessed, metric='RMSE')
    logger.info("Correlation sensitivity analysis completed")

    # Save and plot results for all methods
    methods = {
        'pyViscous': pyviscous_sensitivity,
        'Sobol': sobol_sensitivity,
        'Delta': delta_sensitivity,
        'RBD-FAST': rbd_fast_sensitivity,
        'Correlation': correlation_sensitivity
    }

    for name, sensitivity in methods.items():
        logger.info(f"\n{name} Sensitivity Analysis Results:")
        logger.info(sensitivity)
        sensitivity.to_csv(output_folder / f'{name.lower()}_sensitivity.csv')
        plot_sensitivity(sensitivity, output_folder / f'{name.lower()}_sensitivity.png')
        logger.info(f"Saved {name} sensitivity results and plot")

    # Compare results
    all_results = pd.DataFrame(methods)
    all_results.to_csv(output_folder / 'all_sensitivity_results.csv')
    plot_sensitivity_comparison(all_results, output_folder / 'sensitivity_comparison.png')
    logger.info("Saved comparison of all sensitivity results")

    logger.info("Sensitivity analysis script completed successfully")

if __name__ == "__main__":
    main()