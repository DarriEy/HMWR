import os
import sys
from pathlib import Path
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np

# Add the path to the scripts folder
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))
from calculate_sim_stats import get_KGE, get_KGEp, get_NSE, get_MAE, get_RMSE

# Control file handling
controlFolder = Path('../0_control_files')
controlFile = 'control_Goodpaster_Merit.txt'

def read_from_control(file, setting):
    """Extract a given setting from the control file."""
    with open(file) as contents:
        for line in contents:
            if setting in line and not line.startswith('#'):
                value = line.split('|', 1)[1].split('#', 1)[0].strip()
                if setting.endswith('_period'):
                    start, end = [datetime.strptime(date.strip(), '%Y-%m-%d') for date in value.split(',')]
                    return start, end
                return value
    return None

def make_default_path(suffix):
    """Specify a default path based on the control file settings."""
    rootPath = Path(read_from_control(controlFolder/controlFile, 'root_path'))
    domainName = read_from_control(controlFolder/controlFile, 'domain_name')
    return rootPath / f'domain_{domainName}' / suffix

# Read necessary paths and settings
obs_file_path = read_from_control(controlFolder/controlFile, 'obs_file_path')
sim_file_path = read_from_control(controlFolder/controlFile, 'sim_file_path')
sim_reach_ID = read_from_control(controlFolder/controlFile, 'sim_reach_ID')
calibration_period = read_from_control(controlFolder/controlFile, 'calibration_period')
evaluation_period = read_from_control(controlFolder/controlFile, 'evaluation_period')

def load_and_process_data():
    # Load observation data
    dfObs = pd.read_csv(obs_file_path, index_col='datetime', parse_dates=True)
    dfObs = dfObs['Discharge'].resample('h').mean()

    # Load simulation data
    dfSim = xr.open_dataset(sim_file_path, engine='netcdf4')
    segment_index = dfSim['reachID'].values == int(sim_reach_ID)
    dfSim = dfSim.sel(seg=segment_index)
    dfSim = dfSim['IRFroutedRunoff'].to_dataframe().reset_index()
    dfSim.set_index('time', inplace=True)

    return dfObs, dfSim

def print_data_info(dfObs, dfSim):
    print("Observation Data:")
    print(f"Shape: {dfObs.shape}")
    print(f"Date Range: {dfObs.index.min()} to {dfObs.index.max()}")
    print(f"Missing Values: {dfObs.isna().sum()}")
    print("\nSimulation Data:")
    print(f"Shape: {dfSim.shape}")
    print(f"Date Range: {dfSim.index.min()} to {dfSim.index.max()}")
    print(f"Missing Values: {dfSim['IRFroutedRunoff'].isna().sum()}")
    
    print("\nCalibration Period:")
    print(f"Start: {calibration_period[0]}, End: {calibration_period[1]}")
    print("\nEvaluation Period:")
    print(f"Start: {evaluation_period[0]}, End: {evaluation_period[1]}")

    # Check for data in calibration and evaluation periods
    calib_obs = dfObs.loc[calibration_period[0]:calibration_period[1]]
    calib_sim = dfSim.loc[calibration_period[0]:calibration_period[1]]
    eval_obs = dfObs.loc[evaluation_period[0]:evaluation_period[1]]
    eval_sim = dfSim.loc[evaluation_period[0]:evaluation_period[1]]

    print("\nData available in Calibration Period:")
    print(f"Observations: {calib_obs.shape[0]} entries")
    print(f"Simulations: {calib_sim.shape[0]} entries")

    print("\nData available in Evaluation Period:")
    print(f"Observations: {eval_obs.shape[0]} entries")
    print(f"Simulations: {eval_sim.shape[0]} entries")

def plot_data(dfObs, dfSim):
    plt.figure(figsize=(15, 10))
    plt.plot(dfObs.index, dfObs, label='Observed', alpha=0.7)
    plt.plot(dfSim.index, dfSim['IRFroutedRunoff'], label='Simulated', alpha=0.7)
    plt.axvline(calibration_period[0], color='r', linestyle='--', label='Calibration Start')
    plt.axvline(calibration_period[1], color='r', linestyle='--', label='Calibration End')
    plt.axvline(evaluation_period[0], color='g', linestyle='--', label='Evaluation Start')
    plt.axvline(evaluation_period[1], color='g', linestyle='--', label='Evaluation End')
    plt.xlabel('Date')
    plt.ylabel('Discharge (cms)')
    plt.title('Observed vs Simulated Discharge')
    plt.legend()
    plt.tight_layout()
    plt.savefig('discharge_comparison.png')
    plt.close()

def calculate_metrics(obs, sim):
    if len(obs) == 0 or len(sim) == 0:
        print("Warning: Empty observation or simulation data")
        return {metric: np.nan for metric in ['RMSE', 'KGE', 'KGEp', 'NSE', 'MAE']}
    return {
        'RMSE': get_RMSE(obs, sim, transfo=1),
        'KGE': get_KGE(obs, sim, transfo=1),
        'KGEp': get_KGEp(obs, sim, transfo=1),
        'NSE': get_NSE(obs, sim, transfo=1),
        'MAE': get_MAE(obs, sim, transfo=1)
    }

def print_metrics(dfObs, dfSim):
    calib_obs = dfObs.loc[calibration_period[0]:calibration_period[1]]
    calib_sim = dfSim.loc[calibration_period[0]:calibration_period[1]]
    eval_obs = dfObs.loc[evaluation_period[0]:evaluation_period[1]]
    eval_sim = dfSim.loc[evaluation_period[0]:evaluation_period[1]]

    print("\nCalibration Period Data:")
    print(f"Observations: {calib_obs.shape[0]} entries")
    print(f"Simulations: {calib_sim.shape[0]} entries")

    print("\nEvaluation Period Data:")
    print(f"Observations: {eval_obs.shape[0]} entries")
    print(f"Simulations: {eval_sim.shape[0]} entries")

    if calib_obs.shape[0] > 0 and calib_sim.shape[0] > 0:
        calib_metrics = calculate_metrics(calib_obs.values, calib_sim['IRFroutedRunoff'].values)
        print("\nCalibration Metrics:")
        for metric, value in calib_metrics.items():
            print(f"{metric}: {value:.4f}")
    else:
        print("\nWarning: Not enough data to calculate calibration metrics")

    if eval_obs.shape[0] > 0 and eval_sim.shape[0] > 0:
        eval_metrics = calculate_metrics(eval_obs.values, eval_sim['IRFroutedRunoff'].values)
        print("\nEvaluation Metrics:")
        for metric, value in eval_metrics.items():
            print(f"{metric}: {value:.4f}")
    else:
        print("\nWarning: Not enough data to calculate evaluation metrics")

def main():
    dfObs, dfSim = load_and_process_data()
    print_data_info(dfObs, dfSim)
    plot_data(dfObs, dfSim)
    print_metrics(dfObs, dfSim)

if __name__ == "__main__":
    main()