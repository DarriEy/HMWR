'''
A script to calibrate SUMMA models. 

A list of all possible local parameters to calibrate (139):
upperBoundHead, lowerBoundHead, upperBoundTheta, lowerBoundTheta, upperBoundTemp, lowerBoundTemp, tempCritRain, tempRangeTimestep, frozenPrecipMultip, snowfrz_scale, fixedThermalCond_snow, albedoMax, albedoMinWinter, albedoMinSpring, albedoMaxVisible, albedoMinVisible, albedoMaxNearIR, albedoMinNearIR, albedoDecayRate, albedoSootLoad, albedoRefresh, radExt_snow, directScale, Frad_direct, Frad_vis, newSnowDenMin, newSnowDenMult, newSnowDenScal, constSnowDen, newSnowDenAdd, newSnowDenMultTemp, newSnowDenMultWind, newSnowDenMultAnd, newSnowDenBase, densScalGrowth, tempScalGrowth, grainGrowthRate, densScalOvrbdn, tempScalOvrbdn, baseViscosity, Fcapil, k_snow, mw_exp, z0Snow, z0Soil, z0Canopy, zpdFraction, critRichNumber, Louis79_bparam, Louis79_cStar, Mahrt87_eScale, leafExchangeCoeff, windReductionParam, Kc25, Ko25, Kc_qFac, Ko_qFac, kc_Ha, ko_Ha, vcmax25_canopyTop, vcmax_qFac, vcmax_Ha, vcmax_Hd, vcmax_Sv, vcmax_Kn, jmax25_scale, jmax_Ha, jmax_Hd, jmax_Sv, fractionJ, quantamYield, vpScaleFactor, cond2photo_slope, minStomatalConductance, winterSAI, summerLAI, rootScaleFactor1, rootScaleFactor2, rootingDepth, rootDistExp, plantWiltPsi, soilStressParam, critSoilWilting, critSoilTranspire, critAquiferTranspire, minStomatalResistance, leafDimension, heightCanopyTop, heightCanopyBottom, specificHeatVeg, maxMassVegetation, throughfallScaleSnow, throughfallScaleRain, refInterceptCapSnow, refInterceptCapRain, snowUnloadingCoeff, canopyDrainageCoeff, ratioDrip2Unloading, canopyWettingFactor, canopyWettingExp, soil_dens_intr, thCond_soil, frac_sand, frac_silt, frac_clay, fieldCapacity, wettingFrontSuction, theta_mp, theta_sat, theta_res, vGn_alpha, vGn_n, mpExp, k_soil, k_macropore, kAnisotropic, zScale_TOPMODEL, compactedDepth, aquiferBaseflowRate, aquiferScaleFactor, aquiferBaseflowExp, qSurfScale, specificYield, specificStorage, f_impede, soilIceScale, soilIceCV, minwind, minstep, maxstep, wimplicit, maxiter, relConvTol_liquid, absConvTol_liquid, relConvTol_matric, absConvTol_matric, relConvTol_energy, absConvTol_energy, relConvTol_aquifr, absConvTol_aquifr, zmin, zmax, minTempUnloading, minWindUnloading, rateTempUnloading, rateWindUnloading

A list of all snow parameters (38):
tempCritRain, tempRangeTimestep, frozenPrecipMultip, snowfrz_scale, fixedThermalCond_snow, albedoMax, albedoMinWinter, albedoMinSpring, albedoMaxVisible, albedoMinVisible, albedoMaxNearIR, albedoMinNearIR, albedoDecayRate, albedoSootLoad, albedoRefresh, radExt_snow, directScale, Frad_direct, Frad_vis, newSnowDenMin, newSnowDenMult, newSnowDenScal, constSnowDen, newSnowDenAdd, newSnowDenMultTemp, newSnowDenMultWind, newSnowDenMultAnd, newSnowDenBase, densScalGrowth, tempScalGrowth, grainGrowthRate, densScalOvrbdn, tempScalOvrbdn, baseViscosity, Fcapil, k_snow, mw_exp, z0Snow

A list of groundwater and soil parameters (24)
k_soil, k_macropore, kAnisotropic, aquiferBaseflowRate, aquiferScaleFactor, aquiferBaseflowExp, specificYield, specificStorage, soil_dens_intr, thCond_soil, frac_sand, frac_silt, frac_clay, fieldCapacity, wettingFrontSuction, theta_mp, theta_sat, theta_res, vGn_alpha, vGn_n, mpExp, zScale_TOPMODEL, compactedDepth, qSurfScale

A list of vegetation parameters (49):
winterSAI, summerLAI, rootScaleFactor1, rootScaleFactor2, rootingDepth, rootDistExp, plantWiltPsi, soilStressParam, critSoilWilting, critSoilTranspire, critAquiferTranspire, minStomatalResistance, leafDimension, heightCanopyTop, heightCanopyBottom, specificHeatVeg, maxMassVegetation, throughfallScaleSnow, throughfallScaleRain, refInterceptCapSnow, refInterceptCapRain, snowUnloadingCoeff, canopyDrainageCoeff, ratioDrip2Unloading, canopyWettingFactor, canopyWettingExp, Kc25, Ko25, Kc_qFac, Ko_qFac, kc_Ha, ko_Ha, vcmax25_canopyTop, vcmax_qFac, vcmax_Ha, vcmax_Hd, vcmax_Sv, vcmax_Kn, jmax25_scale, jmax_Ha, jmax_Hd, jmax_Sv, fractionJ, quantamYield, vpScaleFactor, cond2photo_slope, minStomatalConductance, leafExchangeCoeff, windReductionParam

A reasonable starting set of local parameters to begin with in a snow dominated catchment (20):
tempCritRain, frozenPrecipMultip, snowfrz_scale, albedoMax, albedoDecayRate, newSnowDenMin, newSnowDenMultTemp, Fcapil, k_snow, z0Snow, soil_dens_intr, thCond_soil, theta_sat, theta_res, k_soil, k_macropore, vGn_alpha, vGn_n, f_impede, aquiferBaseflowRate

A reasonable starting set of local parameters to begin with in a volcanic island catchment (20):
k_soil, k_macropore, theta_sat, theta_res, vGn_alpha, vGn_n, soil_dens_intr, thCond_soil, aquiferBaseflowRate, aquiferScaleFactor, aquiferBaseflowExp, qSurfScale, tempCritRain, summerLAI, rootingDepth, rootDistExp, critSoilWilting, critSoilTranspire, throughfallScaleRain, refInterceptCapRain

'''

import os
import sys
from datetime import datetime
import subprocess
import pandas as pd
import xarray as xr
from pathlib import Path
from functools import partial
import numpy as np
from scipy.optimize import differential_evolution, minimize
from pyswarm import pso
from spotpy.algorithms import sceua
import spotpy
from scipy.stats import uniform
import logging

# Add path to the utility scripts folder
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from utils.control_file_utils import read_from_control, make_default_path, parse_time_period, get_config_path
from utils.calculate_sim_stats import get_KGE, get_KGEp, get_NSE, get_MAE, get_RMSE
from utils.calibration_utils import read_param_bounds, update_param_files, read_summa_error, write_iteration_results

# Control file handling
controlFolder = Path('../../0_control_files')
controlFile = 'control_active.txt'

# Initialize a global logger variable
logger = None

def setup_logging(domain_path, experiment_id):
    global logger
    
    # Create the log directory if it doesn't exist
    log_dir = Path(domain_path) / 'optimisation' / '_workflow_log'
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create a unique log file name with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'{experiment_id}_{timestamp}.log'

    # Create a logger
    logger = logging.getLogger('SUMMA_optimization')
    logger.setLevel(logging.DEBUG)

    # Remove any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Prevent the logger from propagating messages to the root logger
    logger.propagate = False

    return logger

# Setup logging
rootPath = read_from_control(controlFolder/controlFile, 'root_path')
domainName = read_from_control(controlFolder/controlFile, 'domain_name')
domain_path = rootPath + '/' + f'domain_{domainName}'
experiment_id = read_from_control(controlFolder/controlFile, 'experiment_id')
logger = setup_logging(domain_path, experiment_id)

# Read necessary paths and settings
model_run_folder = get_config_path(controlFolder, controlFile, 'model_run_folder', '6_model_runs', is_folder=True)
summa_run_command = read_from_control(controlFolder/controlFile, 'summa_run_command')
mizuRoute_run_command = read_from_control(controlFolder/controlFile, 'mizuRoute_run_command')
summa_run_path = model_run_folder / summa_run_command
mizuRoute_run_path = model_run_folder / mizuRoute_run_command

#Find the necessary paths for the observations and simulation files
basin_parameters_file = get_config_path(controlFolder, controlFile, 'basin_parameters_file', 'settings/SUMMA/basinParamInfo.txt')
local_parameters_file = get_config_path(controlFolder, controlFile, 'local_parameters_file', 'settings/SUMMA/localParamInfo.txt')
optimization_results_file = get_config_path(controlFolder, controlFile, 'optimization_results_file', 'optimisation/calibration_results.txt')
iteration_results_file = get_config_path(controlFolder, controlFile, 'iteration_results_file', 'optimisation/iteration_results.csv')
obs_file_path = read_from_control(controlFolder/controlFile, 'obs_file_path')
sim_file_path = read_from_control(controlFolder/controlFile, 'sim_file_path')
sim_reach_ID = read_from_control(controlFolder/controlFile, 'sim_reach_ID')
summa_sim_path = make_default_path(controlFolder, controlFile, f'simulations/{read_from_control(controlFolder/controlFile,"experiment_id")}/SUMMA/{read_from_control(controlFolder/controlFile,"experiment_id")}_timestep.nc')

# Read parameters to calibrate from control file and their bounds
params_to_calibrate = read_from_control(controlFolder/controlFile, 'params_to_calibrate').split(',')
basin_params_to_calibrate = read_from_control(controlFolder/controlFile, 'basin_params_to_calibrate').split(',')
local_bounds_dict = read_param_bounds(local_parameters_file, params_to_calibrate)
basin_bounds_dict = read_param_bounds(basin_parameters_file, basin_params_to_calibrate)

# Create a list of bounds in the same order as the parameters
local_bounds = [local_bounds_dict[param] for param in params_to_calibrate]
basin_bounds = [basin_bounds_dict[param] for param in basin_params_to_calibrate]
all_bounds = local_bounds + basin_bounds
all_params = params_to_calibrate + basin_params_to_calibrate

# Calibration functions
def objective_function(params, calib_period, eval_period, metric, add_noise=False):
    global all_bounds, all_params

    for param, value, (lower, upper) in zip(all_params, params, all_bounds):
        if value < lower or value > upper:
            logger.info(f"Parameter {param} out of bounds: {value} (bounds: {lower} - {upper})")
            return np.inf, {'calib': {'RMSE': np.inf, 'KGE': -np.inf, 'KGEp': -np.inf, 'NSE': -np.inf, 'MAE': np.inf},
                            'eval': {'RMSE': np.inf, 'KGE': -np.inf, 'KGEp': -np.inf, 'NSE': -np.inf, 'MAE': np.inf}}

    n_local = len(params_to_calibrate)
    local_params = params[:n_local]
    basin_params = params[n_local:]

    calib_metrics, eval_metrics = run_model_and_evaluate(local_params, basin_params, calib_period, eval_period)
    
    logger.info(f"Calibration metrics: {calib_metrics}")
    logger.info(f"Evaluation metrics: {eval_metrics}")
    
    metrics = {'calib': calib_metrics, 'eval': eval_metrics}
    
    if add_noise:
        noise = np.random.normal(0, 1e-3)
    else:
        noise = 0
    
    value = calib_metrics[metric]
    if metric in ['KGE', 'KGEp', 'NSE']:
        value = -value  # Convert to minimization problem
    
    return value + noise, metrics

def run_model_and_evaluate(local_param_values, basin_param_values, calib_period, eval_period, max_retries=5):
    experiment_id = read_from_control(controlFolder/controlFile, "experiment_id")
    summa_log_path = make_default_path(controlFolder, controlFile, f'simulations/{experiment_id}/SUMMA/SUMMA_logs/summa_log.txt')

    for attempt in range(max_retries):
        try:
            update_param_files(local_param_values, basin_param_values, 
                               params_to_calibrate, basin_params_to_calibrate, 
                               local_parameters_file, basin_parameters_file,
                               local_bounds_dict, basin_bounds_dict)

            summa_result = subprocess.run(str(summa_run_path), shell=True, cwd=str(model_run_folder), 
                                          capture_output=True, text=True, check=True)
            
            if "STOP" in summa_result.stdout or "STOP" in summa_result.stderr:
                error_message = read_summa_error(summa_log_path)
                raise subprocess.CalledProcessError(1, str(summa_run_path), 
                                                    output=summa_result.stdout,
                                                    stderr=f"{summa_result.stderr}\n\nSUMMA Log Error:\n{error_message}")

            mizuroute_result = subprocess.run(str(mizuRoute_run_path), shell=True, cwd=str(model_run_folder), 
                                              capture_output=True, text=True, check=True)


            # Load observation and simulation data
            dfObs = pd.read_csv(obs_file_path, index_col='datetime', parse_dates=True)
            dfObs = dfObs['discharge_cms'].resample('h').mean()

            dfSim = xr.open_dataset(sim_file_path, engine='netcdf4')  
            segment_index = dfSim['reachID'].values == int(sim_reach_ID)
            dfSim = dfSim.sel(seg=segment_index) 
            dfSim = dfSim['IRFroutedRunoff'].to_dataframe().reset_index()
            dfSim.set_index('time', inplace=True)

            def calculate_metrics(obs, sim):
                return {
                    'RMSE': get_RMSE(obs, sim, transfo=1),
                    'KGE': get_KGE(obs, sim, transfo=1),
                    'KGEp': get_KGEp(obs, sim, transfo=1),
                    'NSE': get_NSE(obs, sim, transfo=1),
                    'MAE': get_MAE(obs, sim, transfo=1)
                }

            # Calculate metrics for calibration period
            calib_start, calib_end = calib_period
            calib_obs = dfObs.loc[calib_start:calib_end]
            calib_sim = dfSim.loc[calib_start:calib_end]
            calib_metrics = calculate_metrics(calib_obs.values, calib_sim['IRFroutedRunoff'].values)

            # Calculate metrics for evaluation period
            eval_start, eval_end = eval_period
            eval_obs = dfObs.loc[eval_start:eval_end]
            eval_sim = dfSim.loc[eval_start:eval_end]
            eval_metrics = calculate_metrics(eval_obs.values, eval_sim['IRFroutedRunoff'].values)

            os.remove(sim_file_path)
            os.remove(summa_sim_path)

            return calib_metrics, eval_metrics


        except subprocess.CalledProcessError as e:
            error_message = f"Attempt {attempt + 1} failed. SUMMA or mizuRoute run failed with error: {e}\n"
            error_message += f"STDOUT: {e.stdout}\n"
            error_message += f"STDERR: {e.stderr}\n"
            logger.error(error_message)
            
            if attempt < max_retries - 1:
                logger.info("Retrying with new parameters...")
                # Generate new random parameters within bounds
                local_param_values = [np.random.uniform(local_bounds_dict[param][0], local_bounds_dict[param][1]) 
                                        for param in params_to_calibrate]
                basin_param_values = [np.random.uniform(basin_bounds_dict[param][0], basin_bounds_dict[param][1]) 
                                        for param in basin_params_to_calibrate]
            else:
                logger.error("Max retries reached. Returning failure metrics.")
                return ({'RMSE': np.inf, 'KGE': -np.inf, 'KGEp': -np.inf, 'NSE': -np.inf, 'MAE': np.inf},
                        {'RMSE': np.inf, 'KGE': -np.inf, 'KGEp': -np.inf, 'NSE': -np.inf, 'MAE': np.inf})

        return ({'RMSE': np.inf, 'KGE': -np.inf, 'KGEp': -np.inf, 'NSE': -np.inf, 'MAE': np.inf},
            {'RMSE': np.inf, 'KGE': -np.inf, 'KGEp': -np.inf, 'NSE': -np.inf, 'MAE': np.inf})


def run_differential_evolution(objective_func, bounds, maxiter=50, popsize=15, metric='RMSE'):
    iteration_counter = 0
    
    def objective_wrapper(x):
        nonlocal iteration_counter
        value, metrics = objective_func(x)
        params = dict(zip(all_params, x))
        write_iteration_results(iteration_results_file, 
                                iteration_counter, 
                                params, 
                                metrics)
        iteration_counter += 1
        return value

    result = differential_evolution(
        func=objective_wrapper,
        bounds=bounds,
        maxiter=maxiter,
        popsize=popsize,
        disp=True
    )
    return result.x, result.fun if metric in ['RMSE', 'MAE'] else -result.fun

def run_particle_swarm(objective_func, bounds, maxiter=50, swarmsize=15, metric='RMSE'):
    lb = [b[0] for b in bounds]
    ub = [b[1] for b in bounds]
    iteration_counter = 0
    
    def pso_objective(x):
        nonlocal iteration_counter
        value, metrics = objective_func(x)
        params = dict(zip(all_params, x))
        write_iteration_results(iteration_results_file, 
                                iteration_counter, 
                                params, 
                                metrics)
        iteration_counter += 1
        return value

    xopt, fopt = pso(pso_objective, lb, ub, maxiter=maxiter, swarmsize=swarmsize)
    return xopt, fopt if metric in ['RMSE', 'MAE'] else -fopt

def run_sce_ua(objective_func, params_to_calibrate, bounds, maxiter=50, ngs=15, metric='RMSE'):
    class SpotpySetup(object):
        def __init__(self, params_to_calibrate, bounds):
            self.params = []
            for name, (low, high) in zip(params_to_calibrate, bounds):
                self.params.append(spotpy.parameter.Uniform(name, low=low, high=high))
            self.iteration = 0

        def parameters(self):
            return spotpy.parameter.generate(self.params)

        def simulation(self, vector):
            value, metrics = objective_func(vector)
            params = dict(zip(all_params, vector))
            write_iteration_results(iteration_results_file, 
                                    self.iteration, 
                                    params, 
                                    metrics)
            self.iteration += 1
            return [value]

        def evaluation(self):
            return [0]

        def objectivefunction(self, simulation, evaluation):
            return simulation[0]

    setup = SpotpySetup(params_to_calibrate, bounds)
    sampler = sceua(setup, dbname='SCEUA', dbformat='ram')
    sampler.sample(maxiter)
    results = sampler.getdata()
    best_index = np.argmin(results['like1'])
    return results['par'][best_index], results['like1'][best_index] if metric in ['RMSE', 'MAE'] else -results['like1'][best_index]

def run_basin_hopping(objective_func, bounds, niter=50, metric='RMSE'):
    iteration_counter = 0

    def random_step(x, bounds):
        new_x = x.copy()
        for i in range(len(x)):
            new_x[i] = np.random.uniform(bounds[i][0], bounds[i][1])
        return new_x

    def local_search(func, x0, bounds):
        result = minimize(func, x0, method='L-BFGS-B', bounds=bounds)
        return result.x, result.fun

    def objective_wrapper(x):
        nonlocal iteration_counter
        value, metrics = objective_func(x, add_noise=True)
        params = dict(zip(all_params, x))
        write_iteration_results(iteration_results_file, 
                                iteration_counter, 
                                params, 
                                metrics)
        iteration_counter += 1
        return value

    x_best = np.array([np.random.uniform(b[0], b[1]) for b in bounds])
    f_best = objective_wrapper(x_best)
    
    T = 1.0  # Temperature
    for i in range(niter):
        x_new = random_step(x_best, bounds)
        x_local, f_local = local_search(objective_wrapper, x_new, bounds)
        
        if f_local < f_best or np.random.random() < np.exp(-(f_local - f_best) / T):
            x_best, f_best = x_local, f_local
            logger.info(f"Iteration {i+1}: New best solution found")
        else:
            logger.info(f"Iteration {i+1}: No improvement")

        logger.info(f"Current best parameters: {dict(zip(all_params, x_best))}")
        logger.info(f"Current best objective value: {f_best}")
        logger.info("-" * 50)
        
        # Cool down temperature
        T *= 0.95

    return x_best, f_best if metric in ['RMSE', 'MAE'] else -f_best

def run_dds(objective_func, bounds, maxiter=100, r=0.2):
    def neighbor(x, r, bounds):
        P = uniform.rvs(size=len(x)) < r
        x_new = x.copy()
        for i in range(len(x)):
            if P[i]:
                x_new[i] = uniform.rvs(loc=bounds[i][0], scale=bounds[i][1]-bounds[i][0])
        return x_new

    x_best = np.array([uniform.rvs(loc=b[0], scale=b[1]-b[0]) for b in bounds])
    f_best, _ = objective_func(x_best)
    
    iteration_counter = 0
    for i in range(maxiter):
        r_current = 1 - np.log(i+1) / np.log(maxiter)
        x_new = neighbor(x_best, r_current, bounds)
        f_new, metrics = objective_func(x_new)
        
        if f_new < f_best:
            x_best = x_new
            f_best = f_new
        
        iteration_counter += 1
        params = dict(zip(all_params, x_new))
        write_iteration_results(iteration_results_file, 
                                iteration_counter, 
                                params, 
                                metrics)
        
    return x_best, f_best


def main():
    global local_bounds, basin_bounds, all_bounds
    logger.info(f"Calibration started on {datetime.now().strftime('%Y/%m/%d %H:%M:%S')}")

    # Read the pertinent calibration parameters 
    algorithm = read_from_control(controlFolder/controlFile, 'Optimisation_algorithm')
    optimization_metric = read_from_control(controlFolder/controlFile, 'optimization_metric')
    all_params = params_to_calibrate + basin_params_to_calibrate
    num_iter = int(read_from_control(controlFolder/controlFile, 'num_iter'))   
    
    logger.info(f"Using {algorithm} as the optimization algorithm")
    logger.info(f"Using {optimization_metric} as the optimization metric")
    logger.info(f"Local parameters being calibrated: {params_to_calibrate}")
    logger.info(f"Basin parameters being calibrated: {basin_params_to_calibrate}")

    # Read time periods
    calibration_period_str = read_from_control(controlFolder/controlFile, 'calibration_period')
    evaluation_period_str = read_from_control(controlFolder/controlFile, 'evaluation_period')
    calibration_period = parse_time_period(calibration_period_str)
    evaluation_period = parse_time_period(evaluation_period_str)

    logger.info(f"Calibration period: {calibration_period[0].date()} to {calibration_period[1].date()}")
    logger.info(f"Evaluation period: {evaluation_period[0].date()} to {evaluation_period[1].date()}")

    # Create a partial function with the selected metric and date ranges
    objective_func = partial(objective_function, 
                             calib_period=calibration_period, 
                             eval_period=evaluation_period, 
                             metric=optimization_metric)

    try:
        # Run the selected optimization algorithm
        if algorithm == "DE":
            poplsize = int(read_from_control(controlFolder/controlFile, 'poplsize'))
            best_params, best_value = run_differential_evolution(objective_func, all_bounds, maxiter=num_iter, popsize=poplsize, metric=optimization_metric)
        elif algorithm == "PSO":
            swrmsize = int(read_from_control(controlFolder/controlFile, 'swrmsize'))
            best_params, best_value = run_particle_swarm(objective_func, all_bounds, maxiter=num_iter, swarmsize=swrmsize, metric=optimization_metric)
        elif algorithm == "SCE-UA":
            ngsize = int(read_from_control(controlFolder/controlFile, 'ngsize'))
            best_params, best_value = run_sce_ua(objective_func, all_params, all_bounds, maxiter=num_iter, ngs=ngsize, metric=optimization_metric)
        elif algorithm == "Basin-hopping":
            best_params, best_value = run_basin_hopping(objective_func, all_bounds, niter=num_iter, metric=optimization_metric)
        elif algorithm == "DDS":
            dds_r = float(read_from_control(controlFolder/controlFile, 'dds_r'))
            best_params, best_value = run_dds(objective_func, all_bounds, maxiter=num_iter, r=dds_r)
        else:
            raise ValueError("Invalid algorithm choice")

        n_local = len(params_to_calibrate)
        best_local_params = best_params[:n_local]
        best_basin_params = best_params[n_local:]

        final_calib_metrics, final_eval_metrics = run_model_and_evaluate(best_local_params, best_basin_params, calibration_period, evaluation_period)

        logger.info(f"Optimization completed using {optimization_metric}")
        logger.info(f"Best {optimization_metric} value: {best_value if optimization_metric in ['RMSE', 'MAE'] else -best_value}")
        logger.info("Final performance metrics:")
        logger.info(f"Calibration: {final_calib_metrics}")
        logger.info(f"Evaluation: {final_eval_metrics}")

        # Save results for later analysis
        with open(optimization_results_file, 'w') as f:
            f.write("Calibration Results:\n")
            f.write(f"Algorithm: {algorithm}\n")
            f.write("Best parameters:\n")
            for param, value in zip(all_params, best_params):
                f.write(f"{param}: {value:.6e}\n")
            f.write(f"Final performance metrics:\n")
            f.write(f"Calibration: {final_calib_metrics}\n")
            f.write(f"Evaluation: {final_eval_metrics}\n")

    except Exception as e:
        logger.info(f"Error occurred during calibration: {str(e)}")
        raise

    end_time = datetime.now()
    logger.info(f"Calibration ended on {end_time.strftime('%Y/%m/%d %H:%M:%S')}")
    logger.info(f"Total calibration time: {end_time - start_time}")

if __name__ == "__main__":
    start_time = datetime.now()
    main()