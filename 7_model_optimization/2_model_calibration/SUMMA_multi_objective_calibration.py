# SUMMA_parameter_estimation_multiobjective.py

import os
import sys
import subprocess
import pandas as pd
import xarray as xr
from pathlib import Path
import logging
from datetime import datetime
import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.sms import SMSEMOA
from pymoo.decomposition.pbi import PBI
from pymoo.config import Config
Config.warnings['not_compiled'] = False

# Add path to the utility scripts folder
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from utils.control_file_utils import read_from_control, make_default_path, parse_time_period, get_config_path
from utils.calculate_sim_stats import get_KGE, get_KGEp, get_NSE, get_MAE, get_RMSE
from utils.calibration_utils import read_param_bounds, update_param_files, read_summa_error, write_iteration_results

# Control file handling
controlFolder = Path('../../0_control_files')
controlFile = 'control_active.txt'

# Read necessary paths and settings
model_run_folder = read_from_control(controlFolder/controlFile, 'model_run_folder')
if model_run_folder == 'default':
    model_run_folder = Path(read_from_control(controlFolder/controlFile, 'root_code_path')) / '6_model_runs'
else:
    model_run_folder = Path(model_run_folder)

summa_run_command = read_from_control(controlFolder/controlFile, 'summa_run_command')
mizuRoute_run_command = read_from_control(controlFolder/controlFile, 'mizuRoute_run_command')
summa_run_path = model_run_folder / summa_run_command
mizuRoute_run_path = model_run_folder / mizuRoute_run_command

#Find the necessary paths for the observations and simulation files
obs_file_path = read_from_control(controlFolder/controlFile, 'obs_file_path')
sim_file_path = read_from_control(controlFolder/controlFile, 'sim_file_path')
sim_reach_ID = read_from_control(controlFolder/controlFile, 'sim_reach_ID')
summa_sim_path = make_default_path(controlFolder, controlFile, f'simulations/{read_from_control(controlFolder/controlFile,"experiment_id")}/SUMMA/{read_from_control(controlFolder/controlFile,"experiment_id")}_timestep.nc')

basin_parameters_file = get_config_path(controlFolder, controlFile, 'basin_parameters_file', 'settings/SUMMA/basinParamInfo.txt')
local_parameters_file = get_config_path(controlFolder, controlFile, 'local_parameters_file', 'settings/SUMMA/localParamInfo.txt')
optimization_results_file = get_config_path(controlFolder, controlFile, 'optimization_results_file', 'optimisation/calibration_results.txt')
iteration_results_file = get_config_path(controlFolder, controlFile, 'iteration_results_file', 'optimisation/iteration_results.csv')

# Read parameters to calibrate from control file
params_to_calibrate = read_from_control(controlFolder/controlFile, 'params_to_calibrate').split(',')
basin_params_to_calibrate = read_from_control(controlFolder/controlFile, 'basin_params_to_calibrate').split(',')

# Read parameter bounds
local_bounds_dict = read_param_bounds(local_parameters_file, params_to_calibrate)
basin_bounds_dict = read_param_bounds(basin_parameters_file, basin_params_to_calibrate)

local_bounds = [local_bounds_dict[param] for param in params_to_calibrate]
basin_bounds = [basin_bounds_dict[param] for param in basin_params_to_calibrate]
all_bounds = local_bounds + basin_bounds
all_params = params_to_calibrate + basin_params_to_calibrate

def setup_logger(domain_name):
    # Create logger
    logger = logging.getLogger('SUMMA_multi_objective_calibration')
    logger.setLevel(logging.DEBUG)

    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Create file handler
    log_path = make_default_path(controlFolder, controlFile, f'optimisation/_workflow_log/{domain_name}_multi_objective_calibration.log')
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    # Create console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def run_model_and_evaluate(local_param_values, basin_param_values, calib_period, eval_period, max_retries=5):
    logger.debug(f"Model run: Using parameters - Local: {local_param_values}, Basin: {basin_param_values}")
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

def objective_function(params, calib_period, eval_period, metrics):
    global all_bounds, all_params
    logger = logging.getLogger('SUMMA_multi_objective_calibration')
    for param, value, (lower, upper) in zip(all_params, params, all_bounds):
        if value < lower or value > upper:
            logger.info(f"Parameter {param} out of bounds: {value} (bounds: {lower} - {upper})")
            return [np.inf] * len(metrics), {'calib': {m: np.inf for m in metrics}, 'eval': {m: np.inf for m in metrics}}

    n_local = len(params_to_calibrate)
    local_params = params[:n_local]
    basin_params = params[n_local:]

    calib_metrics, eval_metrics = run_model_and_evaluate(local_params, basin_params, calib_period, eval_period)
    logger.info(f"Calibration metrics: {calib_metrics}")
    logger.info(f"Evaluation metrics: {eval_metrics}")
    
    metrics_dict = {'calib': calib_metrics, 'eval': eval_metrics}
    
    objectives = [calib_metrics[metric] for metric in metrics]
    objectives = [calib_metrics[metric.strip()] for metric in metrics]
    
    # For metrics where higher is better, we negate the value for minimization
    objectives = [-obj if metric in ['KGE', 'KGEp', 'NSE'] else obj for obj, metric in zip(objectives, metrics)]
    
    return objectives, metrics_dict

def run_nsga2(objective_func, bounds, n_obj, pop_size, n_gen, metrics):
    iteration_counter = 0
    
    class MultiObjectiveProblem(Problem):
        def __init__(self, n_var, n_obj, xl, xu):
            super().__init__(n_var=n_var, n_obj=n_obj, xl=xl, xu=xu)

        def _evaluate(self, x, out, *args, **kwargs):
            nonlocal iteration_counter
            f = np.array([objective_func(xi)[0] for xi in x])
            out["F"] = f
            
            for i, xi in enumerate(x):
                params = dict(zip(all_params, xi))
                _, metrics_dict = objective_func(xi)
                write_iteration_results(iteration_results_file, 
                                        iteration_counter * pop_size + i, 
                                        params, 
                                        metrics_dict)
            iteration_counter += 1
    
    logger.info(f"Iteration {iteration_counter} completed")

    problem = MultiObjectiveProblem(
        n_var=len(bounds),
        n_obj=n_obj,
        xl=np.array([b[0] for b in bounds]),
        xu=np.array([b[1] for b in bounds])
    )

    algorithm = NSGA2(
        pop_size=pop_size,
        n_offsprings=pop_size,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PolynomialMutation(eta=20),
        eliminate_duplicates=True
    )

    res = minimize(problem,
                   algorithm,
                   ('n_gen', n_gen),
                   verbose=True)

    return res.X, res.F

def run_nsga3(objective_func, bounds, n_obj, pop_size, n_gen, metrics):
    iteration_counter = 0
    
    class MultiObjectiveProblem(Problem):
        def __init__(self, n_var, n_obj, xl, xu):
            super().__init__(n_var=n_var, n_obj=n_obj, xl=xl, xu=xu)

        def _evaluate(self, x, out, *args, **kwargs):
            nonlocal iteration_counter
            f = np.array([objective_func(xi)[0] for xi in x])
            out["F"] = f
            
            for i, xi in enumerate(x):
                params = dict(zip(all_params, xi))
                _, metrics_dict = objective_func(xi)
                write_iteration_results(iteration_results_file, 
                                        iteration_counter * pop_size + i, 
                                        params, 
                                        metrics_dict)
            iteration_counter += 1
    
    logger.info(f"Iteration {iteration_counter} completed")

    problem = MultiObjectiveProblem(
        n_var=len(bounds),
        n_obj=n_obj,
        xl=np.array([b[0] for b in bounds]),
        xu=np.array([b[1] for b in bounds])
    )

    ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=12)

    algorithm = NSGA3(
        pop_size=pop_size,
        n_offsprings=pop_size,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PolynomialMutation(eta=20),
        ref_dirs=ref_dirs,
        eliminate_duplicates=True
    )

    res = minimize(problem,
                   algorithm,
                   ('n_gen', n_gen),
                   verbose=True)

    return res.X, res.F

def run_borg_moea(objective_func, bounds, n_obj, pop_size, n_gen, metrics):
    # This is a simplified version of Borg MOEA
    # For a full implementation, you would need to use the actual Borg MOEA library
    
    iteration_counter = 0

    class BorgMOEA:
        def __init__(self, n_var, n_obj, xl, xu, pop_size):
            self.n_var = n_var
            self.n_obj = n_obj
            self.xl = xl
            self.xu = xu
            self.pop_size = pop_size if pop_size % 2 == 0 else pop_size + 1  # Ensure even population size
            self.population = np.random.uniform(xl, xu, (self.pop_size, n_var))
            self.archive = []
            self.fitness = np.array([objective_func(ind)[0] for ind in self.population])

        def evolve(self):
            offspring = self.population.copy()
            # Ensure population size is even
            if self.pop_size % 2 != 0:
                self.pop_size += 1
                self.population = np.vstack((self.population, np.random.uniform(self.xl, self.xu, (1, self.n_var))))
                offspring = np.vstack((offspring, np.random.uniform(self.xl, self.xu, (1, self.n_var))))
            
            # Perform SBX crossover
            for i in range(0, self.pop_size - 1, 2):  # Step by 2 to ensure we always have pairs
                if np.random.random() < 0.9:
                    beta = np.random.beta(2, 2)
                    offspring[i], offspring[i+1] = (
                        beta * offspring[i] + (1-beta) * offspring[i+1],
                        beta * offspring[i+1] + (1-beta) * offspring[i]
                    )
            
            # Perform polynomial mutation
            for i in range(self.pop_size):
                for j in range(self.n_var):
                    if np.random.random() < 1/self.n_var:
                        u = np.random.random()
                        if u <= 0.5:
                            delta = (2*u)**(1/21) - 1
                        else:
                            delta = 1 - (2*(1-u))**(1/21)
                        offspring[i,j] += delta * (self.xu[j] - self.xl[j])
            
            # Evaluate offspring
            offspring_fitness = np.array([objective_func(ind)[0] for ind in offspring])
            
            # Update population and archive
            combined = np.vstack((self.population, offspring))
            combined_fitness = np.vstack((self.fitness, offspring_fitness))
            
            # Non-dominated sorting
            fronts = self.non_dominated_sort(combined_fitness)
            
            new_pop = []
            for front in fronts:
                if len(new_pop) + len(front) <= self.pop_size:
                    new_pop.extend(front)
                else:
                    break
            
            self.population = combined[new_pop]
            self.fitness = combined_fitness[new_pop]
            
            # Update archive
            self.archive = self.population[fronts[0]]

        def non_dominated_sort(self, fitness):
            n_points = fitness.shape[0]
            n_dominated = np.zeros(n_points)
            dominated_by = [[] for _ in range(n_points)]
            fronts = [[]]

            for i in range(n_points):
                for j in range(i+1, n_points):
                    if np.all(fitness[i] <= fitness[j]) and np.any(fitness[i] < fitness[j]):
                        dominated_by[j].append(i)
                        n_dominated[i] += 1
                    elif np.all(fitness[j] <= fitness[i]) and np.any(fitness[j] < fitness[i]):
                        dominated_by[i].append(j)
                        n_dominated[j] += 1
                
                if n_dominated[i] == 0:
                    fronts[0].append(i)

            i = 0
            while fronts[i]:
                next_front = []
                for j in fronts[i]:
                    for k in dominated_by[j]:
                        n_dominated[k] -= 1
                        if n_dominated[k] == 0:
                            next_front.append(k)
                i += 1
                fronts.append(next_front)

            return fronts[:-1]

    borg = BorgMOEA(len(bounds), n_obj, [b[0] for b in bounds], [b[1] for b in bounds], pop_size)

    for gen in range(n_gen):
        borg.evolve()
        
        for i, xi in enumerate(borg.population):
            params = dict(zip(all_params, xi))
            _, metrics_dict = objective_func(xi)
            write_iteration_results(iteration_results_file, 
                                    iteration_counter * pop_size + i, 
                                    params, 
                                    metrics_dict)
        iteration_counter += 1
    logger.info(f"Iteration {iteration_counter} completed")

    return borg.archive, np.array([objective_func(ind)[0] for ind in borg.archive])

def run_moead(objective_func, bounds, n_obj, pop_size, n_gen, metrics):
    iteration_counter = 0

    class MultiObjectiveProblem(Problem):
        def __init__(self, n_var, n_obj, xl, xu):
            super().__init__(n_var=n_var, n_obj=n_obj, xl=xl, xu=xu)

        def _evaluate(self, x, out, *args, **kwargs):
            nonlocal iteration_counter
            f = np.array([objective_func(xi)[0] for xi in x])
            out["F"] = f
            
            for i, xi in enumerate(x):
                params = dict(zip(all_params, xi))
                _, metrics_dict = objective_func(xi)
                write_iteration_results(iteration_results_file, 
                                        iteration_counter * pop_size + i, 
                                        params, 
                                        metrics_dict)
            iteration_counter += 1

    problem = MultiObjectiveProblem(
        n_var=len(bounds),
        n_obj=n_obj,
        xl=np.array([b[0] for b in bounds]),
        xu=np.array([b[1] for b in bounds])
    )

    # Choose a decomposition method
    decomposition = PBI()

    algorithm = MOEAD(
        pop_size=pop_size,
        n_neighbors=15,
        decomposition=decomposition,
        prob_neighbor_mating=0.7,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PolynomialMutation(eta=20)
    )

    res = minimize(problem,
                   algorithm,
                   ('n_gen', n_gen),
                   verbose=True)

    return res.X, res.F

def run_smsemoa(objective_func, bounds, n_obj, pop_size, n_gen, metrics):
    iteration_counter = 0

    class MultiObjectiveProblem(Problem):
        def __init__(self, n_var, n_obj, xl, xu):
            super().__init__(n_var=n_var, n_obj=n_obj, xl=xl, xu=xu)

        def _evaluate(self, x, out, *args, **kwargs):
            nonlocal iteration_counter
            f = np.array([objective_func(xi)[0] for xi in x])
            out["F"] = f
            
            for i, xi in enumerate(x):
                params = dict(zip(all_params, xi))
                _, metrics_dict = objective_func(xi)
                write_iteration_results(iteration_results_file, 
                                        iteration_counter * pop_size + i, 
                                        params, 
                                        metrics_dict)
            iteration_counter += 1
    logger.info(f"Iteration {iteration_counter} completed")
    problem = MultiObjectiveProblem(
        n_var=len(bounds),
        n_obj=n_obj,
        xl=np.array([b[0] for b in bounds]),
        xu=np.array([b[1] for b in bounds])
    )

    algorithm = SMSEMOA(
        pop_size=pop_size,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PolynomialMutation(eta=20)
    )

    res = minimize(problem,
                   algorithm,
                   ('n_gen', n_gen),
                   verbose=True)

    return res.X, res.F


# Setup logger
domain_name = read_from_control(controlFolder/controlFile, 'domain_name')
logger = setup_logger(domain_name)

def main():
    # Read domain name from control file
    start_time = datetime.now()
    
    logger.info(f"Multi-objective calibration started for domain: {domain_name}")
    logger.info(f"Multi-objective calibration started on {start_time.strftime('%Y/%m/%d %H:%M:%S')}")

    # Read the pertinent calibration parameters 
    algorithm = read_from_control(controlFolder/controlFile, 'moo_optimisation_algorithm')
    optimization_metrics = read_from_control(controlFolder/controlFile, 'moo_optimization_metrics').split(',')
    num_iter = int(read_from_control(controlFolder/controlFile, 'moo_num_iter'))
    pop_size = int(read_from_control(controlFolder/controlFile, 'moo_pop_size'))
    
    logger.info(f"Using {algorithm} as the optimization algorithm")
    logger.info(f"Optimization metrics: {optimization_metrics}")
    logger.info(f"Local parameters being calibrated: {params_to_calibrate}")
    logger.info(f"Basin parameters being calibrated: {basin_params_to_calibrate}")

    # Read time periods
    calibration_period_str = read_from_control(controlFolder/controlFile, 'calibration_period')
    evaluation_period_str = read_from_control(controlFolder/controlFile, 'evaluation_period')
    calibration_period = parse_time_period(calibration_period_str)
    evaluation_period = parse_time_period(evaluation_period_str)

    logger.info(f"Calibration period: {calibration_period[0].date()} to {calibration_period[1].date()}")
    logger.info(f"Evaluation period: {evaluation_period[0].date()} to {evaluation_period[1].date()}")

    # Create a partial function with the selected metrics and date ranges
    from functools import partial
    objective_func = partial(objective_function, 
                             calib_period=calibration_period, 
                             eval_period=evaluation_period, 
                             metrics=optimization_metrics)

    try:
        if algorithm == "NSGA-II":
            best_params, best_values = run_nsga2(objective_func, all_bounds, n_obj=len(optimization_metrics), 
                                                 pop_size=pop_size, n_gen=num_iter, metrics=optimization_metrics)
        elif algorithm == "NSGA-III":
            best_params, best_values = run_nsga3(objective_func, all_bounds, n_obj=len(optimization_metrics), 
                                                 pop_size=pop_size, n_gen=num_iter, metrics=optimization_metrics)
        elif algorithm == "Borg-MOEA":
            best_params, best_values = run_borg_moea(objective_func, all_bounds, n_obj=len(optimization_metrics), 
                                                     pop_size=pop_size, n_gen=num_iter, metrics=optimization_metrics)
        elif algorithm == "MOEA/D":
            best_params, best_values = run_moead(objective_func, all_bounds, n_obj=len(optimization_metrics), 
                                                 pop_size=pop_size, n_gen=num_iter, metrics=optimization_metrics)
        elif algorithm == "SMS-EMOA":
            best_params, best_values = run_smsemoa(objective_func, all_bounds, n_obj=len(optimization_metrics), 
                                                   pop_size=pop_size, n_gen=num_iter, metrics=optimization_metrics)
        else:
            raise ValueError(f"Invalid algorithm choice: {algorithm}")

        logger.info(f"{algorithm} optimization completed")
        logger.info(f"Number of Pareto optimal solutions: {len(best_params)}")
        
        # Save all Pareto optimal solutions
        with open(optimization_results_file, 'w') as f:
            f.write(f"{algorithm} Optimization Results:\n")
            for i, (params, values) in enumerate(zip(best_params, best_values)):
                f.write(f"\nSolution {i+1}:\n")
                for param, value in zip(all_params, params):
                    f.write(f"{param}: {value:.6e}\n")
                f.write(f"Objective Values: {values}\n")
                
                # Evaluate final metrics for each solution
                n_local = len(params_to_calibrate)
                local_params = params[:n_local]
                basin_params = params[n_local:]
                final_calib_metrics, final_eval_metrics = run_model_and_evaluate(local_params, basin_params, calibration_period, evaluation_period)
                f.write(f"Final Calibration Metrics: {final_calib_metrics}\n")
                f.write(f"Final Evaluation Metrics: {final_eval_metrics}\n")

    except Exception as e:
        logger.exception(f"Error occurred during calibration: {str(e)}")
        raise

    end_time = datetime.now()
    logger.info(f"Calibration ended on {end_time.strftime('%Y/%m/%d %H:%M:%S')}")
    logger.info(f"Total calibration time: {end_time - start_time}")

if __name__ == "__main__":
    main()