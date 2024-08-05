from mpi4py import MPI
import sys
from pathlib import Path
from functools import partial
import numpy as np
from scipy.optimize import differential_evolution, minimize
from pyswarm import pso
from spotpy.algorithms import sceua
import spotpy
from scipy.stats import uniform
from datetime import datetime


# Add path to the utility scripts folder
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from utils.control_file_utils import read_from_control, parse_time_period, get_config_path
from utils.calibration_utils import read_param_bounds 
from utils.parallel_calibration_utils import write_iteration_results, create_iteration_results_file
from utils.logging_utils import get_logger
from utils.model_run_utils import run_mizuroute, run_summa, prepare_model_run, write_failed_run_info
from utils.model_evaluate_utils import evaluate_model

# Initialize a global logger variable
global logger 
logger = None

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

global iteration_count
iteration_count = 0

# Control file handling
controlFolder = Path('../../0_control_files')
controlFile = 'control_active.txt'

# Read necessary paths and settings
if rank == 0:
    basin_parameters_file = get_config_path(controlFolder, controlFile, 'basin_parameters_file', 'settings/SUMMA/basinParamInfo.txt')
    local_parameters_file = get_config_path(controlFolder, controlFile, 'local_parameters_file', 'settings/SUMMA/localParamInfo.txt')
    optimization_results_file = get_config_path(controlFolder, controlFile, 'optimization_results_file', 'optimisation/calibration_results.txt')
    params_to_calibrate = read_from_control(controlFolder/controlFile, 'params_to_calibrate').split(',')
    basin_params_to_calibrate = read_from_control(controlFolder/controlFile, 'basin_params_to_calibrate').split(',')
    obs_file_path = read_from_control(controlFolder/controlFile, 'obs_file_path')
    sim_reach_ID = read_from_control(controlFolder/controlFile, 'sim_reach_ID')
else:
    local_parameters_file = None
    basin_parameters_file = None
    optimization_results_file = None
    params_to_calibrate = None
    basin_params_to_calibrate = None
    obs_file_path = None
    sim_reach_ID = None

# Broadcast variables to all processes
local_parameters_file = comm.bcast(local_parameters_file, root=0)
basin_parameters_file = comm.bcast(basin_parameters_file, root=0)
optimization_results_file = comm.bcast(optimization_results_file, root=0)
params_to_calibrate = comm.bcast(params_to_calibrate, root=0)
basin_params_to_calibrate = comm.bcast(basin_params_to_calibrate, root=0)
obs_file_path = comm.bcast(obs_file_path, root=0)
sim_reach_ID = comm.bcast(sim_reach_ID, root=0)

if rank == 0:
    local_bounds_dict = read_param_bounds(local_parameters_file, params_to_calibrate)
    basin_bounds_dict = read_param_bounds(basin_parameters_file, basin_params_to_calibrate)
    local_bounds = [local_bounds_dict[param] for param in params_to_calibrate]
    basin_bounds = [basin_bounds_dict[param] for param in basin_params_to_calibrate]
    all_bounds = local_bounds + basin_bounds
    all_params = params_to_calibrate + basin_params_to_calibrate
else:
    local_bounds_dict = None
    basin_bounds_dict = None
    local_bounds = None
    basin_bounds = None
    all_bounds = None
    all_params = None

local_bounds_dict = comm.bcast(local_bounds_dict, root=0)
basin_bounds_dict = comm.bcast(basin_bounds_dict, root=0)
local_bounds = comm.bcast(local_bounds, root=0)
basin_bounds = comm.bcast(basin_bounds, root=0)
all_bounds = comm.bcast(all_bounds, root=0)
all_params = comm.bcast(all_params, root=0)

#Get the names of the control and file manager files 
filemanager_name = read_from_control(controlFolder/controlFile, 'settings_summa_filemanager')
mizu_control_file = read_from_control(controlFolder/controlFile, 'settings_mizu_control_file')

# Setup logging
rootPath = read_from_control(controlFolder/controlFile, 'root_path')
domainName = read_from_control(controlFolder/controlFile, 'domain_name')
domain_path = rootPath + '/' + f'domain_{domainName}'
experiment_id = read_from_control(controlFolder/controlFile, 'experiment_id')

# Use this logger throughout your script
logger = get_logger('SUMMA_optimization', domain_path, experiment_id)

def parallel_objective_wrapper(params, optimization_metric = 'KGE'):
    calib_metrics, eval_metrics, _ = parallel_run_model_and_evaluate(params, calib_period, eval_period)
    if rank == 0:
        return objective_function(params, calib_metrics, eval_metrics, optimization_metric)
    else:
        return None

def parallel_run_model_and_evaluate(params, calib_period, eval_period):
    logger.info(f"Rank {rank}: Entering parallel_run_model_and_evaluate")
    n_local = len(params_to_calibrate)
    local_params = params[:n_local]
    basin_params = params[n_local:]

    # Each rank runs the model and evaluates
    result = run_model_and_evaluate(local_params, basin_params, calib_period, eval_period, rank)
    
    # Gather results from all ranks
    all_results = comm.gather((result, local_params, basin_params), root=0)

    # Broadcast the gathered results to all ranks
    all_results = comm.bcast(all_results, root=0)

    logger.info(f"Rank {rank}: all_results: {all_results}")

    # All ranks process the results
    valid_results = [(r, lp, bp) for r, lp, bp in all_results if r is not None and r[0] is not None and r[1] is not None]
    if valid_results:
        try:
            calib_metrics = [r[0] for r, _, _ in valid_results]
            eval_metrics = [r[1] for r, _, _ in valid_results]
            rank_params = [np.concatenate((lp, bp)) for _, lp, bp in valid_results]  # Concatenate local and basin params

            logger.info(f"Rank {rank}: calib_metrics: {calib_metrics}")
            logger.info(f"Rank {rank}: eval_metrics: {eval_metrics}")
            logger.info(f"Rank {rank}: rank_params: {rank_params}")

            # Ensure all calib_metrics and eval_metrics have the same keys
            all_calib_keys = set().union(*calib_metrics)
            all_eval_keys = set().union(*eval_metrics)

            logger.info(f"Rank {rank}: all_calib_keys: {all_calib_keys}")
            logger.info(f"Rank {rank}: all_eval_keys: {all_eval_keys}")

            # Fill in missing values with NaN
            for metrics in calib_metrics:
                for key in all_calib_keys:
                    if key not in metrics:
                        metrics[key] = np.nan
            for metrics in eval_metrics:
                for key in all_eval_keys:
                    if key not in metrics:
                        metrics[key] = np.nan

            logger.info(f"Rank {rank}: Processed results - Calib: {calib_metrics}, Eval: {eval_metrics}")
            return calib_metrics, eval_metrics, rank_params
        except Exception as e:
            logger.error(f"Rank {rank}: Error processing results: {str(e)}")
            logger.debug(f"Rank {rank}: Valid results: {valid_results}")
            return None, None, None
    else:
        logger.warning(f"Rank {rank}: No valid results from any rank")
        return None, None, None
    
def objective_function(params, calib_period, eval_period, metric, iteration_results_file=None, add_noise=False):
    logger.info(f"Rank {rank}: Entering objective_function")
    global all_bounds, all_params, iteration_count

    for param, value, (lower, upper) in zip(all_params, params, all_bounds):
        if value < lower or value > upper:
            logger.warning(f"Rank {rank}: Parameter {param} out of bounds: {value} (bounds: {lower} - {upper})")
            return np.inf

    logger.info(f"Rank {rank}: Calling parallel_run_model_and_evaluate")
    calib_metrics, eval_metrics, rank_params = parallel_run_model_and_evaluate(params, calib_period, eval_period)
    
    if calib_metrics is None or eval_metrics is None or rank_params is None:
        logger.warning(f"Rank {rank}: Model evaluation failed, returning infinity")
        return np.inf

    # Use the metrics from all ranks
    values = [metrics[metric] for metrics in calib_metrics if metric in metrics]
    values = [v for v in values if not np.isnan(v)]  # Remove NaN values
    
    if not values:
        logger.warning(f"Rank {rank}: Metric {metric} not found in calib_metrics or all values are NaN, returning infinity")
        return np.inf
    
    # Here, we're returning all the values instead of their mean
    if metric in ['KGE', 'KGEp', 'NSE']:
        values = [-v for v in values]  # Convert to minimization problem
    
    if add_noise:
        values = [v + np.random.normal(0, 1e-3) for v in values]

    # Write iteration results
    if rank == 0 and iteration_results_file:  # Only rank 0 writes results
        iteration_count += 1  # Increment iteration count
        write_iteration_results(iteration_results_file, iteration_count, all_params, rank_params, calib_metrics, eval_metrics)

    logger.info(f"Rank {rank}: Objective function values: {values}")
    return values  # Return a list of values, one for each rank

def run_model_and_evaluate(local_param_values, basin_param_values, calib_period, eval_period, local_rank, max_retries=3):
    logger.info(f"Entering run_model_and_evaluate for local_rank {local_rank}")
    
    for attempt in range(max_retries):
        try:
            rank_experiment_id = f"{experiment_id}_rank{local_rank}"
            rank_specific_path, mizuroute_rank_specific_path, summa_destination_settings_path, mizuroute_destination_settings_path = prepare_model_run(
                rootPath, domainName, experiment_id, rank_experiment_id, local_rank,
                local_param_values, basin_param_values, params_to_calibrate, basin_params_to_calibrate,
                local_bounds_dict, basin_bounds_dict, filemanager_name, mizu_control_file
            )
            
            # Run SUMMA
            summa_path = read_from_control(controlFolder/controlFile, 'install_path_summa')
            if summa_path == 'default':
                summa_path = str(Path(rootPath) / "installs" / "summa" / "bin")
            summa_exe = read_from_control(controlFolder/controlFile, 'exe_name_summa')
            filemanager_path = summa_destination_settings_path / filemanager_name
            summa_log_path = rank_specific_path / "SUMMA_logs"
            summa_log_name = f"summa_log_attempt{attempt}.txt"
            
            summa_return_code = run_summa(summa_path, summa_exe, filemanager_path, summa_log_path, summa_log_name, local_rank)
            
            if summa_return_code != 0:
                # Create a dictionary of all parameters
                all_param_values = dict(zip(params_to_calibrate, local_param_values))
                all_param_values.update(dict(zip(basin_params_to_calibrate, basin_param_values)))
                
                # Write failed run information
                write_failed_run_info(rootPath, domainName, experiment_id, local_rank, attempt, 
                                      all_param_values, summa_log_path, summa_log_name)
                
                raise RuntimeError(f"SUMMA run failed with return code: {summa_return_code}")
            
            logger.info(f"SUMMA run completed successfully for rank {local_rank}")

            # Run mizuRoute
            mizuroute_path = read_from_control(controlFolder/controlFile, 'install_path_mizuroute')
            if mizuroute_path == 'default':
                mizuroute_path = str(Path(rootPath) / "installs" / "mizuRoute" / "route" / "bin")
            mizuroute_exe = read_from_control(controlFolder/controlFile, 'exe_name_mizuroute')
            mizuroute_control_path = mizuroute_destination_settings_path / mizu_control_file
            mizuroute_log_path = mizuroute_rank_specific_path / "mizuroute_logs"
            mizuroute_log_name = f"mizuroute_log_attempt{attempt}.txt"
            
            mizuroute_return_code = run_mizuroute(mizuroute_path, mizuroute_exe, mizuroute_control_path, mizuroute_log_path, mizuroute_log_name, local_rank)
            
            if mizuroute_return_code != 0:
                raise RuntimeError(f"mizuRoute run failed with return code: {mizuroute_return_code}")
            
            logger.info(f"mizuRoute run completed successfully for rank {local_rank}")

            calib_metrics, eval_metrics = evaluate_model(mizuroute_rank_specific_path, calib_period, eval_period, sim_reach_ID, obs_file_path)

            logger.info(f"Completed model run and evaluation for local_rank {local_rank}")
            return calib_metrics, eval_metrics

        except Exception as e:
            logger.error(f"Error in model run for rank {local_rank}, attempt {attempt + 1}: {str(e)}")
            if attempt == max_retries - 1:
                logger.error(f"Max retries reached for rank {local_rank}. Returning None.")
                return None, None
            else:
                logger.info(f"Retrying with new parameter values for rank {local_rank}")
                # Generate new parameter values
                local_param_values = [np.random.uniform(low, high) for low, high in local_bounds]
                basin_param_values = [np.random.uniform(low, high) for low, high in basin_bounds]

    # This line should never be reached, but just in case
    return None, None

def run_parallel_differential_evolution(bounds, maxiter=50, popsize=15, iteration_results_file = None):
    if rank == 0:
        result = differential_evolution(
            parallel_objective_wrapper,
            bounds,
            maxiter=maxiter,
            popsize=popsize,
            workers=size,  # Use all available MPI processes
            updating='deferred',  # Important for parallelization
            disp=True,  # Display progress
            callback=lambda xk, convergence: write_iteration_results(
                iteration_results_file,
                len(xk),
                all_params,
                [xk],
                [parallel_run_model_and_evaluate(xk, calib_period, eval_period)[0]],
                [parallel_run_model_and_evaluate(xk, calib_period, eval_period)[1]]
            ) if rank == 0 else None
        )
        logger.info(f"DE completed. Best fitness: {result.fun}")
        return result.x, result.fun
    else:
        # Other ranks just wait for tasks
        while True:
            task = comm.recv(source=0, tag=MPI.ANY_TAG)
            if task is None:
                break
            parallel_objective_wrapper(task)
    
    return None, None

def run_parallel_particle_swarm(objective_func, bounds, maxiter=50, swarmsize=15):
    def pso_objective(x):
        value = objective_func(x)
        return float('inf') if value is None else value

    lb = np.array([b[0] for b in bounds])
    ub = np.array([b[1] for b in bounds])

    # Distribute particles among ranks
    local_swarmsize = swarmsize // size
    if rank < swarmsize % size:
        local_swarmsize += 1

    logger.info("Local swarm size = {local_swarmsize}")

    # Initialize local swarm
    local_swarm = np.random.uniform(lb, ub, (local_swarmsize, len(bounds)))
    local_velocity = np.zeros((local_swarmsize, len(bounds)))
    local_best_pos = local_swarm.copy()
    local_best_score = np.array([pso_objective(p) for p in local_swarm])

    global_best_pos = None
    global_best_score = np.inf

    # PSO parameters
    w = 0.729  # Inertia weight
    c1 = 1.49445  # Cognitive parameter
    c2 = 1.49445  # Social parameter

    for iteration in range(maxiter):
        # Update velocities and positions
        for i in range(local_swarmsize):
            r1, r2 = np.random.rand(2)
            local_velocity[i] = (w * local_velocity[i] +
                                 c1 * r1 * (local_best_pos[i] - local_swarm[i]) +
                                 c2 * r2 * (global_best_pos - local_swarm[i]))
            
            local_swarm[i] += local_velocity[i]
            
            # Enforce bounds
            np.clip(local_swarm[i], lb, ub, out=local_swarm[i])

        # Evaluate new positions
        for i in range(local_swarmsize):
            score = pso_objective(local_swarm[i])
            if score < local_best_score[i]:
                local_best_score[i] = score
                local_best_pos[i] = local_swarm[i].copy()

        # Gather all best scores and positions
        all_best_scores = comm.gather(local_best_score, root=0)
        all_best_pos = comm.gather(local_best_pos, root=0)

        if rank == 0:
            all_best_scores = np.concatenate(all_best_scores)
            all_best_pos = np.vstack(all_best_pos)
            best_idx = np.argmin(all_best_scores)
            if all_best_scores[best_idx] < global_best_score:
                global_best_score = all_best_scores[best_idx]
                global_best_pos = all_best_pos[best_idx].copy()
            logger.info(f"Iteration {iteration}: Best score = {global_best_score}")

        # Broadcast global best to all ranks
        global_best_score = comm.bcast(global_best_score, root=0)
        global_best_pos = comm.bcast(global_best_pos, root=0)

    if rank == 0:
        logger.info(f"PSO completed. Best score: {global_best_score}")
    
    return global_best_pos, global_best_score

def run_parallel_sce_ua(objective_func, params_to_calibrate, bounds, maxiter=50, ngs=15):
    def sce_objective(x):
        value = objective_func(x)
        return float('inf') if value is None else value

    # Distribute complexes among ranks
    local_ngs = ngs // size
    if rank < ngs % size:
        local_ngs += 1

    logger.info("Local number of complexes = {local_ngs}")

    # Initialize local complexes
    npg = 2 * len(params_to_calibrate) + 1  # number of points in each complex
    nps = ngs * npg  # total number of points in the sample
    local_nps = local_ngs * npg

    local_population = np.random.rand(local_nps, len(bounds))
    for i, (lower, upper) in enumerate(bounds):
        local_population[:, i] = lower + local_population[:, i] * (upper - lower)
    
    local_fitness = np.array([sce_objective(ind) for ind in local_population])

    for iteration in range(maxiter):
        # Perform complex evolution for each local complex
        for _ in range(local_ngs):
            # Select a complex
            complex_indices = np.random.choice(local_nps, npg, replace=False)
            complex_pop = local_population[complex_indices]
            complex_fitness = local_fitness[complex_indices]

            # Sort complex
            sort_indices = np.argsort(complex_fitness)
            complex_pop = complex_pop[sort_indices]
            complex_fitness = complex_fitness[sort_indices]

            # Generate new point
            worst = complex_pop[-1]
            centroid = np.mean(complex_pop[:-1], axis=0)
            new_point = centroid + np.random.rand() * (centroid - worst)

            # Ensure new point is within bounds
            for i, (lower, upper) in enumerate(bounds):
                new_point[i] = np.clip(new_point[i], lower, upper)

            # Evaluate new point
            new_fitness = sce_objective(new_point)

            # Replace worst point if new point is better
            if new_fitness < complex_fitness[-1]:
                complex_pop[-1] = new_point
                complex_fitness[-1] = new_fitness

            # Update local population
            local_population[complex_indices] = complex_pop
            local_fitness[complex_indices] = complex_fitness

        # Gather all fitness values and individuals
        all_fitness = comm.gather(local_fitness, root=0)
        all_population = comm.gather(local_population, root=0)

        if rank == 0:
            all_fitness = np.concatenate(all_fitness)
            all_population = np.vstack(all_population)
            best_idx = np.argmin(all_fitness)
            global_best_fitness = all_fitness[best_idx]
            global_best_individual = all_population[best_idx]
            logger.info(f"Iteration {iteration}: Best fitness = {global_best_fitness}")

        # Broadcast global best to all ranks
        global_best_fitness = comm.bcast(global_best_fitness, root=0)
        global_best_individual = comm.bcast(global_best_individual, root=0)

    if rank == 0:
        logger.info(f"SCE-UA completed. Best fitness: {global_best_fitness}")
    
    return global_best_individual, global_best_fitness

def run_parallel_basin_hopping(objective_func, bounds, niter=50):
    def bh_objective(x):
        value = objective_func(x)
        return float('inf') if value is None else value

    def random_step(x, bounds):
        new_x = x + np.random.uniform(-0.5, 0.5, size=len(x))
        for i, (lower, upper) in enumerate(bounds):
            new_x[i] = np.clip(new_x[i], lower, upper)
        return new_x

    def local_search(func, x0, bounds):
        result = minimize(func, x0, method='L-BFGS-B', bounds=bounds)
        return result.x, result.fun

    # Initialize local best solution
    x_local = np.array([np.random.uniform(b[0], b[1]) for b in bounds])
    f_local = bh_objective(x_local)

    T = 1.0  # Temperature for Metropolis criterion

    for iteration in range(niter // size + (1 if rank < niter % size else 0)):
        # Perform a random step
        x_new = random_step(x_local, bounds)
        
        # Perform local search
        x_minimized, f_minimized = local_search(bh_objective, x_new, bounds)

        # Metropolis acceptance criterion
        if f_minimized < f_local or np.random.random() < np.exp(-(f_minimized - f_local) / T):
            x_local, f_local = x_minimized, f_minimized
            logger.info("Iteration {iteration}: New best = {f_local}")

        T *= 0.95  # Cooling

    # Gather results from all ranks
    all_results = comm.gather((x_local, f_local), root=0)

    global_best = comm.bcast(global_best, root=0)
    return global_best[0], global_best[1]


def run_parallel_dds(objective_func, bounds, maxiter=100, r=0.2):
    logger.info("Entering run_parallel_dds")
    
    def dds_objective(x):
        value = objective_func(x)
        return float('inf') if value is None else value

    def perturb(x, r, bounds, P):
        x_new = x.copy()
        for i in range(len(x)):
            if P[i]:
                x_new[i] = x[i] + r * (bounds[i][1] - bounds[i][0]) * np.random.normal()
                x_new[i] = np.clip(x_new[i], bounds[i][0], bounds[i][1])
        return x_new
    
    # Initialize best solution
    x_best = np.array([np.random.uniform(b[0], b[1]) for b in bounds])
    f_best = dds_objective(x_best)

    local_iter = maxiter // size + (1 if rank < maxiter % size else 0)

    for iteration in range(local_iter):
        # Calculate probability of selecting each dimension
        P = np.random.random(len(bounds)) < (1 - np.log(iteration + 1) / np.log(local_iter))
        
        # Perturb the current best solution
        x_new = perturb(x_best, r, bounds, P)
        
        # Evaluate the new solution
        f_new = dds_objective(x_new)
        
        # Update the best solution if improvement is found
        if f_new < f_best:
            x_best, f_best = x_new, f_new
            logger.info(f"Iteration {iteration}: New best = {f_best}")

    # Gather results from all ranks
    all_results = comm.gather((x_best, f_best), root=0)

    if rank == 0:
        global_best = min(all_results, key=lambda x: x[1])
        logger.info(f"DDS completed. Best fitness: {global_best[1]}")
    else:
        global_best = None

    global_best = comm.bcast(global_best, root=0)
    return global_best[0], global_best[1]

def main():
    logger.info("Entering main function")
    global local_bounds, basin_bounds, all_bounds, calib_period, eval_period, iteration_count
    iteration_count = 0
    if rank == 0:
        # Read settings from control file
        algorithm = read_from_control(controlFolder/controlFile, 'Optimisation_algorithm')
        optimization_metric = read_from_control(controlFolder/controlFile, 'optimization_metric')
        num_iter = int(read_from_control(controlFolder/controlFile, 'num_iter'))   

        # Read time periods
        calibration_period_str = read_from_control(controlFolder/controlFile, 'calibration_period')
        evaluation_period_str = read_from_control(controlFolder/controlFile, 'evaluation_period')
        calib_period = parse_time_period(calibration_period_str)
        eval_period = parse_time_period(evaluation_period_str)
        iteration_results_file = create_iteration_results_file(experiment_id, rootPath, domainName, all_params)

        logger.info(f"Using {algorithm} as the optimization algorithm")
        logger.info(f"Using {optimization_metric} as the optimization metric")
        logger.info(f"Local parameters being calibrated: {params_to_calibrate}")
        logger.info(f"Basin parameters being calibrated: {basin_params_to_calibrate}")
        logger.info(f"Calibration period: {calib_period[0].date()} to {calib_period[1].date()}")
        logger.info(f"Evaluation period: {eval_period[0].date()} to {eval_period[1].date()}")

        # Read algorithm-specific parameters
        if algorithm == "DE":
            poplsize = int(read_from_control(controlFolder/controlFile, 'poplsize'))
        elif algorithm == "PSO":
            swrmsize = int(read_from_control(controlFolder/controlFile, 'swrmsize'))
        elif algorithm == "SCE-UA":
            ngsize = int(read_from_control(controlFolder/controlFile, 'ngsize'))
        elif algorithm == "DDS":
            dds_r = float(read_from_control(controlFolder/controlFile, 'dds_r'))
    else:
        algorithm = None
        optimization_metric = None
        num_iter = None
        iteration_results_file = None
        calib_period = None
        eval_period = None
        poplsize = None
        swrmsize = None
        ngsize = None
        dds_r = None

    # Broadcast variables to all processes
    algorithm = comm.bcast(algorithm, root=0)
    optimization_metric = comm.bcast(optimization_metric, root=0)
    num_iter = comm.bcast(num_iter, root=0)
    calib_period = comm.bcast(calib_period, root=0)
    eval_period = comm.bcast(eval_period, root=0)
    iteration_results_file = comm.bcast(iteration_results_file, root=0)

    # Broadcast algorithm-specific parameters
    if algorithm == "DE":
        poplsize = comm.bcast(poplsize if rank == 0 else None, root=0)
        logger.info(f"DE population size: {poplsize}")
    elif algorithm == "PSO":
        swrmsize = comm.bcast(swrmsize if rank == 0 else None, root=0)
        logger.info(f"PSO swarm size: {swrmsize}")
    elif algorithm == "SCE-UA":
        ngsize = comm.bcast(ngsize if rank == 0 else None, root=0)
        logger.info(f"SCE-UA ng size: {ngsize}")
    elif algorithm == "DDS":
        dds_r = comm.bcast(dds_r if rank == 0 else None, root=0)
        logger.info(f"DDS r: {dds_r}")

    logger.info(f"Algorithm: {algorithm}")
    logger.info(f"Optimization metric: {optimization_metric}")
    logger.info(f"Number of iterations: {num_iter}")
    logger.info(f"Calibration period: {calib_period}")
    logger.info(f"Evaluation period: {eval_period}")

    # Create a partial function for the objective function
    obj_func = partial(objective_function, calib_period=calib_period, eval_period=eval_period, 
                       metric=optimization_metric, iteration_results_file=iteration_results_file)

    try:
        if algorithm == "DE":
            best_params, best_value = run_parallel_differential_evolution(all_bounds, maxiter=num_iter, popsize=poplsize, iteration_results_file = iteration_results_file)
        elif algorithm == "PSO":
            best_params, best_value = run_parallel_particle_swarm(obj_func, all_bounds, maxiter=num_iter, swarmsize=swrmsize)
        elif algorithm == "SCE-UA":
            best_params, best_value = run_parallel_sce_ua(obj_func, all_params, all_bounds, maxiter=num_iter, ngs=ngsize)
        elif algorithm == "Basin-hopping":
            best_params, best_value = run_parallel_basin_hopping(obj_func, all_bounds, niter=num_iter)
        elif algorithm == "DDS":
            best_params, best_value = run_parallel_dds(obj_func, all_bounds, maxiter=num_iter, r=dds_r)
        else:
            raise ValueError("Invalid algorithm choice")

        if rank == 0:
            n_local = len(params_to_calibrate)
            best_local_params = best_params[:n_local]
            best_basin_params = best_params[n_local:]

            final_calib_metrics, final_eval_metrics = run_model_and_evaluate(best_local_params, best_basin_params, calib_period, eval_period, rank)

            logger.info(f"Optimization completed using {optimization_metric}")
            logger.info(f"Best {optimization_metric} value: {best_value if optimization_metric in ['RMSE', 'MAE'] else -best_value}")
            logger.info("Final performance metrics:")
            logger.info(f"Calibration: {final_calib_metrics}")
            logger.info(f"Evaluation: {final_eval_metrics}")

            with open(optimization_results_file, 'w') as f:
                f.write("Calibration Results:\n")
                f.write(f"Algorithm: {algorithm}\n")
                f.write("Best parameters:\n")
                for param, value in zip(all_params, best_params):
                    f.write(f"{param}: {value:.6e}\n")
                f.write(f"Final performance metrics:\n")
                f.write(f"Calibration: {final_calib_metrics}\n")
                f.write(f"Evaluation: {final_eval_metrics}\n")

            end_time = datetime.now()
            logger.info(f"Calibration ended on {end_time.strftime('%Y/%m/%d %H:%M:%S')}")
            logger.info(f"Total calibration time: {end_time - start_time}")

    except Exception as e:
        if rank == 0:
            logger.error(f"Error occurred during calibration: {str(e)}")
        raise

if __name__ == "__main__":
    start_time = datetime.now()
    logger.info(f"Rank {rank}: Script started")
    main()
    logger.info(f"Rank {rank}: Script completed")