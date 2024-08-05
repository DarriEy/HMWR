import os
import sys
import subprocess
import pandas as pd
import xarray as xr
from pathlib import Path
import numpy as np
from functools import partial
from mpi4py import MPI
import logging
import csv
from datetime import datetime

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.sms import SMSEMOA
from pymoo.core.problem import Problem
from pymoo.core.algorithm import Algorithm
from pymoo.core.population import Population
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.decomposition.pbi import PBI
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation

# Add path to the utility scripts folder
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from utils.control_file_utils import read_from_control, make_default_path, parse_time_period, get_config_path
from utils.calculate_sim_stats import get_KGE, get_KGEp, get_NSE, get_MAE, get_RMSE
from utils.calibration_utils import read_param_bounds, update_param_files, read_summa_error, write_iteration_results

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Control file handling
controlFolder = Path('../../0_control_files')
controlFile = 'control_active.txt'

# Setup logging
def setup_logger(domain_name):
    logger = logging.getLogger('SUMMA_multi_objective_calibration')
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    if rank == 0:
        log_path = make_default_path(controlFolder, controlFile, f'optimisation/_workflow_log/{domain_name}_multi_objective_calibration.log')
        fh = logging.FileHandler(log_path)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    return logger

# Read domain name and setup logger
domain_name = read_from_control(controlFolder/controlFile, 'domain_name')
logger = setup_logger(domain_name)

if rank == 0:
    logger.info(f"Multi-objective calibration started for domain: {domain_name}")
    logger.info(f"Running with {size} MPI processes")

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

basin_parameters_file = get_config_path(controlFolder, controlFile, 'basin_parameters_file', 'settings/SUMMA/basinParamInfo.txt')
local_parameters_file = get_config_path(controlFolder, controlFile, 'local_parameters_file', 'settings/SUMMA/localParamInfo.txt')
optimization_results_file = get_config_path(controlFolder, controlFile, 'optimization_results_file', 'optimisation/calibration_results.txt')
iteration_results_file = get_config_path(controlFolder, controlFile, 'iteration_results_file', 'optimisation/iteration_results.csv')

# Read parameters to calibrate
params_to_calibrate = read_from_control(controlFolder/controlFile, 'params_to_calibrate').split(',')
basin_params_to_calibrate = read_from_control(controlFolder/controlFile, 'basin_params_to_calibrate').split(',')

local_bounds_dict = read_param_bounds(local_parameters_file, params_to_calibrate)
basin_bounds_dict = read_param_bounds(basin_parameters_file, basin_params_to_calibrate)

local_bounds = [local_bounds_dict[param] for param in params_to_calibrate]
basin_bounds = [basin_bounds_dict[param] for param in basin_params_to_calibrate]
all_bounds = local_bounds + basin_bounds
all_params = params_to_calibrate + basin_params_to_calibrate

obs_file_path = read_from_control(controlFolder/controlFile, 'obs_file_path')
sim_file_path = read_from_control(controlFolder/controlFile, 'sim_file_path')
sim_reach_ID = read_from_control(controlFolder/controlFile, 'sim_reach_ID')
summa_sim_path = make_default_path(controlFolder, controlFile, f'simulations/{read_from_control(controlFolder/controlFile,"experiment_id")}/SUMMA/{read_from_control(controlFolder/controlFile,"experiment_id")}_timestep.nc')


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

# Modified optimization algorithms with MPI
def run_mpi_algorithm(algorithm_class, objective_func, bounds, n_obj, pop_size, n_gen, metrics, **kwargs):
    iteration_counter = 0

    class DistributedMultiObjectiveProblem(Problem):
        def __init__(self, n_var, n_obj, xl, xu):
            super().__init__(n_var=n_var, n_obj=n_obj, xl=xl, xu=xu)

        def _evaluate(self, x, out, *args, **kwargs):
            nonlocal iteration_counter
            
            local_pop_size = len(x) // size
            start = rank * local_pop_size
            end = start + local_pop_size if rank < size - 1 else len(x)
            local_x = x[start:end]

            local_f = np.array([objective_func(xi)[0] for xi in local_x])
            
            all_f = comm.allgather(local_f)
            f = np.concatenate(all_f)

            out["F"] = f
            
            if rank == 0:
                for i, xi in enumerate(x):
                    params = dict(zip(all_params, xi))
                    _, metrics_dict = objective_func(xi)
                    write_iteration_results(iteration_results_file, 
                                            iteration_counter * pop_size + i, 
                                            params, 
                                            metrics_dict)
            iteration_counter += 1

    problem = DistributedMultiObjectiveProblem(
        n_var=len(bounds),
        n_obj=n_obj,
        xl=np.array([b[0] for b in bounds]),
        xu=np.array([b[1] for b in bounds])
    )

    algorithm = algorithm_class(
        pop_size=pop_size,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PolynomialMutation(eta=20),
        **kwargs
    )

    res = minimize(problem,
                   algorithm,
                   ('n_gen', n_gen),
                   verbose=rank==0)

    return res.X, res.F

def run_nsga2(objective_func, bounds, n_obj, pop_size, n_gen, metrics):
    return run_mpi_algorithm(NSGA2, objective_func, bounds, n_obj, pop_size, n_gen, metrics,
                             eliminate_duplicates=True)

def run_nsga3(objective_func, bounds, n_obj, pop_size, n_gen, metrics):
    ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=12)
    return run_mpi_algorithm(NSGA3, objective_func, bounds, n_obj, pop_size, n_gen, metrics,
                             ref_dirs=ref_dirs, eliminate_duplicates=True)

def run_moead(objective_func, bounds, n_obj, pop_size, n_gen, metrics):
    decomposition = PBI()
    return run_mpi_algorithm(MOEAD, objective_func, bounds, n_obj, pop_size, n_gen, metrics,
                             n_neighbors=15, decomposition=decomposition, prob_neighbor_mating=0.7)

def run_smsemoa(objective_func, bounds, n_obj, pop_size, n_gen, metrics):
    return run_mpi_algorithm(SMSEMOA, objective_func, bounds, n_obj, pop_size, n_gen, metrics)

class BorgMOEA(Algorithm):

    def __init__(self, 
                 pop_size=100,
                 eta=0.01,
                 pm=1.0,
                 sbx_prob=0.8,
                 sbx_eta=15,
                 de_prob=0.8,
                 de_f=0.5,
                 pcx_prob=0.8,
                 pcx_eta=0.1,
                 spx_prob=0.8,
                 spx_epsilon=1.0,
                 **kwargs):

        super().__init__(**kwargs)
        self.pop_size = pop_size
        self.eta = eta
        self.pm = pm
        self.sbx_prob = sbx_prob
        self.sbx_eta = sbx_eta
        self.de_prob = de_prob
        self.de_f = de_f
        self.pcx_prob = pcx_prob
        self.pcx_eta = pcx_eta
        self.spx_prob = spx_prob
        self.spx_epsilon = spx_epsilon

        self.sampling = FloatRandomSampling()
        self.mutation = PolynomialMutation(prob=pm, eta=eta)
        self.crossover = {
            'sbx': SBX(prob=sbx_prob, eta=sbx_eta),
            'de': DE(prob=de_prob, f=de_f),
            'pcx': PCX(prob=pcx_prob, eta=pcx_eta),
            'spx': SPX(prob=spx_prob, epsilon=spx_epsilon)
        }

    def _initialize_infill(self):
        return self.sampling(self.problem, self.pop_size)

    def _infill(self):
        # Select parents
        parents = self.pop.random(self.n_offsprings)

        # Apply crossover and mutation
        offspring = []
        for i in range(self.n_offsprings):
            off = parents[i]
            crossover_op = np.random.choice(list(self.crossover.values()))
            off = crossover_op.do(self.problem, Population.create(off))[0]
            off = self.mutation.do(self.problem, Population.create(off))[0]
            offspring.append(off)

        return Population.create(*offspring)

    def _advance(self, infills=None, **kwargs):
        if infills is None:
            infills = self.infill()

        # Evaluate the infill solutions
        self.evaluator.eval(self.problem, infills)

        # Merge the infills with the population
        self.pop = Population.merge(self.pop, infills)

        # Update the epsilon grid
        self.eps = self.calculate_epsilon(self.pop)

        # Get the epsilon-box dominance
        E = self.epsilon_dominance(self.pop.get("F"))

        # Do the non-dominated sorting
        fronts = NonDominatedSorting().do(E, sort_by_objectives=True)

        # Fill the new population
        new_pop = Population()
        for front in fronts:
            if len(new_pop) + len(front) <= self.pop_size:
                new_pop = Population.merge(new_pop, self.pop[front])
            else:
                I = self.survival.do(self.problem, self.pop[front], n_survive=self.pop_size - len(new_pop))
                new_pop = Population.merge(new_pop, self.pop[front[I]])
                break

        self.pop = new_pop

    def calculate_epsilon(self, pop):
        return np.std(pop.get("F"), axis=0) * self.eta

    def epsilon_dominance(self, F):
        eps = self.eps
        N = F.shape[0]
        M = F.shape[1]

        # Calculate the boxed objectives
        boxed = (F / eps).astype(int) * eps

        # Calculate the epsilon box dominance
        E = np.zeros((N, N))
        for i in range(N):
            E[i, :] = np.all(boxed[i] <= boxed, axis=1) & np.any(boxed[i] < boxed, axis=1)

        return E

class DE(Crossover):
    def __init__(self, prob=0.8, f=0.5):
        super().__init__(2, 1)
        self.prob = prob
        self.f = f

    def _do(self, problem, X, **kwargs):
        n_parents, n_matings, n_var = X.shape
        
        Xp = X[1:] - X[0]
        Xp = self.f * Xp + X[0]
        
        do_crossover = np.random.random(n_matings) < self.prob
        Xp[~do_crossover] = X[0][~do_crossover]
        
        return Xp

class PCX(Crossover):
    def __init__(self, prob=0.8, eta=0.1):
        super().__init__(3, 1)
        self.prob = prob
        self.eta = eta

    def _do(self, problem, X, **kwargs):
        n_parents, n_matings, n_var = X.shape
        
        g = X.mean(axis=0)
        D = X - g
        
        idx = np.random.choice(n_parents, size=n_matings)
        d = D[idx]
        
        w = np.random.normal(0, self.eta, size=(n_matings, n_var))
        
        Xp = g + d + w
        
        do_crossover = np.random.random(n_matings) < self.prob
        Xp[~do_crossover] = X[0][~do_crossover]
        
        return Xp

class SPX(Crossover):
    def __init__(self, prob=0.8, epsilon=1.0):
        super().__init__(3, 1)
        self.prob = prob
        self.epsilon = epsilon

    def _do(self, problem, X, **kwargs):
        n_parents, n_matings, n_var = X.shape
        
        G = X.mean(axis=0)
        r = np.random.random(size=(n_matings, n_parents))
        r = r / r.sum(axis=1)[:, None]
        
        Xp = (X.T * r.T).sum(axis=1).T
        Xp = G + self.epsilon * (Xp - G)
        
        do_crossover = np.random.random(n_matings) < self.prob
        Xp[~do_crossover] = X[0][~do_crossover]
        
        return Xp

def run_borg_moea(objective_func, bounds, n_obj, pop_size, n_gen, metrics):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    class DistributedProblem(Problem):
        def __init__(self, n_var, n_obj, xl, xu):
            super().__init__(n_var=n_var, n_obj=n_obj, xl=xl, xu=xu)

        def _evaluate(self, x, out, *args, **kwargs):
            local_pop_size = len(x) // size
            start = rank * local_pop_size
            end = start + local_pop_size if rank < size - 1 else len(x)
            local_x = x[start:end]

            local_f = np.array([objective_func(xi)[0] for xi in local_x])
            
            all_f = comm.allgather(local_f)
            f = np.concatenate(all_f)

            out["F"] = f

    problem = DistributedProblem(
        n_var=len(bounds),
        n_obj=n_obj,
        xl=np.array([b[0] for b in bounds]),
        xu=np.array([b[1] for b in bounds])
    )

    algorithm = BorgMOEA(
        pop_size=pop_size,
        n_offsprings=pop_size,
        sampling=FloatRandomSampling(),
        eliminate_duplicates=True
    )

    res = minimize(problem,
                   algorithm,
                   ('n_gen', n_gen),
                   verbose=rank==0)

    return res.X, res.F

class MOPSO(Algorithm):

    def __init__(self,
                 pop_size=100,
                 w=0.4,
                 c1=2.0,
                 c2=2.0,
                 **kwargs):

        super().__init__(**kwargs)
        self.pop_size = pop_size
        self.w = w  # inertia weight
        self.c1 = c1  # cognitive parameter
        self.c2 = c2  # social parameter
        
        self.sampling = FloatRandomSampling()

    def _initialize_infill(self):
        pop = self.sampling(self.problem, self.pop_size)
        
        # Initialize velocity
        self.velocity = np.zeros((self.pop_size, self.problem.n_var))
        
        # Initialize personal best
        self.p_best = pop.get("X").copy()
        self.p_best_F = pop.get("F").copy()
        
        # Initialize global best
        self.g_best = self.p_best[NonDominatedSorting().do(self.p_best_F, only_non_dominated_front=True)]
        self.g_best_F = self.p_best_F[NonDominatedSorting().do(self.p_best_F, only_non_dominated_front=True)]
        
        return pop

    def _infill(self):
        # Update velocity and position
        r1, r2 = np.random.rand(2, self.pop_size, self.problem.n_var)
        
        self.velocity = (self.w * self.velocity + 
                         self.c1 * r1 * (self.p_best - self.pop.get("X")) + 
                         self.c2 * r2 * (self.g_best[np.random.randint(len(self.g_best))] - self.pop.get("X")))
        
        X = self.pop.get("X") + self.velocity
        
        # Ensure the particles stay within bounds
        X = np.clip(X, self.problem.xl, self.problem.xu)
        
        return Population.new(X=X)

    def _advance(self, infills=None, **kwargs):
        if infills is None:
            infills = self.infill()

        # Evaluate the infill solutions
        self.evaluator.eval(self.problem, infills)

        # Update personal best
        mask = (infills.get("F") <= self.p_best_F).all(axis=1)
        self.p_best[mask] = infills.get("X")[mask]
        self.p_best_F[mask] = infills.get("F")[mask]

        # Update global best
        non_dominated = NonDominatedSorting().do(self.p_best_F, only_non_dominated_front=True)
        self.g_best = self.p_best[non_dominated]
        self.g_best_F = self.p_best_F[non_dominated]

        # Replace the populations
        self.pop = infills

def run_mopso(objective_func, bounds, n_obj, pop_size, n_gen, metrics):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    class DistributedProblem(Problem):
        def __init__(self, n_var, n_obj, xl, xu):
            super().__init__(n_var=n_var, n_obj=n_obj, xl=xl, xu=xu)

        def _evaluate(self, x, out, *args, **kwargs):
            local_pop_size = len(x) // size
            start = rank * local_pop_size
            end = start + local_pop_size if rank < size - 1 else len(x)
            local_x = x[start:end]

            local_f = np.array([objective_func(xi)[0] for xi in local_x])
            
            all_f = comm.allgather(local_f)
            f = np.concatenate(all_f)

            out["F"] = f

    problem = DistributedProblem(
        n_var=len(bounds),
        n_obj=n_obj,
        xl=np.array([b[0] for b in bounds]),
        xu=np.array([b[1] for b in bounds])
    )

    algorithm = MOPSO(
        pop_size=pop_size,
        sampling=FloatRandomSampling(),
        eliminate_duplicates=True
    )

    res = minimize(problem,
                   algorithm,
                   ('n_gen', n_gen),
                   verbose=rank==0)

    return res.X, res.F

def main():
    start_time = datetime.now()
    if rank == 0:
        logger.info(f"Multi-objective calibration started on {start_time.strftime('%Y/%m/%d %H:%M:%S')}")

    # Read the pertinent calibration parameters 
    algorithm = read_from_control(controlFolder/controlFile, 'moo_optimisation_algorithm')
    optimization_metrics = read_from_control(controlFolder/controlFile, 'moo_optimization_metrics').split(',')
    num_iter = int(read_from_control(controlFolder/controlFile, 'moo_num_iter'))
    pop_size = int(read_from_control(controlFolder/controlFile, 'moo_pop_size'))
    
    if rank == 0:
        logger.info(f"Using {algorithm} as the optimization algorithm")
        logger.info(f"Optimization metrics: {optimization_metrics}")
        logger.info(f"Local parameters being calibrated: {params_to_calibrate}")
        logger.info(f"Basin parameters being calibrated: {basin_params_to_calibrate}")

    # Read time periods
    calibration_period_str = read_from_control(controlFolder/controlFile, 'calibration_period')
    evaluation_period_str = read_from_control(controlFolder/controlFile, 'evaluation_period')
    calibration_period = parse_time_period(calibration_period_str)
    evaluation_period = parse_time_period(evaluation_period_str)

    if rank == 0:
        logger.info(f"Calibration period: {calibration_period[0].date()} to {calibration_period[1].date()}")
        logger.info(f"Evaluation period: {evaluation_period[0].date()} to {evaluation_period[1].date()}")

    # Create a partial function with the selected metrics and date ranges
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
        elif algorithm == "MOPSO":
            best_params, best_values = run_mopso(objective_func, all_bounds, n_obj=len(optimization_metrics), 
                                                pop_size=pop_size, n_gen=num_iter, metrics=optimization_metrics)
        else:
            raise ValueError(f"Invalid algorithm choice: {algorithm}")

        if rank == 0:
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
        if rank == 0:
            logger.exception(f"Error occurred during calibration: {str(e)}")
        raise

    if rank == 0:
        end_time = datetime.now()
        logger.info(f"Calibration ended on {end_time.strftime('%Y/%m/%d %H:%M:%S')}")
        logger.info(f"Total calibration time: {end_time - start_time}")

if __name__ == "__main__":
    main()