import os
import subprocess
from typing import List, Tuple, Dict, Any
from mpi4py import MPI # type: ignore
import numpy as np
import netCDF4 as nc # type: ignore
import xarray as xr # type: ignore
import sys
from pathlib import Path
import shutil
import datetime
import time
import pandas as pd # type: ignore

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from utils.config import initialize_config, Config # type: ignore  # type: ignore 
from utils.logging_utils import get_logger # type: ignore
from utils.calculate_sim_stats import get_KGE, get_KGEp, get_NSE, get_MAE, get_RMSE, get_KGEnp  # type: ignore

class OstrichOptimizer:
    def __init__(self, config: Config, comm: MPI.Comm, rank: int):
        self.config = config
        self.comm = comm
        self.rank = rank
        self.size = comm.Get_size()
        self.logger = get_logger('OstrichOptimizer', config.root_path, config.domain_name, config.experiment_id)
        self.use_mpi = config.use_mpi

    def prepare_ostrich_files(self):
        self.generate_multiplier_bounds()
        self.create_ostrich_input()


    def create_ostrich_input(self):
        self.logger.info("Creating Ostrich input files")
        
        # Generate multiplier template file
        self.create_multiplier_template()
        
        # Generate initial multiplier values file
        self.create_initial_multipliers()
        
        # Generate Ostrich configuration file (ostIn.txt)
        self.create_ostrich_config()
        
        self.logger.info("Ostrich input files created successfully")

    def create_multiplier_template(self):
        template_file = self.config.ostrich_path / 'multipliers.tpl'
        with open(template_file, 'w') as f:
            for param in self.config.params_to_calibrate:
                f.write(f"{param}_multp\n")
        self.logger.info(f"Multiplier template file created: {template_file}")

    def create_initial_multipliers(self):
        initial_file = self.config.ostrich_path / 'multipliers.txt'
        with open(initial_file, 'w') as f:
            for _ in self.config.params_to_calibrate:
                f.write("1.0\n")  # Start with neutral multipliers
        self.logger.info(f"Initial multipliers file created: {initial_file}")

    def calculate_objective_function(self):
        self.logger.info("Calculating objective function")

        try:
            calib_metrics, eval_metrics = self.evaluate_model()
            
            if self.config.optimization_metric in calib_metrics:
                objective_value = calib_metrics[self.config.optimization_metric]
                
                # For metrics where higher values are better, we negate the value for minimization
                if self.config.optimization_metric in ['KGE', 'KGEp', 'KGEnp', 'NSE']:
                    objective_value = -objective_value
                
                self.logger.info(f"Objective function ({self.config.optimization_metric}) value: {objective_value}")
                return objective_value
            else:
                self.logger.error(f"Optimization metric {self.config.optimization_metric} not found in calibration metrics")
                return float('inf')  # Return a large value to indicate failure
        
        except Exception as e:
            self.logger.error(f"Error calculating objective function: {str(e)}")
            return float('inf')  # Return a large value to indicate failure

    def evaluate_model(self):
        self.logger.info("Evaluating model performance")

        try:
            # Get the mizuRoute output file
            mizuroute_output_path = self.get_mizuroute_output_path()
            
            # Open the mizuRoute output file
            sim_data = xr.open_dataset(mizuroute_output_path)
            
            # Get the simulated streamflow for the specified reach
            segment_index = sim_data['reachID'].values == int(self.config.sim_reach_ID)
            sim_df = sim_data.sel(seg=segment_index)['IRFroutedRunoff'].to_dataframe().reset_index()
            sim_df.set_index('time', inplace=True)

            # Read observation data
            obs_df = pd.read_csv(self.config.obs_file_path, index_col='datetime', parse_dates=True)
            obs_df = obs_df['discharge_cms'].resample('h').mean()

            # Calculate metrics for calibration period
            calib_metrics = self.calculate_metrics(
                obs_df.loc[self.config.calib_period[0]:self.config.calib_period[1]],
                sim_df.loc[self.config.calib_period[0]:self.config.calib_period[1]]['IRFroutedRunoff']
            )

            # Calculate metrics for evaluation period
            eval_metrics = self.calculate_metrics(
                obs_df.loc[self.config.eval_period[0]:self.config.eval_period[1]],
                sim_df.loc[self.config.eval_period[0]:self.config.eval_period[1]]['IRFroutedRunoff']
            )

            self.logger.info(f"Calibration metrics: {calib_metrics}")
            self.logger.info(f"Evaluation metrics: {eval_metrics}")

            return calib_metrics, eval_metrics

        except Exception as e:
            self.logger.error(f"Error evaluating model: {str(e)}")
            return None, None

    def calculate_metrics(self, obs: np.ndarray, sim: np.ndarray) -> Dict[str, float]:
        """Calculate performance metrics."""
        return {
            'RMSE': get_RMSE(obs, sim, transfo=1),
            'KGE': get_KGE(obs, sim, transfo=1),
            'KGEp': get_KGEp(obs, sim, transfo=1),
            'NSE': get_NSE(obs, sim, transfo=1),
            'MAE': get_MAE(obs, sim, transfo=1),
            'KGEnp': get_KGEnp(obs, sim, transfo=1)
        }

    def get_mizuroute_output_path(self):
        mizuroute_settings_path = self.get_mizuroute_settings_path()
        output_dir = mizuroute_settings_path.parent
        output_file = output_dir / f"{self.config.experiment_id}_rank{self.rank + 1}.h.*.nc"  # Adjust the file pattern as needed
        matching_files = list(output_dir.glob(output_file.name))
        
        if not matching_files:
            raise FileNotFoundError(f"No matching mizuRoute output file found: {output_file}")
        
        return str(matching_files[0])

    def create_ostrich_config(self):
        self.logger.info("Creating Ostrich configuration")
        
        template_file = self.config.ostrich_path / 'ostIn.tpl'
        ostrich_config = self.config.ostrich_path / 'ostIn.txt'
        
        self.logger.info(f"Creating Ostrich template file: {template_file}")
        template_content = f"""ProgramType {self.config.ostrich_algorithm}

BeginFilePairs
multipliers.tpl ; multipliers.txt
EndFilePairs

BeginExtraFiles
run_trial.sh
run_trial.py
EndExtraFiles

BeginParams
{self.generate_param_definitions()}
EndParams

BeginObservations
Obj1 1.0 1.0 ostrich_objective.txt ; OST_NULL 0 1
EndObservations

BeginResponseVars
Obj1 ostrich_objective.txt ; OST_NULL 0 1
EndResponseVars

BeginGCOP
PenaltyFunction APM
EndGCOP

BeginConstraints
EndConstraints

BeginObjFun
Obj1
EndObjFun

ModelExecutable ./run_trial.sh

BeginDDSAlg
PerturbationValue 0.2
MaxIterations {self.config.num_iter}
UseRandomParamValues
EndDDSAlg
"""
        with open(template_file, 'w') as f:
            f.write(template_content)
        
        # Read template and create actual config
        with open(template_file, 'r') as src, open(ostrich_config, 'w') as dst:
            content = src.read()
        
            # Replace placeholders
            content = content.replace('ALGORITHM_PLACEHOLDER', self.config.ostrich_algorithm)
            content = content.replace('MODEL_COMMAND_PLACEHOLDER', './run_trial.sh')
            content = content.replace('RANDOM_SEED_PLACEHOLDER', str(int(time.time() * 1000000) % 1000000000))
            content = content.replace('MAX_ITERATIONS_PLACEHOLDER', str(self.config.num_iter))
            
            # Add parameter definitions
            param_definitions = self.generate_param_definitions()
            content = content.replace('PARAM_DEFINITIONS_PLACEHOLDER', param_definitions)
            
            dst.write(content)
        
        self.logger.info(f"Ostrich configuration file created: {ostrich_config}")

    def generate_param_definitions(self):
        definitions = []
        for param, (lower, upper) in zip(self.config.params_to_calibrate, self.config.all_bounds):
            definitions.append(f"{param}_multp\t1.0\t{lower}\t{upper}\tnone\tnone\tnone\tfree")
        return "\n".join(definitions)

    def run_optimization(self):
        self.generate_multiplier_bounds()
        self.create_ostrich_input()
        
        if self.run_ostrich():
            best_params, best_objective = self.parse_ostrich_results()
            if best_params is not None and best_objective is not None:
                self.logger.info("Optimization completed successfully")
                return best_params, best_objective
            else:
                self.logger.error("Failed to parse Ostrich results")
                return None, None
        else:
            self.logger.error("Ostrich optimization failed")
            return None, None

    def run_ostrich(self):
        ostrich_path = Path(self.config.ostrich_path)
        ostrich_exe = ostrich_path / self.config.ostrich_exe

        if self.use_mpi:
            # For MPI version, all processes participate
            mpi_command = f"mpiexec -n {self.size} {ostrich_exe}"
            self.logger.info(f"Running Ostrich with MPI: {mpi_command}")
            result = self.comm.bcast(subprocess.run(mpi_command, shell=True, check=True, capture_output=True, text=True, cwd=ostrich_path) if self.rank == 0 else None, root=0)
        else:
            # For serial version, only root process runs Ostrich
            if self.rank == 0:
                ostrich_command = f"{ostrich_exe}"
                self.logger.info(f"Running Ostrich in serial mode: {ostrich_command}")
                result = subprocess.run(ostrich_command, shell=True, check=True, capture_output=True, text=True, cwd=ostrich_path)
            else:
                result = None

        if self.rank == 0:
            self.logger.info(f"Ostrich run completed with return code: {result.returncode}")
            if result.stdout:
                self.logger.debug(f"Ostrich stdout: {result.stdout}")
            if result.stderr:
                self.logger.warning(f"Ostrich stderr: {result.stderr}")

        self.comm.Barrier()  # Ensure all processes wait for Ostrich to complete

    def parse_ostrich_results(self):
        ostrich_output = Path(self.config.ostrich_path) / 'OstOutput0.txt'
        
        if self.rank == 0:
            with open(ostrich_output, 'r') as f:
                lines = f.readlines()
            
            # Find the line with the best parameter set
            best_line = None
            for line in reversed(lines):
                if line.strip().startswith('0'):
                    best_line = line
                    break
            
            if best_line is None:
                raise ValueError("Could not find best parameter set in Ostrich output")
            
            parts = best_line.split()
            best_value = float(parts[1])
            best_params = [float(p) for p in parts[2:]]
            
            self.logger.info(f"Best objective value: {best_value}")
            self.logger.info(f"Best parameters: {best_params}")
        else:
            best_value = None
            best_params = None
        
        # Broadcast results to all processes
        best_value = self.comm.bcast(best_value, root=0)
        best_params = self.comm.bcast(best_params, root=0)
        
        return best_params, best_value

    def update_trial_params(self):
        self.logger.info("Updating trial parameters")

        # Read multiplier values

        multipliers_file = self.config.ostrich_path / 'multipliers.txt'
        with open(multipliers_file, 'r') as f:
            multipliers = [float(line.strip()) for line in f]

        if len(multipliers) != len(self.config.params_to_calibrate):
            self.logger.error("Mismatch between number of multipliers and parameters to calibrate")
            return False

        # Get paths for parameter files
        summa_settings_path = self.get_summa_settings_path()
        trial_param_file = summa_settings_path / self.config.trial_param_file
        trial_param_file_priori = trial_param_file.with_name(f"{trial_param_file.stem}.priori.nc")

        # Update parameter values
        with xr.open_dataset(trial_param_file_priori) as src, xr.open_dataset(trial_param_file, mode='a') as dst:
            for param, multiplier in zip(self.config.params_to_calibrate, multipliers):
                if param in src.variables:
                    if param != 'thickness':
                        # Update regular parameters
                        dst[param][:] = src[param][:] * multiplier
                    else:
                        # Special handling for 'thickness' parameter
                        canopy_top = dst['heightCanopyTop'][:]
                        canopy_bottom = dst['heightCanopyBottom'][:]
                        thickness = canopy_top - canopy_bottom
                        new_thickness = thickness * multiplier
                        dst['heightCanopyTop'][:] = canopy_bottom + new_thickness

                    self.logger.info(f"Updated parameter {param} with multiplier {multiplier}")
                else:
                    self.logger.warning(f"Parameter {param} not found in trial parameter file")

        self.logger.info("Trial parameters updated successfully")
        return True

    def run_model_and_calculate_objective(self):
        self.logger.info("Running model and calculating objective function")

        # Update trial parameters
        self.update_trial_params()
            #self.logger.error("Failed to update trial parameters")
            #return float('inf')  # Return a large value to indicate failure

        # Run SUMMA
        summa_success = self.run_summa()
        if not summa_success:
            self.logger.error("SUMMA run failed")
            return float('inf')

        # Run mizuRoute
        mizuroute_success = self.run_mizuroute()
        if not mizuroute_success:
            self.logger.error("mizuRoute run failed")
            return float('inf')

        # Calculate objective function
        objective_value = self.calculate_objective_function()

        self.logger.info(f"Objective function value: {objective_value}")
        return objective_value

    def run_summa(self):
        summa_exe = f'{self.config.root_path}/installs/summa/bin/{self.config.exe_name_summa}'
        file_manager = self.get_summa_settings_path() / self.config.settings_summa_filemanager
        cmd = f"{summa_exe} -m {file_manager}"
        
        self.logger.info(f"Running SUMMA with command: {cmd}")
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        return result.returncode == 0

    def run_mizuroute(self):
        mizuroute_exe = f'{self.config.root_path}/installs/mizuroute/route/bin/{self.config.exe_name_mizuroute}'
        control_file = self.get_mizuroute_settings_path() / self.config.mizu_control_file
        cmd = f"{mizuroute_exe} {control_file}"
        
        self.logger.info(f"Running mizuRoute with command: {cmd}")
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        return result.returncode == 0

    def get_rank_specific_path(self, base_path: Path, rank_experiment_id: str, model: str) -> Path:
        return base_path / "simulations" / rank_experiment_id / model

    def get_summa_settings_path(self) -> Path:
        rank_experiment_id = f"{self.config.experiment_id}_rank{self.rank + 1}"
        rank_specific_path = self.get_rank_specific_path(Path(self.config.root_path) / f"domain_{self.config.domain_name}", 
                                                         rank_experiment_id, "SUMMA")
        return rank_specific_path / "run_settings"

    def get_mizuroute_settings_path(self) -> Path:
        rank_experiment_id = f"{self.config.experiment_id}_rank{self.rank + 1}"
        rank_specific_path = self.get_rank_specific_path(Path(self.config.root_path) / f"domain_{self.config.domain_name}", 
                                                         rank_experiment_id, "mizuRoute")
        return rank_specific_path / "run_settings"

    def update_ostrich_config_for_calibration(self):
        if self.rank == 0:  # Only root process updates the config
            ostrich_config_path = Path(self.config.ostrich_path) / 'ostIn.txt'

            with open(ostrich_config_path, 'r') as f:
                lines = f.readlines()

            for i, line in enumerate(lines):
                if 'BeginPeriod' in line:
                    lines[i+1] = f"   {self.config.calib_period[0].strftime('%Y-%m-%d %H:%M')}\n"
                    lines[i+2] = f"   {self.config.calib_period[1].strftime('%Y-%m-%d %H:%M')}\n"
                    break

            with open(ostrich_config_path, 'w') as f:
                f.writelines(lines)

            self.logger.info(f"Updated Ostrich config for calibration period: {self.config.calib_period[0]} to {self.config.calib_period[1]}")

        self.comm.Barrier()  # Ensure all processes wait for config update

    def generate_priori_trial_params(self):
        self.logger.info("Generating a priori trial parameters")
        
        summa_settings_path = self.get_summa_settings_path()
        
        # Update outputControl.txt
        self.update_output_control(summa_settings_path)
        
        # Update fileManager.txt
        self.update_file_manager(summa_settings_path)
        
        # Run SUMMA to generate a priori parameter values
        self.run_summa_for_priori()
        
        # Extract a priori parameter values and create trialParam.priori.nc
        self.create_priori_trial_param_file(summa_settings_path)
        
        self.reset_file_manager(summa_settings_path)

        self.logger.info("A priori trial parameters generated successfully")

    def update_output_control(self, settings_path):
        output_control_file = settings_path / self.config.settings_summa_output
        output_control_temp = output_control_file.with_suffix('.temp')
        
        output_params = self.config.params_to_calibrate.copy()
        
        # Add additional parameters if needed (e.g., soil and canopy height parameters)
        soil_params = ['theta_res', 'critSoilWilting', 'critSoilTranspire', 'fieldCapacity', 'theta_sat']
        height_params = ['heightCanopyBottom', 'heightCanopyTop']
        
        output_params.extend([param for param in soil_params + height_params if param not in output_params])
        
        with open(output_control_file, 'r') as src, open(output_control_temp, 'w') as dst:
            for param in output_params:
                dst.write(f"{param}\n")
            dst.write(src.read())
        
        shutil.move(output_control_temp, output_control_file)

    def update_file_manager(self, settings_path):
        file_manager = settings_path / self.config.settings_summa_filemanager
        file_manager_temp = file_manager.with_suffix('.temp')
        
        sim_start_time = self.config.calib_period[0]
        sim_end_time = (sim_start_time + datetime.timedelta(days=1)).strftime('%Y-%m-%d %H:%M')
        
        with open(file_manager, 'r') as src, open(file_manager_temp, 'w') as dst:
            for line in src:
                if line.startswith('simStartTime'):
                    line = f"simStartTime '{sim_start_time}'\n"
                elif line.startswith('simEndTime'):
                    line = f"simEndTime '{sim_end_time}'\n"
                dst.write(line)
        
        shutil.move(file_manager_temp, file_manager)

    def reset_file_manager(self, settings_path):
        file_manager = settings_path / self.config.settings_summa_filemanager
        file_manager_temp = file_manager.with_suffix('.temp')
        
        sim_start_time = (self.config.calib_period[0] - datetime.timedelta(days=365))
        sim_end_time = self.config.eval_period[1]
        
        with open(file_manager, 'r') as src, open(file_manager_temp, 'w') as dst:
            for line in src:
                if line.startswith('simStartTime'):
                    line = f"simStartTime '{sim_start_time}'\n"
                elif line.startswith('simEndTime'):
                    line = f"simEndTime '{sim_end_time}'\n"
                dst.write(line)
        
        shutil.move(file_manager_temp, file_manager)

    def run_summa_for_priori(self):

        summa_exe = f'{self.config.root_path}/installs/summa/bin/{self.config.exe_name_summa}'
        file_manager = self.get_summa_settings_path() / self.config.settings_summa_filemanager
        cmd = f"{summa_exe} -m {file_manager}"
        
        self.logger.info(f"Running SUMMA with command: {cmd}")
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        self.logger.info("SUMMA run completed")

    def create_priori_trial_param_file(self, settings_path):
        rank_experiment_id = f"{self.config.experiment_id}_rank{self.rank + 1}" #Remember to fix for rank == 0
        output_path = self.get_rank_specific_path(Path(self.config.root_path) / f"domain_{self.config.domain_name}", rank_experiment_id, "SUMMA")
        
        output_prefix = rank_experiment_id
        output_file = output_path / f"{output_prefix}_timestep.nc"
        
        attribute_file = settings_path / self.config.settings_summa_attributes
        trial_param_file = settings_path / self.config.trial_param_file
        trial_param_file_priori = trial_param_file.with_name(f"{trial_param_file.stem}.priori.nc")
        
        with nc.Dataset(output_file, 'r') as ff, nc.Dataset(attribute_file, 'r') as src, nc.Dataset(trial_param_file_priori, 'w') as dst:
            # Copy dimensions from attribute file
            for name, dimension in src.dimensions.items():
                dst.createDimension(name, len(dimension) if not dimension.isunlimited() else None)
            
            # Copy gruId and hruId variables
            for name in ['gruId', 'hruId']:
                x = dst.createVariable(name, src[name].datatype, src[name].dimensions)
                dst[name].setncatts(src[name].__dict__)
                dst[name][:] = src[name][:]
            
            # Create parameter variables
            for param_name in self.config.params_to_calibrate:
                if param_name in ff.variables:
                    param_dims = ff[param_name].dimensions
                    if 'hru' in param_dims:
                        param_dim = 'hru'
                    elif 'gru' in param_dims:
                        param_dim = 'gru'
                    else:
                        self.logger.warning(f"Unexpected dimensions for parameter {param_name}: {param_dims}")
                        continue
                    
                    dst.createVariable(param_name, 'f8', param_dim, fill_value=np.nan)
                    if param_dims == ('depth', 'hru'):
                        dst[param_name][:] = ff[param_name][0, :]  # Use first depth value
                    else:
                        dst[param_name][:] = ff[param_name][:]
                else:
                    self.logger.warning(f"Parameter {param_name} not found in SUMMA output")
        
        # Copy priori file to trial param file
        shutil.copy2(trial_param_file_priori, trial_param_file)

    def generate_multiplier_bounds(self):
        self.generate_priori_trial_params()
        self.logger.info("Generating multiplier bounds")
        
        summa_settings_path = self.get_summa_settings_path()
        
        # Read object parameters
        object_params = self.config.params_to_calibrate
        self.logger.info(f"Parameters to calibrate: {object_params}")
        
        # Read parameter files
        basin_param_file = summa_settings_path / self.config.basin_parameters_file_name
        local_param_file = summa_settings_path / self.config.local_parameters_file_name
        
        basin_params = self.read_param_file(basin_param_file)
        local_params = self.read_param_file(local_param_file)
        
        # Read a priori parameter values
        trial_param_file = summa_settings_path / self.config.trial_param_file
        self.logger.info(f"Reading trial parameter file: {trial_param_file}")
        with xr.open_dataset(trial_param_file) as ds:
            self.logger.info(f"Variables in trial parameter file: {list(ds.variables)}")
            a_priori_params = {}
            for param in object_params:
                if param in ds.variables:
                    a_priori_params[param] = ds[param].values
                else:
                    self.logger.warning(f"Parameter {param} not found in trial parameter file")
        
        # Calculate multiplier bounds
        multp_bounds_list = []
        for param in object_params:
            if param not in a_priori_params:
                self.logger.error(f"No a priori value found for parameter {param}")
                continue
            
            if param in local_params:
                bounds = self.calculate_local_param_bounds(param, local_params[param], a_priori_params[param])
            elif param in basin_params:
                bounds = self.calculate_basin_param_bounds(param, basin_params[param], a_priori_params[param])
            else:
                self.logger.error(f"Parameter {param} not found in local or basin parameters")
                continue
            
            multp_bounds_list.append([f"{param}_multp"] + bounds)
        
        # Save multiplier bounds
        self.save_multiplier_bounds(multp_bounds_list)
        
        self.logger.info("Multiplier bounds generated successfully")
        self.logger.info(f"Generated bounds for {len(multp_bounds_list)} parameters")

    def read_param_file(self, file_path):
        params = {}
        with open(file_path, 'r') as f:
            for line in f:
                if not line.startswith(('!', "'")):
                    parts = line.strip().split('|')
                    if len(parts) >= 4:
                        name = parts[0].strip()
                        min_val = float(parts[2].strip().replace('d', 'e'))
                        max_val = float(parts[3].strip().replace('d', 'e'))
                        params[name] = (min_val, max_val)
        return params

    def calculate_local_param_bounds(self, param, bounds, a_priori):
        min_val, max_val = bounds
        multp_min = np.max(min_val / a_priori)
        multp_max = np.min(max_val / a_priori)
        multp_initial = 1.0 if multp_min < 1 < multp_max else np.mean([multp_min, multp_max])
        return [multp_initial, multp_min, multp_max]

    def calculate_basin_param_bounds(self, param, bounds, a_priori):
        return self.calculate_local_param_bounds(param, bounds, a_priori)

    def save_multiplier_bounds(self, multp_bounds_list):
        output_file = self.config.ostrich_path / 'multiplier_bounds.txt'
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            f.write('# MultiplierName,InitialValue,LowerLimit,UpperLimit\n')
            for bounds in multp_bounds_list:
                f.write(f"{bounds[0]},{bounds[1]:.6f},{bounds[2]:.6f},{bounds[3]:.6f}\n")

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    config = initialize_config(rank, comm)
    optimizer = OstrichOptimizer(config, comm, rank)
    
    best_params, best_objective = optimizer.run_optimization()
    
    if rank == 0:
        if best_params is not None and best_objective is not None:
            print(f"Optimization completed. Best objective: {best_objective}")
            print("Best parameters:")
            for param, value in best_params.items():
                print(f"{param}: {value}")
        else:
            print("Optimization failed.")

if __name__ == "__main__":
    main()
