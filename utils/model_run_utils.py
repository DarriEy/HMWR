import subprocess
from pathlib import Path
from utils.parallel_calibration_utils import copy_summa_settings, copy_mizuroute_settings, update_file_manager, update_mizu_control_file, create_rank_specific_directory # type: ignore
from utils.calibration_utils import update_param_files # type: ignore
import json
import os

def run_summa(summa_path, summa_exe, filemanager_path, log_path, log_name, local_rank):
    """
    Run SUMMA model.

    Parameters:
    summa_path (str): Path to SUMMA executable directory
    summa_exe (str): Name of SUMMA executable
    filemanager_path (Path): Path to SUMMA file manager
    log_path (Path): Path to store log files
    log_name (str): Name of the log file
    local_rank (int): Rank of the current process

    Returns:
    int: Return code of the SUMMA process
    """
    # Ensure log directory exists
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Construct SUMMA command
    summa_command = [str(Path(summa_path) / summa_exe), "-m", str(filemanager_path)]

    # Run SUMMA
    with open(log_path / log_name, 'w') as log_file:
        summa_result = subprocess.run(summa_command, check=False, stdout=log_file, stderr=subprocess.STDOUT, text=True)
    
    return summa_result.returncode

def run_mizuroute(mizuroute_path, mizuroute_exe, control_path, log_path, log_name, local_rank):
    """
    Run mizuRoute model.

    Parameters:
    mizuroute_path (str): Path to mizuRoute executable directory
    mizuroute_exe (str): Name of mizuRoute executable
    control_path (Path): Path to mizuRoute control file
    log_path (Path): Path to store log files
    log_name (str): Name of the log file
    local_rank (int): Rank of the current process

    Returns:
    int: Return code of the mizuRoute process
    """
    # Ensure log directory exists
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Construct mizuRoute command
    mizuroute_command = [str(Path(mizuroute_path) / mizuroute_exe), str(control_path)]

    # Run mizuRoute
    with open(log_path / log_name, 'w') as log_file:
        mizuroute_result = subprocess.run(mizuroute_command, check=False, stdout=log_file, stderr=subprocess.STDOUT, text=True)
    
    return mizuroute_result.returncode



def prepare_model_run(root_path, domain_name, experiment_id, rank_experiment_id, local_rank, 
                      local_param_values, basin_param_values, params_to_calibrate, basin_params_to_calibrate,
                      local_bounds_dict, basin_bounds_dict, filemanager_name, mizu_control_file):
    """
    Prepare the model run by setting up directories and updating configuration files.

    Parameters:
    root_path (str): Root path of the project
    domain_name (str): Name of the domain
    experiment_id (str): ID of the experiment
    rank_experiment_id (str): Rank-specific experiment ID
    local_rank (int): Rank of the current process
    local_param_values (list): Local parameter values
    basin_param_values (list): Basin parameter values
    params_to_calibrate (list): Names of local parameters to calibrate
    basin_params_to_calibrate (list): Names of basin parameters to calibrate
    local_bounds_dict (dict): Dictionary of local parameter bounds
    basin_bounds_dict (dict): Dictionary of basin parameter bounds
    filemanager_name (str): Name of the SUMMA file manager file
    mizu_control_file (str): Name of the mizuRoute control file

    Returns:
    tuple: Paths to rank-specific directories and settings
    """
    rank_specific_path = create_rank_specific_directory(root_path, domain_name, rank_experiment_id, local_rank, "SUMMA")
    mizuroute_rank_specific_path = create_rank_specific_directory(root_path, domain_name, rank_experiment_id, local_rank, "mizuRoute")
    
    # Copy SUMMA and mizuRoute settings to rank-specific directory
    summa_source_settings_path = Path(root_path) / f"domain_{domain_name}" / "settings" / "SUMMA"
    summa_destination_settings_path = rank_specific_path / "run_settings"
    mizuroute_source_settings_path = Path(root_path) / f"domain_{domain_name}" / "settings" / "mizuRoute"
    mizuroute_destination_settings_path = mizuroute_rank_specific_path / "run_settings"
    copy_mizuroute_settings(mizuroute_source_settings_path, mizuroute_destination_settings_path)
    copy_summa_settings(summa_source_settings_path, summa_destination_settings_path)

    # Update parameter files
    update_param_files(local_param_values, basin_param_values, 
                       params_to_calibrate, basin_params_to_calibrate, 
                       summa_destination_settings_path / "localParamInfo.txt", 
                       summa_destination_settings_path / "basinParamInfo.txt",
                       local_bounds_dict, basin_bounds_dict)
    
    # Update file manager
    update_file_manager(summa_destination_settings_path / filemanager_name, rank_experiment_id, experiment_id)
    update_mizu_control_file(mizuroute_destination_settings_path / mizu_control_file, rank_experiment_id, experiment_id)

    return rank_specific_path, mizuroute_rank_specific_path, summa_destination_settings_path, mizuroute_destination_settings_path


def write_failed_run_info(root_path, domain_name, experiment_id, local_rank, attempt, params, summa_log_path, summa_log_name):
    """
    Write information about a failed SUMMA run to a special file with a unique, sequentially numbered name.

    Parameters:
    root_path (str): Root path of the project
    domain_name (str): Name of the domain
    experiment_id (str): ID of the experiment
    local_rank (int): Rank of the current process
    attempt (int): Attempt number of the failed run
    params (dict): Dictionary of parameter names and values
    summa_log_path (Path): Path to the SUMMA log file
    summa_log_name (str): Name of the SUMMA log file
    """
    failed_runs_dir = Path(root_path) / f"domain_{domain_name}" / "optimisation" / "failed_runs"
    failed_runs_dir.mkdir(parents=True, exist_ok=True)
    
    # Get the current highest number for failed run files
    existing_files = [f for f in os.listdir(failed_runs_dir) if f.startswith(f"{experiment_id}_rank{local_rank}_attempt{attempt}_failed_run")]
    current_highest = 0
    for file in existing_files:
        try:
            num = int(file.split('_')[-1].split('.')[0])
            current_highest = max(current_highest, num)
        except ValueError:
            pass
    
    # Create a new filename with the next number
    new_file_number = current_highest + 1
    failed_run_file = failed_runs_dir / f"{experiment_id}_rank{local_rank}_attempt{attempt}_failed_run_{new_file_number:03d}.txt"
    
    with open(failed_run_file, 'w') as f:
        f.write(f"Failed run information for {experiment_id}, rank {local_rank}, attempt {attempt}\n\n")
        f.write("Parameter values:\n")
        f.write(json.dumps(params, indent=2))
        f.write("\n\nSUMMA log:\n")
        with open(summa_log_path / summa_log_name, 'r') as log_file:
            f.write(log_file.read())