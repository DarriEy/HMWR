import shutil
from pathlib import Path
from mpi4py import MPI # type: ignore
import os
import csv
import numpy as np # type: ignore
import logging

def create_rank_specific_directory(root_path, domain_name, rank_experiment_id, rank, model):
    rank_specific_path = Path(root_path) / f"domain_{domain_name}" / "simulations" / rank_experiment_id / model
    
    # Create directories
    rank_specific_path.mkdir(parents=True, exist_ok=True)
    (rank_specific_path / "run_settings").mkdir(exist_ok=True)
    (rank_specific_path / f"{model}_logs").mkdir(exist_ok=True)

    return rank_specific_path

def copy_summa_settings(source_settings_path, destination_settings_path):
    for file in ['attributes.nc', 'trialParams.nc', 'forcingFileList.txt', 'outputControl.txt', 'modelDecisions.txt', 'localParamInfo.txt', 'basinParamInfo.txt', 'fileManager.txt','coldState.nc', 'TBL_GENPARM.TBL', 'TBL_MPTABLE.TBL', 'TBL_SOILPARM.TBL', 'TBL_VEGPARM.TBL']:
        shutil.copy(source_settings_path / file, destination_settings_path / file)

def copy_mizuroute_settings(source_settings_path, destination_settings_path):
    for file in ['mizuroute.control', 'param.nml.default', 'topology.nc']:
        shutil.copy(source_settings_path / file, destination_settings_path / file)

def update_file_manager(file_path, rank_experiment_id, experiment_id):
    base_settings = "/settings/SUMMA/' !"
    rank_settings = f"/simulations/{rank_experiment_id}/SUMMA/run_settings/' !"
    with open(file_path, 'r') as f: 
        content = f.read()
    
    content = content.replace(experiment_id, rank_experiment_id)
    content = content.replace(base_settings, rank_settings)

    with open(file_path, 'w') as f:
        f.write(content)

def update_mizu_control_file(file_path, rank_experiment_id, experiment_id):
    base_settings = "/settings/mizuRoute/"
    rank_settings = f"/simulations/{rank_experiment_id}/mizuRoute/run_settings/"
    with open(file_path, 'r') as f:
        content = f.read()
    
    content = content.replace(experiment_id, rank_experiment_id)
    content = content.replace(base_settings, rank_settings)

    with open(file_path, 'w') as f:
        f.write(content)

def create_iteration_results_file(experiment_id, root_path, domain_name, all_params):
    """
    Create a new iteration results file with appropriate headers.
    """
    iteration_results_dir = Path(root_path) / f"domain_{domain_name}" / "optimisation"
    iteration_results_dir.mkdir(parents=True, exist_ok=True)
    iteration_results_file = iteration_results_dir / f"{experiment_id}_parallel_iteration_results.csv"
    
    # We'll create the file, but not write the header yet
    # The header will be written in the first call to write_iteration_results
    iteration_results_file.touch()
    
    return str(iteration_results_file)

def write_iteration_results(file_path, iteration, param_names, rank_params, calib_metrics, eval_metrics):
    """
    Write iteration results to the CSV file.
    """
    logger = logging.getLogger('write_iteration_results')
    logger.info(f"Writing results for iteration {iteration}")
    logger.info(f"Calib metrics: {calib_metrics}")
    logger.info(f"Eval metrics: {eval_metrics}")

    try:
        all_calib_keys = set().union(*[m for m in calib_metrics if m])
        all_eval_keys = set().union(*[m for m in eval_metrics if m])
        
        fieldnames = ['Iteration', 'Rank'] + param_names + \
                     [f'Calib_{k}' for k in sorted(all_calib_keys)] + \
                     [f'Eval_{k}' for k in sorted(all_eval_keys)]
        
        file_exists = os.path.exists(file_path) and os.path.getsize(file_path) > 0
        
        with open(file_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            
            for rank, (params, calib, eval) in enumerate(zip(rank_params, calib_metrics, eval_metrics)):
                row = {'Iteration': iteration, 'Rank': rank}
                row.update(dict(zip(param_names, params)))
                row.update({f'Calib_{k}': calib.get(k, np.nan) for k in sorted(all_calib_keys)})
                row.update({f'Eval_{k}': eval.get(k, np.nan) for k in sorted(all_eval_keys)})
                writer.writerow(row)
        
        logger.info(f"Results written successfully for iteration {iteration}")
    except Exception as e:
        logger.error(f"Error writing results for iteration {iteration}: {str(e)}", exc_info=True)
        raise

        