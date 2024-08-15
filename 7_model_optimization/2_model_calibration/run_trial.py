import os
import sys
from pathlib import Path

# Get the directory of the current script
current_dir = Path(__file__).resolve().parent

# Add the parent directory of the current script to the Python path
sys.path.append(str(current_dir.parent.parent))

from mpi4py import MPI # type: ignore
from ostrich_util import OstrichOptimizer
from utils.config import initialize_config # type: ignore

def find_control_file():
    # Start from the current directory and go up the directory tree
    current_path = Path.cwd()
    while current_path != current_path.root:
        control_file_path = current_path / 'code' / 'CWARHM' / '0_control_files' / 'control_active.txt'
        if control_file_path.exists():
            return control_file_path
        current_path = current_path.parent
    
    raise FileNotFoundError("Control file 'control_active.txt' not found in the directory tree")

def main():
    print(f"Current working directory: {os.getcwd()}")
    print(f"run_trial.py location: {os.path.abspath(__file__)}")
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    try:
        control_file_path = find_control_file()
        print(f"Control file found at: {control_file_path}")
    except FileNotFoundError as e:
        print(str(e))
        return

    config = initialize_config(rank, comm)
    optimizer = OstrichOptimizer(config, comm, rank)

    objective_value = optimizer.run_model_and_calculate_objective()

    # Write the objective value to a file that Ostrich can read
    with open('ostrich_objective.txt', 'w') as f:
        f.write(f"{objective_value}\n")

if __name__ == "__main__":
    main()