# logging_utils.py

import logging
from pathlib import Path
from datetime import datetime
from mpi4py import MPI # type: ignore

def setup_logging(domain_path, experiment_id, log_level=logging.DEBUG):
    # Get the MPI rank
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Create the log directory if it doesn't exist
    log_dir = Path(domain_path) / 'optimisation' / '_workflow_log'
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create a unique log file name with timestamp and rank
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'{experiment_id}_{timestamp}_rank{rank}.log'

    # Create a logger
    logger = logging.getLogger(f'SUMMA_optimization_rank{rank}')
    logger.setLevel(log_level)

    # Remove any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create formatter and add it to the handlers
    formatter = logging.Formatter(f'%(asctime)s - Rank {rank} - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Prevent the logger from propagating messages to the root logger
    logger.propagate = False

    return logger

def get_logger(name, root_path, domain_name, experiment_id, log_level=logging.DEBUG):
    log_dir = Path(root_path) / f'domain_{domain_name}' / 'optimisation' / '_workflow_log'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'{experiment_id}_{name}_{timestamp}.log'
    
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    
    return logger