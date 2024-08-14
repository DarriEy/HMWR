#!/usr/bin/env python3

import sys
from mpi4py import MPI
from ostrich_util import OstrichOptimizer
from utils.config import initialize_config

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    config = initialize_config(rank, comm)
    optimizer = OstrichOptimizer(config, comm, rank)

    objective_value = optimizer.run_model_and_calculate_objective()

    # Write the objective value to a file that Ostrich can read
    with open('ostrich_objective.txt', 'w') as f:
        f.write(f"{objective_value}\n")

if __name__ == "__main__":
    main()