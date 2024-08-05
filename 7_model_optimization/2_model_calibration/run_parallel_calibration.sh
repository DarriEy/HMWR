#!/bin/bash
#SBATCH --job-name=SUMMA_calibration
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=32
#SBATCH --time=12:00:00
#SBATCH --output=calibration_%j.out
#SBATCH --error=calibration_%j.err
#SBATCH --mem=30G

module load python/3.11.5
module load openmpi/4.1.5
module load mpi4py/3.1.6

mpirun -np 128 python SUMMA_parameter_estimation_parallel.py