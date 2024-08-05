#!/bin/bash
#SBATCH --cpus-per-task=32
#SBATCH --time=10:00:00
#SBATCH --mem=2GB
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --job-name=merge_array
#SBATCH --output=%x-%j.out
#SBATCH --account=def-mclark
#SBATCH --mail-type=ALL
#SBATCH --mail-user=darri.eythorsson@ucalgary.ca
python 2_make_all_weighted_forcing_files.py.py