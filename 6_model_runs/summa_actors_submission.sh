#!/bin/bash
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --mem=0
#SBATCH --constraint=broadwell
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --job-name=Summa-Actors
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
#SBATCH --account=def-mclark
#SBATCH --array=0-46


# #################################################
# Authours: Kyle Klenk & Raymond J. Spiteri
# Script to submit Summa-Actors Jobs to Graham
# Job submitted as "sbatch summa_actors_submission.sh"
# #################################################

module load gcc/9.3.0
module load netcdf-fortran
module load openblas
module load caf

gru_max=18225
gru_count=400
summa_exe=/project/6079554/darri/data/CWARHM_data/installs/summa/bin/summa.exe
config_summa=/project/6079554/darri/data/CWARHM_data/domain_Yukon/settings/SUMMA/fileManager.txt

offset=$SLURM_ARRAY_TASK_ID
gru_start=$(( 1 + gru_count*offset ))
check=$(( $gru_start + $gru_count ))

# Adust the number of grus for the last job
if [ $check -gt $gru_max ]
then
  gru_count=$(( gru_max-gru_start+1 ))
fi

$summa_exe \
  -g $gru_start \
  -n $gru_count \
  -c $config_summa \
  --caf.scheduler.max-threads=$SLURM_CPUS_PER_TASK
