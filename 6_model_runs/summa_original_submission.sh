#!/bin/bash
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --mem=0
#SBATCH --constraint=broadwell
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --job-name=Summa-Actors
#SBATCH --output=
#SBATCH --account=
#SBATCH --array=0-129
#SBATCH --mail-type=ALL
#SBATCH --mail-user=darri.eythorsson@ucalgary.ca

# #################################################
# Authours: Kyle Klenk & Raymond J. Spiteri
# Script to submit Summa-Actors Jobs to Graham
# Job submitted as "sbatch summa_actors_submission.sh"
# #################################################

gru_max=18225
gru_count=128
max_job=142
cpu_num=32
summa_exe=home/darri/code/CWARHM/6_model_runs/1_run_summa_as_array.sh
file_manager=/project/6079554/darri/data/CWARHM_data/domain_Yukon/settings/SUMMA/fileManager.txt
log_dir=logs

offset=$SLURM_ARRAY_TASK_ID
gru_start=$(( 1 + gru_count*offset ))
job_check=$(( $gru_start + $gru_count ))

# Adust the number of grus for the last job
if [ $job_check -gt $gru_max ]
then
  gru_count=$(( gru_max-gru_start+1 ))
fi

gru_per_task=$(( gru_count/cpu_num ))

for ((cpu=0; cpu<$cpu_num; cpu++))
do
  task_gru_start=$(( gru_start + cpu * gru_per_task ))
  task_check=$(( task_gru_start + gru_per_task ))

# Adust the number of grus for the last task
if [ $task_check -gt $gru_max ] || \
    ( [ $task_check -lt $gru_max ] && \
      [ $cpu -eq $(( cpu_num-1 )) ] && \
      [ $offset -eq $(( max_job-1 )) ] )
then
  gru_per_task=$(( gru_max - task_gru_start))
fi

sbatch $summa_exe -g $task_gru_start $gru_per_task
#    -m $file_manager \
#    > $log_dir/summa_log_$task_gru_start.txt &
done
wait
