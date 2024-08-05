#!/bin/bash
#SBATCH --cpus-per-task=32
#SBATCH --time=1:00:00
#SBATCH --mem=0
#SBATCH --constraint=broadwell
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --job-name=Summa-Actors
#SBATCH --output=%x-%j.out

#SBATCH --account=def-mclark
#SBATCH --array=0-3
#SBATCH --mail-type=ALL
#SBATCH --mail-user=darri.eythorsson@ucalgary.ca

# Script to run SUMMA as array job
# Reads most required info from 'summaWorkflow_public/0_control_files/control_active.txt'
#
# Also needs two arguments as input:
# gru_start and gru_count
#
# These are used to supply SUMMA with the -g argument: -g gru_start gru_count

# --- Command line arguments

module load gcc/9.3.0
module load netcdf-fortran
module load openblas
module load caf

gru_max=153
gru_count=40

offset=$SLURM_ARRAY_TASK_ID
gru_start=$(( 1 + gru_count*offset ))
check=$(( $gru_start + $gru_count ))

# Adust the number of grus for the last job
if [ $check -gt $gru_max ]
then
  gru_count=40
fi

# --- Settings
# - Find the SUMMA install dir 
# ----------------------------
setting_line=$(grep -m 1 "^install_path_summa" ../0_control_files/control_active.txt) # -m 1 ensures we only return the top-most result. This is needed because variable names are sometimes used in comments in later lines

# Extract the path
summa_path=$(echo ${setting_line##*|}) # remove the part that ends at "|"
summa_path=$(echo ${summa_path%%#*}) # remove the part starting at '#'; does nothing if no '#' is present

# Specify the default path if needed
if [ "$summa_path" = "default" ]; then
  
 # Get the root path
 root_line=$(grep -m 1 "^root_path" ../0_control_files/control_active.txt)
 root_path=$(echo ${root_line##*|}) 
 root_path=$(echo ${root_path%%#*}) 
 summa_path="${root_path}/installs/Summa-Actors/bin/"
fi


# - Find the SUMMA executable
# ---------------------------
setting_line=$(grep -m 1 "^exe_name_summa" ../0_control_files/control_active.txt) 
summa_exe=$(echo ${setting_line##*|}) 
summa_exe=$(echo ${summa_exe%%#*}) 


# - Find where the SUMMA settings are
# -----------------------------------
setting_line=$(grep -m 1 "^settings_summa_path" ../0_control_files/control_active.txt) 
settings_path=$(echo ${setting_line##*|}) 
settings_path=$(echo ${settings_path%%#*}) 

# Specify the default path if needed
if [ "$settings_path" = "default" ]; then
  
 # Get the root path
 root_line=$(grep -m 1 "^root_path" ../0_control_files/control_active.txt)
 root_path=$(echo ${root_line##*|}) 
 root_path=$(echo ${root_path%%#*}) 
 
 # Get the domain name
 domain_line=$(grep -m 1 "^domain_name" ../0_control_files/control_active.txt)
 domain_name=$(echo ${domain_line##*|}) 
 domain_name=$(echo ${domain_name%%#*})
 
 # Make the default path
 settings_path="${root_path}/domain_${domain_name}/settings/SUMMA/"
fi


# - Find the filemanager name
# ---------------------------
setting_line=$(grep -m 1 "^settings_summa_filemanager" ../0_control_files/control_active.txt) 
filemanager=$(echo ${setting_line##*|}) 
filemanager=$(echo ${filemanager%%#*})


# - Find where the SUMMA logs need to go
# --------------------------------------
setting_line=$(grep -m 1 "^experiment_log_summa" ../0_control_files/control_active.txt) 
summa_log_path=$(echo ${setting_line##*|}) 
summa_log_path=$(echo ${summa_log_path%%#*})
summa_log_name="summa_log_${array_id}.txt"

# Specify the default path if needed
if [ "$summa_log_path" = "default" ]; then
 
 # Get the root path
 root_line=$(grep -m 1 "^root_path" ../0_control_files/control_active.txt)
 root_path=$(echo ${root_line##*|}) 
 root_path=$(echo ${root_path%%#*}) 
 
 # Get the domain name
 domain_line=$(grep -m 1 "^domain_name" ../0_control_files/control_active.txt)
 domain_name=$(echo ${domain_line##*|}) 
 domain_name=$(echo ${domain_name%%#*})
 
 # Get the experiment ID
 exp_line=$(grep -m 1 "^experiment_id" ../0_control_files/control_active.txt)
 exp_name=$(echo ${exp_line##*|}) 
 exp_name=$(echo ${exp_name%%#*})
 
 # Make the default path
 summa_log_path="${root_path}/domain_${domain_name}/simulations/${exp_name}/SUMMA/SUMMA_logs/"
fi


# - Get the SUMMA output path (for code provenance and possibly settings backup)
# ------------------------------------------------------------------------------
summa_out_line=$(grep -m 1 "^settings_summa_path" ../0_control_files/control_active.txt)
summa_out_path=$(echo ${summa_out_line##*|}) 
summa_out_path=$(echo ${summa_out_path%%#*})

# Specify the default path if needed
if [ "$summa_out_path" = "default" ]; then
 
 # Get the root path
 root_line=$(grep -m 1 "^root_path" ../0_control_files/control_active.txt)
 root_path=$(echo ${root_line##*|}) 
 root_path=$(echo ${root_path%%#*}) 
 
 # Get the domain name
 domain_line=$(grep -m 1 "^domain_name" ../0_control_files/control_active.txt)
 domain_name=$(echo ${domain_line##*|}) 
 domain_name=$(echo ${domain_name%%#*})
 
 # Get the experiment ID
 exp_line=$(grep -m 1 "^experiment_id" ../0_control_files/control_active.txt)
 exp_name=$(echo ${exp_line##*|}) 
 exp_name=$(echo ${exp_name%%#*})
 
 # Make the default path
 summa_out_path="${root_path}/domain_${domain_name}/simulations/${exp_name}/SUMMA/"
fi


# - Find if we need to backup the settings and find the path if so
# ----------------------------------------------------------------
setting_line=$(grep -m 1 "^experiment_backup_settings" ../0_control_files/control_active.txt) 
do_backup=$(echo ${setting_line##*|}) 
do_backup=$(echo ${do_backup%%#*}) 

# Specify the path (inside the experiment output folder)
if [ "$do_backup" = "yes" ]; then
 # Make the setting backup path
 backup_path="${summa_out_path}run_settings"
fi


# -- echo settings
echo "summa dir   = ${summa_path}"
echo "summa exe   = ${summa_exe}"
echo "setting dir = ${settings_path}"
echo "filemanager = ${filemanager}"
echo "log dir     = ${summa_log_path}"
echo "output dir  = ${summa_out_path}"
echo "backup dir  = ${backup_path}"


# --- Run
# Do the settings backup if needed
if [ "$do_backup" = "yes" ]; then
 mkdir -p $backup_path
 copy_command="cp -R -n ${settings_path}. ${backup_path}"
 $copy_command
fi

# Run SUMMA
mkdir -p $summa_log_path
summa_command="${summa_path}${summa_exe} -g ${gru_start} ${gru_count} -m ${settings_path}${filemanager} --caf.scheduler.max-threads=$SLURM_CPUS_PER_TASK"
$summa_command > $summa_log_path$summa_log_name 

# --- Code provenance
# Generates a basic log file in the domain folder and copies the control file and itself there.
# Make a log directory if it doesn't exist
log_path="${summa_out_path}/_workflow_log"
mkdir -p $log_path

# Log filename
today=`date '+%F'`
log_file="${today}_SUMMA_run_log_grus_${gru_start}_${gru_count}.txt"

# Make the log
this_file='1_run_summa_as_array.sh'
echo "Log generated by ${this_file} on `date '+%F %H:%M:%S'`"  > $log_path/$log_file # 1st line, store in new file
echo 'Ran SUMMA for ${gru_count} GRUs, starting at GRU ${gru_start}.' >> $log_path/$log_file # 2nd line, append to existing file

# Copy this file to log directory
cp -n $this_file $log_path

done
wait








