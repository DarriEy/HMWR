#!/bin/bash
#SBATCH --cpus-per-task=32
#SBATCH --time=10:00:00
#SBATCH --mem=200
#SBATCH --constraint=broadwell
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --job-name=BatchPython
#SBATCH --output=%x-%j.out
#SBATCH --account=def-mclark
#SBATCH --mail-type=ALL
#SBATCH --mail-user=darri.eythorsson@ucalgary.ca

# Script to run python script as array job
# Reads most required info from 'summaWorkflow_public/0_control_files/control_active.txt'


# --- Command line arguments

module load gcc/9.3.0
module load netcdf-fortran
module load openblas
module load caf

python SUMMA_concat_split_summa.py '/project/6079554/darri/data/CWARHM_data/domain_Yukon/simulations/run_Yukon_Merit_2/SUMMA/' '*_day.nc' 'Run_Yukon_Merit_2_day.nc'


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
 # settings_path="${root_path}/domain_${domain_name}/settings/SUMMA/"
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
 #summa_log_path="${root_path}/domain_${domain_name}/simulations/${exp_name}/SUMMA/SUMMA_logs/"
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
 #summa_out_path="${root_path}/domain_${domain_name}/simulations/${exp_name}/SUMMA/"
fi


# - Find if we need to backup the settings and find the path if s
# -- echo settings
echo "summa dir   = ${summa_path}"
echo "summa exe   = ${summa_exe}"
echo "setting dir = ${settings_path}"
echo "filemanager = ${filemanager}"
echo "log dir     = ${summa_log_path}"
echo "output dir  = ${summa_out_path}"
echo "backup dir  = ${backup_path}"







