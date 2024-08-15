#!/bin/bash


# Add the directory containing ostrich_util.py to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:/Users/darrieythorsson/compHydro/code/CWARHM/7_model_optimization/2_model_calibration/

# Set the path to the control file
control_file="/Users/darrieythorsson/compHydro/code/CWARHM/0_control_files/control_active.txt"

# Function to extract a setting from the control file
read_from_control () {
    grep -m 1 "^$2" "$1" | cut -d '|' -f 2 | sed 's/^[ \t]*//;s/[ \t]*$//' | cut -d '#' -f 1
}

# Read necessary paths and settings
root_path=$(read_from_control "$control_file" "root_path")
domain_name=$(read_from_control "$control_file" "domain_name")
experiment_id=$(read_from_control "$control_file" "experiment_id")


# Set up Python environment if necessary
# Uncomment and modify the following line if you're using a virtual environment
source /Users/darrieythorsson/compHydro/data/CAMELS_spat/camels-spat-env/bin/activate

# Run the Python script to update parameters and run the model
python run_trial.py

exit 0