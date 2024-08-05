import itertools
import os
import sys
import csv
import subprocess
import pandas as pd
import xarray as xr
from pathlib import Path
import matplotlib.pyplot as plt
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))
from calculate_sim_stats import get_KGE, get_KGEp, get_NSE, get_MAE, get_RMSE

#####################################
### --- Control file handling --- ###
#####################################

# Easy access to control file folder
controlFolder = Path('../../0_control_files')

# Store the name of the 'active' file in a variable
controlFile = 'control_active.txt'

# Function to extract a given setting from the control file
def read_from_control( file, setting ):
    
    # Open 'control_active.txt' and ...
    with open(file) as contents:
        for line in contents:
            
            # ... find the line with the requested setting
            if setting in line and not line.startswith('#'):
                break
    
    # Extract the setting's value
    substring = line.split('|',1)[1]      # Remove the setting's name (split into 2 based on '|', keep only 2nd part)
    substring = substring.split('#',1)[0] # Remove comments, does nothing if no '#' is found
    substring = substring.strip()         # Remove leading and trailing whitespace, tabs, newlines
       
    # Return this value    
    return substring

# Function to specify a default path
def make_default_path(suffix):
    
    # Get the root path
    rootPath = Path(read_from_control(controlFolder/controlFile,'root_path'))
    
    # Get the domain folder
    domainName = read_from_control(controlFolder/controlFile,'domain_name')
    domainFolder = 'domain_' + domainName
    
    # Specify the forcing path
    defaultPath = rootPath / domainFolder / suffix
    
    return defaultPath

# --- Find where the modelDecisions.txt file is 
model_decisions_path = read_from_control(controlFolder/controlFile,'model_decisions_path')

# Specify the default paths if required 
if model_decisions_path == 'default':
    
    model_decisions_path = make_default_path('settings/SUMMA/modelDecisions.txt') # outputs a Path()
else:
    model_decisions_path = Path(model_decisions_path) # make sure a user-specified path is a Path()

# --- Find where the modelDecisions.txt file is 
model_decisions_comparison = read_from_control(controlFolder/controlFile,'model_decisions_comparison')
rootPath = read_from_control(controlFolder/controlFile,'root_path')
domainName = read_from_control(controlFolder/controlFile,'domain_name')

# Specify the default paths if required 
if model_decisions_comparison == 'default':
    model_decisions_comparison_folder = rootPath + '/domain_' + domainName + '/optimisation/'
    model_decisions_comparison = model_decisions_comparison_folder + 'model_decisions_comparison.csv'
else:
    model_decisions_comparison = model_decisions_comparison

Path(model_decisions_comparison_folder).mkdir(parents=True, exist_ok=True)


# --- Find where the model run folder is 
rootCodePath = read_from_control(controlFolder/controlFile,'root_code_path')
model_run_folder = read_from_control(controlFolder/controlFile,'model_run_folder')

# Specify the default paths if required 
if model_run_folder == 'default':
    model_run_folder = rootCodePath + '/6_model_runs' # outputs a Path()
else:
    model_run_folder = model_run_folder # make sure a user-specified path is a Path()

#Specify SUMMA and mizuRoute run Paths
summa_run_command = read_from_control(controlFolder/controlFile,'summa_run_command')
mizuRoute_run_command = read_from_control(controlFolder/controlFile,'mizuRoute_run_command')

summa_run_path =  model_run_folder + '/' + summa_run_command
mizuRoute_run_path =  model_run_folder + '/' + mizuRoute_run_command

#####################################
###### ---    Functions    --- ######
#####################################

#Function to generate the model decision combinations to consider
def generate_combinations(decision_options):
    return list(itertools.product(*decision_options.values()))

#Function to update the modelDecisions.txt file
def update_model_decisions(combination, decision_keys, input_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    option_map = dict(zip(decision_keys, combination))
    
    for i, line in enumerate(lines):
        for option, value in option_map.items():
            if line.strip().startswith(option):
                lines[i] = f"{option.ljust(30)} {value.ljust(15)} ! {line.split('!')[-1].strip()}\n"
    
    with open(input_file, 'w') as f:
        f.writelines(lines)

# Define the decisions you want to try
decision_options = {
    'snowLayers': ['CLM_2010', 'jrdn1991'],
    'thCondSnow': ['tyen1965', 'melr1977', 'jrdn1991', 'smnv2000'],
    'f_Richards': ['mixdform'],#'moisture'],
    'groundwatr': ['qTopmodl', 'bigBuckt', 'noXplict'],
    'bcLowrTdyn': ['zeroFlux','presTemp'],
    'bcLowrSoiH': ['zeroFlux', 'drainage', 'presHead', 'bottmPsi'],
    #'thCondSoil': ['funcSoilWet', 'mixConstit', 'hanssonVZJ'],
    #'canopySrad': ['noah_mp', 'CLM_2stream', 'UEB_2stream', 'NL_scatter', 'BeersLaw'],
    'alb_method': ['conDecay', 'varDecay'],
    #'windPrfile': ['exponential', 'logBelowCanopy'],
    'astability': ['standard']#, 'louisinv', 'mahrtexp']
}

###################################################
### --- Main: Generate and run combinations --- ###
###################################################

def main():
    # Generate all combinations
    combinations = generate_combinations(decision_options)

    # Create a master file to log all decisions
    master_file = model_decisions_comparison
    with open(master_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Iteration'] + list(decision_options.keys()) + list(['kge', 'kgep', 'nse', 'mae', 'rmse']))

    # Update the model decisions file for each combination -- need to make sure to skip inconsistent combinations
    for i, combination in enumerate(combinations, 1):
        update_model_decisions(combination, list(decision_options.keys()), model_decisions_path)
        #print(f"Updated modelDecisions.txt for iteration {i}")
        
        obs_file_path = read_from_control(controlFolder/controlFile,'obs_file_path') 
        sim_file_path = read_from_control(controlFolder/controlFile,'sim_file_path')
        sim_reach_ID = read_from_control(controlFolder/controlFile,'sim_reach_ID')
        summa_sim_path = rootPath + '/domain_' + domainName + '/simulations/' + read_from_control(controlFolder/controlFile,'experiment_id') + '/SUMMA/' + read_from_control(controlFolder/controlFile,'experiment_id') + '_timestep.nc'

        #Run SUMMA and mizuRoute
        subprocess.run(summa_run_path, shell=True, cwd=model_run_folder, capture_output=True, text=True)
        subprocess.run(mizuRoute_run_path, shell=True, cwd=model_run_folder, capture_output=True, text=True)

        #Calculate the performance metrics
        #Extract timeseries of observation and simulations    
        try:
            dfObs = pd.read_csv(obs_file_path, index_col= 'datetime', parse_dates=True)
            dfObs = dfObs['Discharge'].resample('h').mean()

            dfSim = xr.open_dataset(sim_file_path, engine = 'netcdf4')  
            segment_index = dfSim['reachID'].values == int(sim_reach_ID)
            dfSim = dfSim.sel(seg=segment_index) 
            dfSim = dfSim['IRFroutedRunoff'].to_dataframe().reset_index()
            dfSim.set_index('time', inplace = True)

            dfObs = dfObs.reindex(dfSim.index).dropna()
            dfSim = dfSim.reindex(dfObs.index).dropna()

            obs = dfObs.values
            sim = dfSim['IRFroutedRunoff'].values
            kge = get_KGE(obs, sim, transfo = 1)
            kgep = get_KGEp(obs, sim, transfo = 1)
            nse = get_NSE(obs,sim, transfo = 1)
            mae = get_MAE(obs,sim, transfo = 1)
            rmse = get_RMSE(obs,sim, transfo = 1)

            print(kge, kgep, nse, mae, rmse)

            # Log the decisions in the master file
            with open(master_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([i] + list(combination) + list([kge, kgep, nse, mae, rmse]))

            os.remove(sim_file_path)
            os.remove(summa_sim_path)
        
        except:
            # Log the erronous decisions in the master file
            with open(master_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([i] + list(combination) + list(['erronous combination']))

if __name__ == "__main__":
    main()