# Update mizuRoute base settings
# Updates the tscale parameter in param.nml.default with the forcing_time_step_size from control_active.txt

# modules
import os
from pathlib import Path
from datetime import datetime
import fileinput

# --- Control file handling
# Easy access to control file folder
controlFolder = Path('../../../0_control_files')

# Store the name of the 'active' file in a variable
controlFile = 'control_active.txt'

# Function to extract a given setting from the control file
def read_from_control(file, setting):
    with open(file) as contents:
        for line in contents:
            if setting in line and not line.startswith('#'):
                break
    substring = line.split('|', 1)[1]
    substring = substring.split('#', 1)[0]
    substring = substring.strip()
    return substring

# Function to specify a default path
def make_default_path(suffix):
    rootPath = Path(read_from_control(controlFolder/controlFile, 'root_path'))
    domainName = read_from_control(controlFolder/controlFile, 'domain_name')
    domainFolder = 'domain_' + domainName
    defaultPath = rootPath / domainFolder / suffix
    return defaultPath

# --- Find where the settings are located
settings_path = read_from_control(controlFolder/controlFile, 'settings_mizu_path')

# Specify default path if needed
if settings_path == 'default':
    settings_path = make_default_path('settings/mizuRoute')
else:
    settings_path = Path(settings_path)

# --- Get the forcing_time_step_size value
forcing_time_step_size = read_from_control(controlFolder/controlFile, 'forcing_time_step_size')

# --- Update the param.nml.default file
param_file = settings_path / 'param.nml.default'

# Read the file, update the tscale value, and write back
with fileinput.input(files=(param_file), inplace=True) as file:
    for line in file:
        if line.strip().startswith('tscale'):
            print(f"  tscale = {forcing_time_step_size}")
        else:
            print(line, end='')

# --- Code provenance
# Generates a basic log file in the domain folder and copies the control file and itself there.

# Set the log path and file name
logPath = settings_path
log_suffix = '_update_base_settings.txt'

# Create a log folder
logFolder = '_workflow_log'
Path(logPath / logFolder).mkdir(parents=True, exist_ok=True)

# Copy this script
thisFile = '2_update_base_settings.py'
from shutil import copyfile
copyfile(thisFile, logPath / logFolder / thisFile)

# Get current date and time
now = datetime.now()

# Create a log file 
logFile = now.strftime('%Y%m%d') + log_suffix
with open(logPath / logFolder / logFile, 'w') as file:
    lines = [
        f"Log generated by {thisFile} on {now.strftime('%Y/%m/%d %H:%M:%S')}\n",
        f"Updated tscale in param.nml.default to {forcing_time_step_size}."
    ]
    for txt in lines:
        file.write(txt + '\n')

print(f"Updated tscale in {param_file} to {forcing_time_step_size}")