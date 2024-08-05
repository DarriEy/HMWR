import os
from pathlib import Path
from shutil import copyfile
from datetime import datetime

# --- Control file handling
controlFolder = Path('../../../0_control_files')
controlFile = 'control_active.txt'

def read_from_control(file, setting):
    with open(file) as contents:
        for line in contents:
            if setting in line and not line.startswith('#'):
                substring = line.split('|', 1)[1].split('#', 1)[0].strip()
                return substring
    return None

def make_default_path(suffix):
    rootPath = Path(read_from_control(controlFolder/controlFile, 'root_path'))
    domainName = read_from_control(controlFolder/controlFile, 'domain_name')
    domainFolder = 'domain_' + domainName
    return rootPath / domainFolder / suffix

# --- Get forcing dataset and domain name
forcing_dataset = read_from_control(controlFolder/controlFile, 'forcing_dataset').lower()
domain_name = read_from_control(controlFolder/controlFile, 'domain_name')

# --- Find forcing location
forcing_path = read_from_control(controlFolder/controlFile, 'forcing_summa_path')
forcing_path = Path(forcing_path) if forcing_path != 'default' else make_default_path('forcing/4_SUMMA_input')

# --- Find where forcing file list needs to go
file_list_path = read_from_control(controlFolder/controlFile, 'settings_summa_path')
file_list_name = read_from_control(controlFolder/controlFile, 'settings_summa_forcing_list')
file_list_path = Path(file_list_path) if file_list_path != 'default' else make_default_path('settings/SUMMA')
file_list_path.mkdir(parents=True, exist_ok=True)

# --- Make the file
# Find a list of forcing files based on the dataset
if forcing_dataset == 'carra':
    forcing_files = [f for f in os.listdir(forcing_path) if f.startswith(f"{domain_name}_{forcing_dataset}_remapped_") and f.endswith('.nc')]
elif forcing_dataset == 'era5':
    forcing_files = [f for f in os.listdir(forcing_path) if f.startswith(f"{domain_name}_remapped_") and f.endswith('.nc')]
elif forcing_dataset == 'rdrs':
    forcing_files = [f for f in os.listdir(forcing_path) if f.startswith(f"{domain_name}_{forcing_dataset}_remapped_") and f.endswith('.nc')]
else:
    raise ValueError(f"Unsupported forcing dataset: {forcing_dataset}")

# Sort this list
forcing_files.sort()

# Create the file list
with open(file_list_path / file_list_name, 'w') as f:
    for file in forcing_files:
        f.write(str(file) + "\n")

# --- Code provenance
def create_log(path, suffix, script_name):
    log_folder = path / '_workflow_log'
    log_folder.mkdir(parents=True, exist_ok=True)
    copyfile(script_name, log_folder / script_name)
    
    now = datetime.now()
    log_file = now.strftime('%Y%m%d') + suffix
    with open(log_folder / log_file, 'w') as file:
        lines = [
            f"Log generated by {script_name} on {now.strftime('%Y/%m/%d %H:%M:%S')}\n",
            f"Generated forcing file list for {forcing_dataset.upper()} dataset."
        ]
        file.writelines(line + '\n' for line in lines)

create_log(file_list_path, '_make_forcing_file_list.txt', '1_create_forcing_file_list.py')

print(f"Completed creating forcing file list for {forcing_dataset.upper()} dataset.")