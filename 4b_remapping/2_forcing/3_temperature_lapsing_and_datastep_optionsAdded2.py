# Temperature lapsing and data_step

# modules
import os
import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path
from shutil import copyfile
from datetime import datetime

# --- Control file handling
# Easy access to control file folder
controlFolder = Path('../../0_control_files')

# Store the name of the 'active' file in a variable
controlFile = 'control_active.txt'

# Function to extract a given setting from the control file
def read_from_control(file, setting):
    with open(file) as contents:
        for line in contents:
            if setting in line and not line.startswith('#'):
                break
    substring = line.split('|',1)[1].split('#',1)[0].strip()
    return substring

# Function to specify a default path
def make_default_path(suffix):
    rootPath = Path(read_from_control(controlFolder/controlFile,'root_path'))
    domainName = read_from_control(controlFolder/controlFile,'domain_name')
    domainFolder = 'domain_' + domainName
    defaultPath = rootPath / domainFolder / suffix
    return defaultPath

# --- Get forcing dataset
forcing_dataset = read_from_control(controlFolder/controlFile,'forcing_dataset').lower()

# --- Find location of intersection file
intersect_path = read_from_control(controlFolder/controlFile,'intersect_forcing_path')
if intersect_path == 'default':
    intersect_path = make_default_path('shapefiles/catchment_intersection/with_forcing')
else:
    intersect_path = Path(intersect_path)

# Make the file name
domain = read_from_control(controlFolder/controlFile,'domain_name')
intersect_name = domain + '_intersected_shapefile.csv'

# --- Find where the EASYMORE-prepared forcing files are
forcing_easymore_path = read_from_control(controlFolder/controlFile,'forcing_basin_avg_path')
if forcing_easymore_path == 'default':
    forcing_easymore_path = make_default_path('forcing/3_basin_averaged_data')
else:
    forcing_easymore_path = Path(forcing_easymore_path)

# Find the files
_,_,forcing_files = next(os.walk(forcing_easymore_path))
forcing_files.sort()

# Filter files based on the forcing dataset
if forcing_dataset == 'era5':
    forcing_files = [f for f in forcing_files if f.startswith(f"{domain}_remapped_")]
elif forcing_dataset == 'rdrs':
    forcing_files = [f for f in forcing_files if f.startswith(f"{domain}_rdrs_remapped_")]
else:
    raise ValueError(f"Unknown forcing dataset: {forcing_dataset}")

# --- Find the time step size of the forcing data
data_step = int(read_from_control(controlFolder/controlFile,'forcing_time_step_size'))

# --- Find where the final forcing needs to go
forcing_summa_path = read_from_control(controlFolder/controlFile,'forcing_summa_path')
if forcing_summa_path == 'default':
    forcing_summa_path = make_default_path('forcing/4_SUMMA_input')
else:
    forcing_summa_path = Path(forcing_summa_path)

forcing_summa_path.mkdir(parents=True, exist_ok=True)

# --- Get the correct forcing shape file name
if forcing_dataset == 'era5':
    forcing_shape_name = read_from_control(controlFolder/controlFile,'forcing_shape_name')
elif forcing_dataset == 'rdrs':
    forcing_shape_name = read_from_control(controlFolder/controlFile,'forcing_rdrs_shape_name')

# --- Find the area-weighted lapse value for each basin
topo_data = pd.read_csv(intersect_path/intersect_name)

hru_ID_name = read_from_control(controlFolder/controlFile,'catchment_shp_hruid')
gru_ID_name = read_from_control(controlFolder/controlFile,'catchment_shp_gruid')

gru_ID = 'S_1_' + gru_ID_name
hru_ID = 'S_1_' + hru_ID_name
forcing_ID = 'S_2_ID'
catchment_elev = 'S_1_elev_mean'
forcing_elev = 'S_2_elev_m'
weights = 'weight'

lapse_rate = 0.0065 # [K m-1]

topo_data['lapse_values'] = topo_data[weights] * lapse_rate * (topo_data[forcing_elev] - topo_data[catchment_elev])

if gru_ID == hru_ID:
    lapse_values = topo_data.groupby([hru_ID]).lapse_values.sum().reset_index()
else:
    lapse_values = topo_data.groupby([gru_ID,hru_ID]).lapse_values.sum().reset_index()

lapse_values = lapse_values.sort_values(hru_ID).set_index(hru_ID)

del topo_data

# --- Loop over forcing files; apply lapse rates and add data-step variable
for file in forcing_files:
    print('Starting on ' + file)
    if os.path.isfile(forcing_summa_path / file):
        print(file + ' already exists ... skipping')
    else:
        with xr.open_dataset(forcing_easymore_path / file) as dat:
            # Temperature lapse rates
            lapse_values_sorted = lapse_values['lapse_values'].loc[dat['hruId'].values]
            addThis = xr.DataArray(np.tile(lapse_values_sorted.values, (len(dat['time']),1)), dims=('time','hru'))
            
            tmp_longname = dat['airtemp'].attrs.get('long_name', 'air temperature')
            tmp_units = dat['airtemp'].attrs.get('units', 'K')
            
            dat['airtemp'] = dat['airtemp'] + addThis
            
            dat['airtemp'].attrs['long_name'] = tmp_longname
            dat['airtemp'].attrs['units'] = tmp_units
            
            # Time step specification 
            dat['data_step'] = data_step
            dat['data_step'].attrs['long_name'] = 'data step length in seconds'
            dat['data_step'].attrs['units'] = 's'
            
            # Save to file in new location
            dat.to_netcdf(forcing_summa_path / file)

# --- Code provenance
logPath = forcing_summa_path
log_suffix = '_temperature_lapse_and_datastep.txt'

logFolder = '_workflow_log'
Path(logPath / logFolder).mkdir(parents=True, exist_ok=True)

thisFile = '3_temperature_lapsing_and_datastep.py'
copyfile(thisFile, logPath / logFolder / thisFile)

now = datetime.now()

logFile = now.strftime('%Y%m%d') + log_suffix
with open(logPath / logFolder / logFile, 'w') as file:
    lines = ['Log generated by ' + thisFile + ' on ' + now.strftime('%Y/%m/%d %H:%M:%S') + '\n',
             f'Applied temperature lapse rate to {forcing_dataset.upper()} forcing data and added data_step variable.']
    for txt in lines:
        file.write(txt)