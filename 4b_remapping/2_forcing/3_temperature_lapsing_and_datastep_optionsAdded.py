import os
import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path
from shutil import copyfile
from datetime import datetime
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from utils.control_file_utils import read_from_control, make_default_path

# --- Control file handling
controlFolder = Path('../../0_control_files')
controlFile = 'control_active.txt'

# --- Get forcing dataset
forcing_dataset = read_from_control(controlFolder/controlFile, 'forcing_dataset').lower()

# --- Find location of intersection file
intersect_path = read_from_control(controlFolder/controlFile, 'intersect_forcing_path')
intersect_path = Path(intersect_path) if intersect_path != 'default' else make_default_path(controlFolder, controlFile, 'shapefiles/catchment_intersection/with_forcing')

domain = read_from_control(controlFolder/controlFile, 'domain_name')
intersect_name = f"{domain}_{forcing_dataset}_intersected_shapefile.csv"

# --- Find where the EASYMORE-prepared forcing files are
forcing_easymore_path = read_from_control(controlFolder/controlFile, 'forcing_basin_avg_path')
forcing_easymore_path = Path(forcing_easymore_path) if forcing_easymore_path != 'default' else make_default_path(controlFolder, controlFile, f'forcing/3_basin_averaged_data')

# Select files for RDRS dataset
forcing_files = [f for f in os.listdir(forcing_easymore_path) if f.startswith(f"{domain}_{forcing_dataset}_remapped_") and f.endswith('.nc')]
forcing_files.sort()

# --- Find the time step size of the forcing data
data_step = int(read_from_control(controlFolder/controlFile, 'forcing_time_step_size'))

# --- Find where the final forcing needs to go
forcing_summa_path = read_from_control(controlFolder/controlFile, 'forcing_summa_path')
forcing_summa_path = Path(forcing_summa_path) if forcing_summa_path != 'default' else make_default_path(controlFolder, controlFile, f'forcing/4_SUMMA_input')
forcing_summa_path.mkdir(parents=True, exist_ok=True)

# --- Find the area-weighted information for each basin
topo_data = pd.read_csv(intersect_path/intersect_name)

hru_ID_name = read_from_control(controlFolder/controlFile, 'catchment_shp_hruid')
gru_ID_name = read_from_control(controlFolder/controlFile, 'catchment_shp_gruid')

# Specify the column names
# Note that column names are truncated at 10 characters in the ESRI shapefile, but NOT in the .csv we use here
gru_ID         = 'S_1_' + gru_ID_name # EASYMORE prefix + user's hruId name
hru_ID         = 'S_1_' + hru_ID_name # EASYMORE prefix + user's hruId name
forcing_ID     = 'S_2_ID'             # fixed name assigned by EASYMORE
catchment_elev = 'S_1_elev_mean'      # EASYMORE prefix + name used in catchment+DEM intersection step
forcing_elev   = 'S_2_elev_m'         # EASYMORE prefix + name used in ERA5 shapefile generation
weights        = 'weight'             # EASYMORE feature


# Define the lapse rate
lapse_rate = 0.0065 # [K m-1]

# Calculate weighted lapse values for each HRU 
# Note that these lapse values need to be ADDED to ERA5 temperature data
topo_data['lapse_values'] = topo_data[weights] * lapse_rate * (topo_data[forcing_elev] - topo_data[catchment_elev]) # [K]

# Find the total lapse value per basin; i.e. sum the individual contributions of each HRU+ERA5-grid overlapping part
# Account for the special case where gru_ID and hru_ID share the same column and thus name
if gru_ID == hru_ID:
    lapse_values = topo_data.groupby([hru_ID]).lapse_values.sum().reset_index() # Sort by HRU
else:
    lapse_values = topo_data.groupby([gru_ID,hru_ID]).lapse_values.sum().reset_index() # sort by GRU first and HRU second

# Sort and set hruID as the index variable
lapse_values = lapse_values.sort_values(hru_ID).set_index(hru_ID)

del topo_data

# --- Loop over forcing files; apply lapse rates and add data-step variable
# Initiate the loop
for file in forcing_files:
    
    # Progress
    print('Starting on ' + file)
    if os.path.isfile(file):
        print (file + 'already exists ... skipping')
    else:
        # load the data
        with xr.open_dataset(forcing_easymore_path / file) as dat:
        
            # --- Temperature lapse rates
            # Find the lapse rates by matching the HRU order in the forcing file with that in 'lapse_values'
            lapse_values_sorted = lapse_values['lapse_values'].loc[dat['hruId'].values]
        
            # Make a data array of size (nTime,nHru) 
            addThis = xr.DataArray(np.tile(lapse_values_sorted.values, (len(dat['time']),1)), dims=('time','hru')) 
        
            # Get the air temperature variables
            #tmp_longname = dat['airtemp'].long_name
            tmp_units = dat['airtemp'].units    
        
            # Subtract lapse values from existing temperature data
            dat['airtemp'] = dat['airtemp'] + addThis
        
            # Add the attributes back in
            #dat.airtemp.attrs['long_name'] = tmp_longname
            dat.airtemp.attrs['units'] = tmp_units
        
            # --- Time step specification 
            dat['data_step'] = data_step
            dat.data_step.attrs['long_name'] = 'data step length in seconds'
            dat.data_step.attrs['units'] = 's'
        
            # --- Save to file in new location
            dat.to_netcdf(forcing_summa_path/file) 
        
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
            f"Applied temperature lapse rate to {forcing_dataset.upper()} forcing data and added data_step variable."
        ]
        file.writelines(line + '\n' for line in lines)

create_log(forcing_summa_path, '_temperature_lapse_and_datastep.txt', '3_temperature_lapsing_and_datastep.py')

print(f"Completed processing of {forcing_dataset.upper()} forcing files with time-varying temperature lapsing and data step addition.")
