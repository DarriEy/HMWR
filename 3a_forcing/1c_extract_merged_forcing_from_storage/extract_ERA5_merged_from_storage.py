## Code to extract ERA5 data from storage and subset to domain instead of downloading from copernicus
# coding: utf-8

# modules
from datetime import datetime
from shutil import copyfile
from pathlib import Path
import xarray as xr
import netCDF4 as nc4
import numpy as np
import time
import sys
import os

# --- Control file handling
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
    rootPath = Path( read_from_control(controlFolder/controlFile,'root_path') )
    
    # Get the domain folder
    domainName = read_from_control(controlFolder/controlFile,'domain_name')
    domainFolder = 'domain_' + domainName
    
    # Specify the forcing path
    defaultPath = rootPath / domainFolder / suffix
    
    return defaultPath

# --- Find source and destination paths
# Find the path where the raw forcing is
# Immediately store as a 'Path' to avoid issues with '/' and '\' on different operating systems
forcingPath = Path('/project/6079554/data/meteorological-data/era5/')

# Find the path where the merged forcing needs to go
mergePath = read_from_control(controlFolder/controlFile,'forcing_merged_path')

# Find the coordinates of the subset domain
coords = read_from_control(controlFolder/controlFile,'forcing_raw_space').split('/')

max_lat = np.float64(coords[0])
min_lon = np.float64(coords[1])
max_lon = np.float64(coords[3])
min_lat = np.float64(coords[2])

if mergePath == 'default':
    mergePath = make_default_path('forcing/2_merged_data')
else: 
    mergePath = Path(mergePath) # ensure Path() object 
    
# Make the merge folder if it doesn't exist
mergePath.mkdir(parents=True, exist_ok=True)

#Loop over all the files inthe directory
for file in os.listdir(forcingPath):
    # check only text files
    if file.endswith('.nc'):
        if os.path.isfile(mergePath/file):
          print (file + 'already exists ... skipping')
        else:
            ds = xr.open_dataset(forcingPath/file)
            ds_sub = ds.sel(latitude = slice(max_lat,min_lat), longitude = slice(min_lon,max_lon))
            save_path = mergePath/file        
            #print(ds_sub)
            ds_sub.to_netcdf(mergePath/file)
            print('Finished subsetting ' + file)
        

