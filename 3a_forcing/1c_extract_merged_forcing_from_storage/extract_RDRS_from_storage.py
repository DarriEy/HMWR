import xarray as xr
import numpy as np
from pathlib import Path
import os
from datetime import datetime, timedelta

# Control file handling functions 
def read_from_control(file, setting):
    with open(file) as contents:
        for line in contents:
            if setting in line and not line.startswith('#'):
                break
    substring = line.split('|', 1)[1].split('#', 1)[0].strip()
    return substring

def make_default_path(suffix):
    rootPath = Path(read_from_control(controlFolder/controlFile, 'root_path'))
    domainName = read_from_control(controlFolder/controlFile, 'domain_name')
    domainFolder = 'domain_' + domainName
    defaultPath = rootPath / domainFolder / suffix
    return defaultPath

# Set up paths and coordinates
controlFolder = Path('../../0_control_files')
controlFile = 'control_Goodpaster_Merit.txt'

# Source path for RDRS v2.1 data
sourcePath = Path('/project/6079554/data/meteorological-data/rdrsv2.1')  

# Destination path for merged data
mergePath = read_from_control(controlFolder/controlFile, 'forcing_rdrs_raw_path')
if mergePath == 'default':
    mergePath = make_default_path('forcing/1b_RDRS_raw_data')
else:
    mergePath = Path(mergePath)

mergePath.mkdir(parents=True, exist_ok=True)

# Variables to extract (matching ERA5 variables)
variables_to_extract = ['LWRadAtm', 'SWRadAtm', 'pptrate', 'airpres', 'airtemp', 'spechum', 'windspd', 'windspd_u', 'windspd_v']

# Variable mapping
variable_mapping = {
    'LWRadAtm': 'RDRS_v2.1_P_FI_SFC',
    'SWRadAtm': 'RDRS_v2.1_P_FB_SFC',
    'pptrate': 'RDRS_v2.1_A_PR0_SFC',
    'airpres': 'RDRS_v2.1_P_P0_SFC',
    'airtemp': 'RDRS_v2.1_P_TT_1.5m',
    'spechum': 'RDRS_v2.1_P_HU_1.5m',
    'windspd': 'RDRS_v2.1_P_UVC_10m',
    'windspd_u': 'RDRS_v2.1_P_UUC_10m',
    'windspd_v': 'RDRS_v2.1_P_VVC_10m'
}

# Get coordinates for subsetting
coords = read_from_control(controlFolder/controlFile, 'forcing_rdrs_raw_space').split('/')
lat_max, lon_min, lat_min, lon_max = map(float, coords)


def process_file(file_path, output_path):
    ds = xr.open_dataset(file_path)
    
    # Define the exact bounds we want
    lat_min_target, lat_max_target = lat_min, lat_max
    lon_min_target, lon_max_target = lon_min, lon_max
    
    #print(f"Original data extent:")
    #print(f"Latitude range: {ds.lat.min().values} to {ds.lat.max().values}")
    #print(f"Longitude range: {ds.lon.min().values} to {ds.lon.values.max()}")

    # Create a boolean mask for the region of interest
    lat_mask = (ds.lat >= lat_min_target) & (ds.lat <= lat_max_target)
    lon_mask = (ds.lon >= lon_min_target) & (ds.lon <= lon_max_target)
    combined_mask = lat_mask & lon_mask

    # Subset the data using the mask
    ds_sub = ds.where(combined_mask, drop=True)
    
    # Check if the subset is empty
    if ds_sub.sizes['rlat'] == 0 or ds_sub.sizes['rlon'] == 0:
        print(f"Warning: Empty subset for {file_path.name}. Skipping this file.")
        return False

    #print(f"\nSubsetted data extent:")
    #print(f"Latitude range: {ds_sub.lat.min().values} to {ds_sub.lat.max().values}")
    #print(f"Longitude range: {ds_sub.lon.min().values} to {ds_sub.lon.max().values}")
    #print(f"Grid size: {ds_sub.sizes['rlat']} x {ds_sub.sizes['rlon']}")

    # Select and rename variables
    variables_to_keep = list(variable_mapping.values())
    ds_sub = ds_sub[variables_to_keep]
    
    # Rename variables to match ERA5 names
    ds_sub = ds_sub.rename({v: k for k, v in variable_mapping.items()})
    
    # Convert units and add unit attributes
    ds_sub['airpres'] = ds_sub['airpres'] * 100  # Convert from mb to Pa
    ds_sub['airpres'].attrs['units'] = 'Pa'
    
    ds_sub['airtemp'] = ds_sub['airtemp'] + 273.15  # Convert from deg_C to K
    ds_sub['airtemp'].attrs['units'] = 'K'
    
    ds_sub['pptrate'] = ds_sub['pptrate'] / 3600  # Convert from m/hour to m/s
    ds_sub['pptrate'].attrs['units'] = 'm s-1'
    
    ds_sub['windspd'] = ds_sub['windspd'] * 0.514444  # Convert from knots to m/s
    ds_sub['windspd'].attrs['units'] = 'm s-1'
    
    # Ensure all variables have units specified
    for var in ds_sub.data_vars:
        if 'units' not in ds_sub[var].attrs:
            print(f"Warning: No units specified for {var}")
    
    # Add CF-compliant attributes for QGIS
    ds_sub.lat.attrs['units'] = 'degrees_north'
    ds_sub.lat.attrs['standard_name'] = 'latitude'
    ds_sub.lon.attrs['units'] = 'degrees_east'
    ds_sub.lon.attrs['standard_name'] = 'longitude'
    
    # Add global attributes for QGIS and CRS information
    ds_sub.attrs['Conventions'] = 'CF-1.6'
    ds_sub.attrs['crs'] = 'EPSG:4326'
    
    # Save the subsetted data
    ds_sub.to_netcdf(output_path)
    print(f'Finished subsetting {file_path.name}')
    return True

# Main processing loop
total_files = 0
processed_files = 0

start_year = 2009  # Adjust as needed
end_year = 2018    # Adjust as needed

for year in range(start_year, end_year + 1):
    year_path = sourcePath / str(year)
    if not year_path.exists():
        print(f"Path for year {year} does not exist. Skipping.")
        continue
    
    for file in year_path.glob('*.nc'):
        total_files += 1
        output_file = mergePath / f"RDRS_merged_{file.name}"
        if output_file.exists():
            print(f"{output_file.name} already exists ... skipping")
            processed_files += 1
        else:
            if process_file(file, output_file):
                processed_files += 1

print(f"Processing complete. Processed {processed_files} out of {total_files} files.")