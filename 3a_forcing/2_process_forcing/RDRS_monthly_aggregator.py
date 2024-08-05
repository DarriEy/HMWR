import xarray as xr
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import os
import sys
import pandas as pd
import time

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from utils.control_file_utils import read_from_control, make_default_path

# Control file handling
controlFolder = Path('../../0_control_files')
controlFile = 'control_active.txt'

def construct_file_name_pattern(domain_name):
    return f"{domain_name}_rdrsv2.1_*.nc"

# Raw data path
rawPath = read_from_control(controlFolder/controlFile, 'forcing_rdrs_raw_path')
if rawPath == 'default':
    rawPath = make_default_path(controlFolder, controlFile,'forcing/1_RDRS_raw_data')
else:
    rawPath = Path(rawPath)

# Destination path for merged data
mergePath = read_from_control(controlFolder/controlFile, 'forcing_merged_path')
if mergePath == 'default':
    mergePath = make_default_path(controlFolder, controlFile,'forcing/2_merged_data')
else:
    mergePath = Path(mergePath)

mergePath.mkdir(parents=True, exist_ok=True)

variable_mapping = {
    'RDRS_v2.1_P_FI_SFC': 'LWRadAtm',
    'RDRS_v2.1_P_FB_SFC': 'SWRadAtm',
    'RDRS_v2.1_A_PR0_SFC': 'pptrate',
    'RDRS_v2.1_P_P0_SFC': 'airpres',
    'RDRS_v2.1_P_TT_1.5m': 'airtemp',
    'RDRS_v2.1_P_HU_1.5m': 'spechum',
    'RDRS_v2.1_P_UVC_10m': 'windspd',
    'RDRS_v2.1_P_UUC_10m': 'windspd_u',
    'RDRS_v2.1_P_VVC_10m': 'windspd_v',
    'RDRS_v2.1_P_GZ_SFC': 'geopotential'
}

def print_dataset_info(ds):
    print("Variables in the dataset:")
    for var in ds.variables:
        print(f"- {var}")

def process_rdrs_data(ds):
    # Create a mapping of only the variables that exist in the dataset
    existing_vars = {old: new for old, new in variable_mapping.items() if old in ds.variables}
    
    # Rename only the variables that exist
    ds = ds.rename(existing_vars)
    
    # Convert units and add unit attributes
    if 'airpres' in ds:
        ds['airpres'] = ds['airpres'] * 100  # Convert from mb to Pa
        ds['airpres'].attrs['units'] = 'Pa'
        ds['airpres'].attrs['long_name'] = 'air pressure'
        ds['airpres'].attrs['standard_name'] = 'air_pressure'
    
    if 'airtemp' in ds:
        ds['airtemp'] = ds['airtemp'] + 273.15  # Convert from deg_C to K
        ds['airtemp'].attrs['units'] = 'K'
        ds['airtemp'].attrs['long_name'] = 'air temperature'
        ds['airtemp'].attrs['standard_name'] = 'air_temperature'
    
    if 'pptrate' in ds:
        ds['pptrate'] = ds['pptrate'] / 3600  # Convert from m/hour to m/s
        ds['pptrate'].attrs['units'] = 'm s-1'
        ds['pptrate'].attrs['long_name'] = 'precipitation rate'
        ds['pptrate'].attrs['standard_name'] = 'precipitation_rate'
    
    if 'windspd' in ds:
        ds['windspd'] = ds['windspd'] * 0.514444  # Convert from knots to m/s
        ds['windspd'].attrs['units'] = 'm s-1'
        ds['windspd'].attrs['long_name'] = 'wind speed'
        ds['windspd'].attrs['standard_name'] = 'wind_speed'
    
    if 'LWRadAtm' in ds:
        ds['LWRadAtm'].attrs['long_name'] = 'downward longwave radiation at the surface'
        ds['LWRadAtm'].attrs['standard_name'] = 'surface_downwelling_longwave_flux_in_air'
    
    if 'SWRadAtm' in ds:
        ds['SWRadAtm'].attrs['long_name'] = 'downward shortwave radiation at the surface'
        ds['SWRadAtm'].attrs['standard_name'] = 'surface_downwelling_shortwave_flux_in_air'
    
    if 'spechum' in ds:
        ds['spechum'].attrs['long_name'] = 'specific humidity'
        ds['spechum'].attrs['standard_name'] = 'specific_humidity'
    
    return ds

def combine_to_monthly(input_folder, output_folder, years, file_name_pattern):
    output_folder.mkdir(parents=True, exist_ok=True)

    for year in years:
        print(f"Processing year {year}")
        year_folder = input_folder / str(year)
        for month in range(1, 13):
            print(f"Processing {year}-{month:02d}")
            
            daily_files = sorted(year_folder.glob(file_name_pattern.replace('*', f'{year}{month:02d}*')))
            
            if not daily_files:
                print(f"No files found for {year}-{month:02d}")
                continue
            
            datasets = []
            for file in daily_files:
                try:
                    ds = xr.open_dataset(file)
                    datasets.append(ds)
                except Exception as e:
                    print(f"Error opening file {file}: {str(e)}")

            if not datasets:
                print(f"No valid datasets for {year}-{month:02d}")
                continue

            processed_datasets = []
            for ds in datasets:
                try:
                    processed_ds = process_rdrs_data(ds)
                    processed_datasets.append(processed_ds)
                except Exception as e:
                    print(f"Error processing dataset: {str(e)}")

            if not processed_datasets:
                print(f"No processed datasets for {year}-{month:02d}")
                continue

            monthly_data = xr.concat(processed_datasets, dim="time")
            monthly_data = monthly_data.sortby("time")

            start_time = pd.Timestamp(year, month, 1)
            if month == 12:
                end_time = pd.Timestamp(year + 1, 1, 1) - pd.Timedelta(hours=1)
            else:
                end_time = pd.Timestamp(year, month + 1, 1) - pd.Timedelta(hours=1)

            monthly_data = monthly_data.sel(time=slice(start_time, end_time))

            expected_times = pd.date_range(start=start_time, end=end_time, freq='h')
            monthly_data = monthly_data.reindex(time=expected_times, method='nearest')

            monthly_data['time'].encoding['units'] = 'hours since 1900-01-01'
            monthly_data['time'].encoding['calendar'] = 'gregorian'

            # Add general attributes
            monthly_data.attrs['History'] = f'Created {time.ctime(time.time())}'
            monthly_data.attrs['Language'] = 'Written using Python'
            monthly_data.attrs['Reason'] = 'RDRS data aggregated to monthly files and variables renamed for SUMMA compatibility'

            # Set missing value attribute for all variables
            for var in monthly_data.data_vars:
                monthly_data[var].attrs['missing_value'] = -999

            output_file = output_folder / f"RDRS_monthly_{year}{month:02d}.nc"
            monthly_data.to_netcdf(output_file)

            for ds in datasets:
                ds.close()

def main():
    years = read_from_control(controlFolder/controlFile, 'forcing_raw_time')
    years = [int(year) for year in years.split(',')]
    years = range(years[0], years[1]+1)

    domain_name = read_from_control(controlFolder/controlFile, 'domain_name')
    file_name_pattern = construct_file_name_pattern(domain_name)
    
    combine_to_monthly(rawPath, mergePath, years, file_name_pattern)

    print("Processing complete.")

    # Code provenance
    logFolder = '_workflow_log'
    Path(mergePath / logFolder).mkdir(parents=True, exist_ok=True)

    thisFile = 'RDRS_monthly_aggregator.py'
    from shutil import copyfile
    copyfile(thisFile, mergePath / logFolder / thisFile)

    now = datetime.now()
    logFile = now.strftime('%Y%m%d') + '_RDRS_monthly_aggregator_log.txt'
    with open(mergePath / logFolder / logFile, 'w') as file:
        lines = [
            f'Log generated by {thisFile} on {now.strftime("%Y/%m/%d %H:%M:%S")}\n',
            'Aggregated RDRS data into monthly files and set appropriate attributes.'
        ]
        for txt in lines:
            file.write(txt)

if __name__ == "__main__":
    main()