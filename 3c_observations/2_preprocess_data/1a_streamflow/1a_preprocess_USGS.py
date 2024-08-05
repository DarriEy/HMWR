import pandas as pd
import sys
from pathlib import Path
import numpy as np

#####################################
### --- Control file handling --- ###
#####################################

# Easy access to control file folder
controlFolder = Path('../../../0_control_files')

# Store the name of the 'active' file in a variable
controlFile = 'control_active.txt'

def read_from_control(file, setting):
    """Extract a given setting from the control file."""
    with open(file) as contents:
        for line in contents:
            if setting in line and not line.startswith('#'):
                return line.split('|', 1)[1].split('#', 1)[0].strip()
    return None

def make_default_path(suffix):
    """Specify a default path based on the control file settings."""
    rootPath = Path(read_from_control(controlFolder/controlFile, 'root_path'))
    domainName = read_from_control(controlFolder/controlFile, 'domain_name')
    return rootPath / f'domain_{domainName}' / suffix

# Read necessary paths and settings
domain_name = read_from_control(controlFolder/controlFile, 'domain_name')
forcing_time_step_size = int(read_from_control(controlFolder/controlFile, 'forcing_time_step_size'))

streamflow_raw_path = read_from_control(controlFolder/controlFile, 'streamflow_raw_path')
if streamflow_raw_path == 'default' or streamflow_raw_path is None:
    streamflow_raw_path = make_default_path(Path('observations') / 'streamflow' / 'raw_data')
else:
    streamflow_raw_path = Path(streamflow_raw_path)

streamflow_processed_path = read_from_control(controlFolder/controlFile, 'streamflow_processed_path')
if streamflow_processed_path == 'default' or streamflow_processed_path is None:
    streamflow_processed_path = make_default_path(Path('observations') / 'streamflow' / 'preprocessed')
else:
    streamflow_processed_path = Path(streamflow_processed_path)

streamflow_raw_name = read_from_control(controlFolder/controlFile, 'streamflow_raw_name')

# Error checking
if streamflow_raw_name is None:
    raise ValueError("streamflow_raw_name not found in control file")

if not (streamflow_raw_path / streamflow_raw_name).exists():
    raise FileNotFoundError(f"Raw streamflow file not found: {streamflow_raw_path / streamflow_raw_name}")

# Ensure output directory exists
streamflow_processed_path.mkdir(parents=True, exist_ok=True)
def get_resample_freq(time_step_size):
    """Determine appropriate resampling frequency based on time step size."""
    if time_step_size == 3600:
        return 'h'
    elif time_step_size == 86400:
        return 'D'
    else:
        return f'{time_step_size}s'

# Read the USGS streamflow data, skipping the row with '20d'
usgs_data = pd.read_csv(streamflow_raw_path / streamflow_raw_name, 
                        comment='#', sep='\t', 
                        skiprows=[5],  # Skip the 6th row (index 5) which contains '20d'
                        parse_dates=['datetime'],
                        date_format='%Y-%m-%d %H:%M',
                        dtype={'site_no': str, '1770_00060': str})  # Specify dtypes for columns with mixed types

# Clean the discharge data
usgs_data['1770_00060'] = pd.to_numeric(usgs_data['1770_00060'], errors='coerce')

# Convert discharge from cubic feet per second to cubic meters per second
usgs_data['discharge_cms'] = usgs_data['1770_00060'] * 0.028316847

# Ensure datetime is set as the index and is of datetime type
usgs_data['datetime'] = pd.to_datetime(usgs_data['datetime'], format='%Y-%m-%d %H:%M', errors='coerce')
usgs_data = usgs_data.dropna(subset=['datetime'])  # Remove rows with invalid dates
usgs_data.set_index('datetime', inplace=True)

# Determine resampling frequency
resample_freq = get_resample_freq(forcing_time_step_size)

# Resample to the model timestep
resampled_data = usgs_data['discharge_cms'].resample(resample_freq).mean()

# Interpolate missing values (optional, remove if not desired)
resampled_data = resampled_data.interpolate(method='time', limit=24)  # Interpolate up to 24 hours of missing data

# Save processed data
output_file = streamflow_processed_path / f'{domain_name}_streamflow_processed.csv'
resampled_data.to_csv(output_file)

print(f"Processed streamflow data saved to: {output_file}")
print(f"Total rows in processed data: {len(resampled_data)}")
print(f"Number of non-null values: {resampled_data.count()}")
print(f"Number of null values: {resampled_data.isnull().sum()}")