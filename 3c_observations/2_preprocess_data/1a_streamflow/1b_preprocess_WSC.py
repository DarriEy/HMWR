import pandas as pd
import sys
from pathlib import Path
import numpy as np
import sys
import csv
from datetime import datetime

sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))
from utils.control_file_utils import read_from_control, make_default_path

# Easy access to control file folder
controlFolder = Path('../../../0_control_files')
controlFile = 'control_active.txt'

# Read necessary paths and settings
domain_name = read_from_control(controlFolder/controlFile, 'domain_name')
forcing_time_step_size = int(read_from_control(controlFolder/controlFile, 'forcing_time_step_size'))

streamflow_raw_path = read_from_control(controlFolder/controlFile, 'streamflow_raw_path')
if streamflow_raw_path == 'default' or streamflow_raw_path is None:
    streamflow_raw_path = make_default_path(controlFolder, controlFile, Path('observations') / 'streamflow' / 'raw_data')
else:
    streamflow_raw_path = Path(streamflow_raw_path)

streamflow_processed_path = read_from_control(controlFolder/controlFile, 'streamflow_processed_path')
if streamflow_processed_path == 'default' or streamflow_processed_path is None:
    streamflow_processed_path = make_default_path(controlFolder, controlFile, Path('observations') / 'streamflow' / 'preprocessed')
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
        return f'{time_step_size}S'

# Read the WSC streamflow data
wsc_data = pd.read_csv(streamflow_raw_path / streamflow_raw_name, 
                       comment='#', 
                       low_memory=False)

# Convert 'ISO 8601 UTC' to datetime and set as index
wsc_data['ISO 8601 UTC'] = pd.to_datetime(wsc_data['ISO 8601 UTC'], format='ISO8601')
wsc_data.set_index('ISO 8601 UTC', inplace=True)

# Convert from UTC to the desired time zone, then make it tz-naive
wsc_data.index = wsc_data.index.tz_convert('America/Edmonton').tz_localize(None)

# Convert discharge to numeric, assuming it's in the 'Value' column and already in m^3/s
wsc_data['discharge_cms'] = pd.to_numeric(wsc_data['Value'], errors='coerce')

# Determine resampling frequency
resample_freq = get_resample_freq(forcing_time_step_size)

# Resample to the model timestep
resampled_data = wsc_data['discharge_cms'].resample(resample_freq).mean()

# Interpolate missing values (optional, remove if not desired)
resampled_data = resampled_data.interpolate(method='time', limit=24)  # Interpolate up to 24 hours of missing data

# Save processed data
output_file = streamflow_processed_path / f'{domain_name}_streamflow_processed.csv'

# Convert the resampled data to a list of tuples
data_to_write = [('datetime', 'discharge_cms')] + list(resampled_data.items())

# Write to CSV using csv module
with open(output_file, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    for row in data_to_write:
        # Format the datetime to string manually
        if isinstance(row[0], datetime):
            formatted_datetime = row[0].strftime('%Y-%m-%d %H:%M:%S')
            csv_writer.writerow([formatted_datetime, row[1]])
        else:
            csv_writer.writerow(row)

print(f"Processed streamflow data saved to: {output_file}")
print(f"Total rows in processed data: {len(resampled_data)}")
print(f"Number of non-null values: {resampled_data.count()}")
print(f"Number of null values: {resampled_data.isnull().sum()}")