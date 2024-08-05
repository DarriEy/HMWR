import xarray as xr
import numpy as np
from datetime import datetime, timedelta
import os
from pathlib import Path

# Control file handling
controlFolder = Path('../../0_control_files')
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


def process_carra_data(input_file, output_file):
    # Open the CARRA dataset
    ds = xr.open_dataset(input_file)

    # Create a new dataset for processed data
    new_ds = xr.Dataset()

    # Copy dimensions
    new_ds['time'] = ds['time']
    new_ds['point'] = ds['point']

    # Copy and rename coordinates
    new_ds['longitude'] = ds['longitude']
    new_ds['latitude'] = ds['latitude']

    # Process variables
    # Wind speed
    u10 = ds['10u']
    v10 = ds['10v']
    new_ds['windspd'] = xr.DataArray(
        np.sqrt(u10**2 + v10**2),
        dims=['time', 'point'],
        attrs={
            '_FillValue': np.nan,
            'units': 'm s**-1',
            'long_name': 'wind speed at 10m',
            'standard_name': 'wind_speed',
            'missing_value': -999.
        }
    )

    # Air temperature
    new_ds['airtemp'] = ds['2t'].rename('airtemp').assign_attrs({
        '_FillValue': np.nan,
        'units': 'K',
        'long_name': 'Temperature at 2m',
        'standard_name': 'air_temperature',
        'missing_value': -999.
    })

    # Air pressure
    new_ds['airpres'] = ds['sp'].rename('airpres').assign_attrs({
        '_FillValue': np.nan,
        'units': 'Pa',
        'long_name': 'Surface pressure',
        'standard_name': 'surface_air_pressure',
        'missing_value': -999.
    })

    # Specific humidity (now using the direct 2sh variable)
    new_ds['spechum'] = ds['2sh'].rename('spechum').assign_attrs({
        '_FillValue': np.nan,
        'units': 'kg kg**-1',
        'long_name': 'Specific humidity at 2m',
        'standard_name': 'specific_humidity',
        'missing_value': -999.
    })

    # Short-wave radiation (using ssr and converting from J/m^2 to W/m^2)
    #time_step = (ds.time[1] - ds.time[0]).values.astype('timedelta64[s]').astype(int)
    time_step = 3600
    new_ds['SWRadAtm'] = (ds['ssr'] / time_step).rename('SWRadAtm').assign_attrs({
        '_FillValue': np.nan,
        'units': 'W m**-2',
        'long_name': 'Surface solar radiation downwards',
        'standard_name': 'surface_downwelling_shortwave_flux_in_air',
        'missing_value': -999.
    })

    # Long-wave radiation (using str and converting from J/m^2 to W/m^2)
    lw_radiation = ds['strd'] / time_step  

    new_ds['LWRadAtm'] = lw_radiation.rename('LWRadAtm').assign_attrs({
        '_FillValue': np.nan,
        'units': 'W m**-2',
        'long_name': 'Surface thermal radiation downwards',
        'standard_name': 'surface_downwelling_longwave_flux_in_air',
        'missing_value': -999.
    })

    # Precipitation rate
    new_ds['pptrate'] = (ds['tp'] / time_step).rename('pptrate').assign_attrs({
        '_FillValue': np.nan,
        'units': 'kg m**-2 s**-1',
        'long_name': 'Mean total precipitation rate',
        'standard_name': 'precipitation_flux',
        'missing_value': -999.
    })

    # Add global attributes
    new_ds.attrs = {
        'History': f"Created {datetime.now().strftime('%a %b %d %H:%M:%S %Y')}",
        'Language': "Written using Python",
        'Reason': "Processing CARRA data to match format for SUMMA model",
        'Conventions': "CF-1.6",
    }

    # Save the processed dataset
    new_ds.to_netcdf(output_file)
    print(f"Processed data saved to {output_file}")

def process_all_carra_files(input_dir, output_dir):
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Process all .nc files in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith('.nc'):
            input_file = os.path.join(input_dir, filename)
            output_file = os.path.join(output_dir, f"CARRA_processed_{filename}")
            print(f"Processing {filename}...")
            process_carra_data(input_file, output_file)

def main():
    # Read necessary paths and settings
    # Find the raw data path
    forcing_raw_path = read_from_control(controlFolder/controlFile, 'forcing_raw_path')
    if forcing_raw_path == 'default':
        forcing_raw_path = make_default_path('forcing/1b_CARRA_raw_data')
    else:
        forcing_raw_path = Path(forcing_raw_path)

    # Find the raw data path
    forcing_merged_path = read_from_control(controlFolder/controlFile, 'forcing_merged_path')
    if forcing_merged_path == 'default':
        forcing_merged_path =  make_default_path('forcing/2_merged_data')
    else:
        forcing_merged_path = Path(forcing_merged_path)

    #run the processing function 
    process_all_carra_files(forcing_raw_path, forcing_merged_path)

if __name__ == "__main__":
    main()