import pandas as pd
import geopandas as gpd
from pathlib import Path
import numpy as np
from shapely.geometry import Point

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

snow_raw_path = read_from_control(controlFolder/controlFile, 'snow_raw_path')
if snow_raw_path == 'default' or snow_raw_path is None:
    snow_raw_path = make_default_path(Path('observations') / 'Snow' / 'raw_data')
else:
    snow_raw_path = Path(snow_raw_path)

snow_processed_path = read_from_control(controlFolder/controlFile, 'snow_processed_path')
if snow_processed_path == 'default' or snow_processed_path is None:
    snow_processed_path = make_default_path(Path('observations') / 'Snow' / 'preprocessed')
else:
    snow_processed_path = Path(snow_processed_path)

catchment_shp_path = read_from_control(controlFolder/controlFile, 'catchment_shp_path')
if catchment_shp_path == 'default' or catchment_shp_path is None:
    catchment_shp_path = make_default_path('shapefiles/catchment')
else:
    catchment_shp_path = Path(catchment_shp_path)

catchment_shp_name = read_from_control(controlFolder/controlFile, 'catchment_shp_name')

# Ensure output directory exists
snow_processed_path.mkdir(parents=True, exist_ok=True)

def get_resample_freq(time_step_size):
    """Determine appropriate resampling frequency based on time step size."""
    if time_step_size == 3600:
        return 'H'
    elif time_step_size == 86400:
        return 'D'
    else:
        return f'{time_step_size}S'

# Read SNOTEL locations
snotel_locations_path = make_default_path(Path('shapefiles') / 'observations' / f'NRCS_SNOTEL_Locations_{read_from_control(controlFolder/controlFile, 'domain_name')}.shp')

snotel_locations = gpd.read_file(snotel_locations_path)
# Read catchment shapefile
catchment = gpd.read_file(catchment_shp_path / catchment_shp_name)

# Process each SNOTEL file
for file in snow_raw_path.glob('SNOTEL_Site_*.txt'):
    # Extract site number from filename
    site_number = file.stem.split('_')[-1]
    
    # Read SNOTEL data
    snotel_data = pd.read_csv(file, comment='#', parse_dates=['Date'])
    
    # Extract SWE data
    swe_data = snotel_data[['Date', 'Snow Water Equivalent (in) Start of Day Values']]
    swe_data.columns = ['Date', 'SWE']
    swe_data.set_index('Date', inplace=True)
    
    # Convert SWE from inches to meters
    swe_data = swe_data.assign(SWE=swe_data['SWE'] * 0.0254)
    
    # Remove rows with no observations
    # swe_data = swe_data.dropna()
    
    # Find HRU for this SNOTEL site
    site_location = snotel_locations[snotel_locations['site_name'].str.extract(r'\((\d+)\)')[0] == site_number]
    if not site_location.empty:
        point = site_location.geometry.iloc[0]
        hru = catchment[catchment.geometry.contains(point)]
        if not hru.empty:
            hru_id = hru.iloc[0]['HRU_ID']
            print(f"SNOTEL site {site_number} is located in HRU_ID: {hru_id}")
        else:
            print(f"Could not find HRU for SNOTEL site {site_number}")
            continue  # Skip this site if HRU not found
    else:
        print(f"Could not find location for SNOTEL site {site_number}")
        continue  # Skip this site if location not found
    
    # Save processed data
    output_file = snow_processed_path / f'{domain_name}_SWE_processed_{site_number}_HRU{hru_id}.csv'
    swe_data.to_csv(output_file)
    
    print(f"Processed SWE data for site {site_number} saved to: {output_file}")
    print(f"Total rows in processed data: {len(swe_data)}")
    print(f"Number of non-null values: {swe_data.count()}")