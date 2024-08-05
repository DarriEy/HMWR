import pandas as pd
import geopandas as gpd
from pathlib import Path
import numpy as np
from shapely.geometry import Point
from datetime import datetime
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))
from utils.control_file_utils import read_from_control, make_default_path

# Easy access to control file folder
controlFolder = Path('../../../0_control_files')
controlFile = 'control_active.txt'

# Read necessary paths and settings
domain_name = read_from_control(controlFolder/controlFile, 'domain_name')
forcing_raw_time = read_from_control(controlFolder/controlFile, 'forcing_raw_time')
start_year, end_year = map(int, forcing_raw_time.split(','))

snow_raw_path = read_from_control(controlFolder/controlFile, 'snow_raw_path')
if snow_raw_path == 'default' or snow_raw_path is None:
    snow_raw_path = make_default_path(controlFolder, controlFile, Path('observations') / 'snow' / 'raw_data')
else:
    snow_raw_path = Path(snow_raw_path)

snow_processed_path = read_from_control(controlFolder/controlFile, 'snow_processed_path')
if snow_processed_path == 'default' or snow_processed_path is None:
    snow_processed_path = make_default_path(controlFolder, controlFile, Path('observations') / 'snow' / 'preprocessed')
else:
    snow_processed_path = Path(snow_processed_path)

catchment_shp_path = read_from_control(controlFolder/controlFile, 'catchment_shp_path')
if catchment_shp_path == 'default' or catchment_shp_path is None:
    catchment_shp_path = make_default_path(controlFolder, controlFile, 'shapefiles/catchment')
else:
    catchment_shp_path = Path(catchment_shp_path)

catchment_shp_name = read_from_control(controlFolder/controlFile, 'catchment_shp_name')

# Ensure output directories exist
snow_raw_path.mkdir(parents=True, exist_ok=True)
snow_processed_path.mkdir(parents=True, exist_ok=True)

# Read the catchment shapefile
catchment = gpd.read_file(catchment_shp_path / catchment_shp_name)

# Read the snow data CSV
snow_raw_name = read_from_control(controlFolder/controlFile, 'snow_raw_name')
snow_data_file = snow_raw_path / snow_raw_name  
snow_data = pd.read_csv(snow_data_file, low_memory=False)

# Create a GeoDataFrame from the snow data
geometry = [Point(xy) for xy in zip(snow_data['lon'], snow_data['lat'])]
snow_gdf = gpd.GeoDataFrame(snow_data, geometry=geometry, crs=catchment.crs)

# Find points within the catchment
points_within = gpd.sjoin(snow_gdf, catchment, how="inner", predicate="within")

# Save all data within catchment to raw_data directory
all_data_csv = snow_raw_path / f'{domain_name}_all_snow_observations.csv'
points_within.to_csv(all_data_csv, index=False)

# Convert time to datetime
points_within['time'] = pd.to_datetime(points_within['time'])

# Identify stations with valid observations in the period of interest
stations_with_valid_data = points_within[
    (points_within['time'].dt.year >= start_year) & 
    (points_within['time'].dt.year <= end_year)
]['station_id'].unique()

# Filter data to keep all observations from stations with valid data in the period of interest
filtered_data = points_within[points_within['station_id'].isin(stations_with_valid_data)]

# Keep only required columns
filtered_data = filtered_data[['time', 'station_id', 'snw', 'snd', 'den']]

# Save filtered data to preprocessed folder
filtered_csv = snow_processed_path / f'{domain_name}_filtered_snow_observations.csv'
filtered_data.to_csv(filtered_csv, index=False)

# Create shapefile with locations of all historical stations
all_stations = points_within.drop_duplicates('station_id')

# Create shapefile with locations of stations with valid observations in the period of interest
stations_with_data = points_within[points_within['station_id'].isin(stations_with_valid_data)].drop_duplicates('station_id')

# Select only essential columns for the shapefiles
essential_columns = ['station_id', 'station_name', 'lat', 'lon', 'elevation', 'HRU_ID', 'geometry']
all_stations_shapefile = all_stations[essential_columns].copy()
stations_shapefile = stations_with_data[essential_columns].copy()

# Rename columns to ensure they are unique and not too long
all_stations_shapefile.columns = ['station_id', 'station_nm', 'lat', 'lon', 'elevation', 'HRU_ID', 'geometry']
stations_shapefile.columns = ['station_id', 'station_nm', 'lat', 'lon', 'elevation', 'HRU_ID', 'geometry']

# Create output shapefile directory
output_shp_dir = make_default_path(controlFolder, controlFile, 'shapefiles/observations')
output_shp_dir.mkdir(parents=True, exist_ok=True)

# Save shapefile with all historical stations
all_stations_shp = output_shp_dir / 'all_snow_stations.shp'
try:
    all_stations_shapefile.to_file(all_stations_shp)
except PermissionError:
    print(f"Permission denied: Unable to write to {all_stations_shp}")
    print("Please check your file permissions and try again.")
    exit(1)

# Save shapefile with stations that have data in the period of interest
active_stations_shp = output_shp_dir / 'snow_stations_with_data.shp'
try:
    stations_shapefile.to_file(active_stations_shp)
except PermissionError:
    print(f"Permission denied: Unable to write to {active_stations_shp}")
    print("Please check your file permissions and try again.")
    exit(1)

print(f"All snow observations within catchment saved to: {all_data_csv}")
print(f"Filtered snow observations saved to: {filtered_csv}")
print(f"Shapefile of all historical stations saved to: {all_stations_shp}")
print(f"Shapefile of stations with valid observations in period of interest saved to: {active_stations_shp}")
print(f"Total number of historical stations: {len(all_stations_shapefile)}")
print(f"Number of stations with data in time period: {len(stations_shapefile)}")
print(f"Total number of observations for stations with data in time period: {len(filtered_data)}")
print(f"Number of observations within the specified time period: {len(filtered_data[(filtered_data['time'].dt.year >= start_year) & (filtered_data['time'].dt.year <= end_year)])}")