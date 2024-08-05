import geopandas as gpd
import xarray as xr
import numpy as np
from pathlib import Path

# --- Control file handling
controlFolder = Path('../0_control_files')
controlFile = 'control_active.txt'

def read_from_control(file, setting):
    for line in open(file):
        if setting in line:
            substring = line.split('|', 1)[1].split('#', 1)[0].strip()
            return substring

def make_default_path(suffix):
    rootPath = Path(read_from_control(controlFolder/controlFile, 'root_path'))
    domainName = read_from_control(controlFolder/controlFile, 'domain_name')
    domainFolder = 'domain_' + domainName
    return rootPath / domainFolder / suffix

# Find name and location of catchment shapefile
shp_path = read_from_control(controlFolder/controlFile, 'catchment_shp_path')
shp_name = read_from_control(controlFolder/controlFile, 'catchment_shp_name')

# Specify default path if needed
if shp_path == 'default':
    shp_path = make_default_path('shapefiles/catchment')
else:
    shp_path = Path(shp_path)

# Open the shapefile
shp = gpd.read_file(shp_path/shp_name)

# Get the latitude and longitude of the bounding box
bounding_box = shp.total_bounds

# Function to find nearest RDRS grid points
def find_nearest_rdrs_points(rdrs_lats, rdrs_lons, target_lat, target_lon):
    distances = (rdrs_lats - target_lat)**2 + (rdrs_lons - target_lon)**2
    nearest_index = np.unravel_index(distances.argmin(), distances.shape)
    return rdrs_lats[nearest_index], rdrs_lons[nearest_index]

# Read RDRS grid information
rdrs_file = '/Users/darrieythorsson/compHydro/data/CWARHM_data/domain_Goodpaster_Merit/forcing/2b_RDRS_merged_data/RDRS_monthly_201006.nc'

with xr.open_dataset(rdrs_file) as ds:
    rdrs_lats = ds.lat.values
    rdrs_lons = ds.lon.values

# Find nearest RDRS grid points for bounding box corners
lat_min_rdrs, lon_min_rdrs = find_nearest_rdrs_points(rdrs_lats, rdrs_lons, bounding_box[1], bounding_box[0])
lat_max_rdrs, lon_max_rdrs = find_nearest_rdrs_points(rdrs_lats, rdrs_lons, bounding_box[3], bounding_box[2])

# Add a buffer (e.g., one grid cell in each direction)
lat_buffer = np.abs(rdrs_lats[1, 0] - rdrs_lats[0, 0])
lon_buffer = np.abs(rdrs_lons[0, 1] - rdrs_lons[0, 0])

lat_min_buffered = max(rdrs_lats.min(), lat_min_rdrs - lat_buffer)
lat_max_buffered = min(rdrs_lats.max(), lat_max_rdrs + lat_buffer)
lon_min_buffered = max(rdrs_lons.min(), lon_min_rdrs - lon_buffer)
lon_max_buffered = min(rdrs_lons.max(), lon_max_rdrs + lon_buffer)

# Format the coordinates string
coordinates = f"{lat_max_buffered:.4f}/{lon_min_buffered:.4f}/{lat_min_buffered:.4f}/{lon_max_buffered:.4f}"

print(f"Specify coordinates as {coordinates} in control file.")
print("Note: These coordinates are in the RDRS native grid system.")
print(f"Original shapefile bounds: {bounding_box}")
print(f"Nearest RDRS grid points: {lat_min_rdrs:.4f}, {lon_min_rdrs:.4f}, {lat_max_rdrs:.4f}, {lon_max_rdrs:.4f}")
print(f"Buffered RDRS bounds: {lat_min_buffered:.4f}, {lon_min_buffered:.4f}, {lat_max_buffered:.4f}, {lon_max_buffered:.4f}")