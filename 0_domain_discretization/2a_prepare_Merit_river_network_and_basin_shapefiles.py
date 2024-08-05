import sys
from pathlib import Path
import geopandas as gpd
import pandas as pd
from pyproj import CRS
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.control_file_utils import read_from_control, make_default_path

#####################################
### --- Control file handling --- ###
#####################################

# Easy access to control file folder
controlFolder = Path('../0_control_files')
controlFile = 'control_active.txt'

# Read necessary paths and settings
root_path = Path(read_from_control(controlFolder/controlFile, 'root_path'))
domain_name = read_from_control(controlFolder/controlFile, 'domain_name')

# Define input/output file paths
river_network_path = make_default_path(controlFolder, controlFile, f'shapefiles/river_network/{domain_name}_network.shp')
basins_path = make_default_path(controlFolder, controlFile, f'shapefiles/river_basins/{domain_name}_basins.shp')

# Load the shapefiles
river_network = gpd.read_file(river_network_path)
basins = gpd.read_file(basins_path)

# Store the original CRS
original_crs = river_network.crs

# Define a suitable projected CRS (you may need to adjust this based on your study area)
# This example uses UTM Zone 6N, which is suitable for parts of Alaska including the Chena River
projected_crs = CRS.from_epsg(32606)

# Process river network
print("Processing river network...")
river_network = river_network.to_crs(projected_crs)
river_network['length'] = river_network.geometry.length
river_network = river_network[['COMID', 'slope', 'NextDownID', 'length', 'geometry']]
river_network['length'] = river_network['length'].round(2)  # Round to 2 decimal places

# Process basins
print("Processing basins...")
basins = basins.to_crs(projected_crs)
basins['area'] = basins.geometry.area
basins['hru_to_seg'] = basins['COMID']  # Shortened column name
basins = basins[['COMID', 'area', 'hru_to_seg', 'geometry']]
basins['area'] = basins['area'].round(2)  # Round to 2 decimal places

# Project back to original CRS
river_network = river_network.to_crs(original_crs)
basins = basins.to_crs(original_crs)

# Round the numbers after projecting back to original CRS
river_network['length'] = river_network['length'].round(2)
basins['area'] = basins['area'].round(2)

# Save the processed shapefiles, overwriting the original files
print("Saving processed shapefiles...")
river_network.to_file(river_network_path)
basins.to_file(basins_path)

print(f"Processed river network saved to: {river_network_path}")
print(f"Processed basins saved to: {basins_path}")
