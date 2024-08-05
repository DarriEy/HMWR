# Intersect catchment with MODIS-derived IGBP land classes
# Counts the occurence of each land class in each HRU in the model setup with rasterstats.

# Modules
import os
import sys
from pathlib import Path
from shutil import copyfile
from datetime import datetime
import geopandas as gpd
import rasterio
from rasterstats import zonal_stats
import pandas as pd
import glob 

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
    
    
# --- Find location of shapefile and land class .tif
# Catchment shapefile path & name
catchment_path = read_from_control(controlFolder/controlFile,'catchment_shp_path')
catchment_name = read_from_control(controlFolder/controlFile,'catchment_shp_name')

# Specify default path if needed
if catchment_path == 'default':
    catchment_path = make_default_path('shapefiles/catchment') # outputs a Path()
else:
    catchment_path = Path(catchment_path) # make sure a user-specified path is a Path()
    
# Forcing shapefile path & name
land_path = read_from_control(controlFolder/controlFile,'parameter_land_mode_path')
land_name = read_from_control(controlFolder/controlFile,'parameter_land_tif_name')

# Specify default path if needed
if land_path == 'default':
    land_path = make_default_path('parameters/landclass/7_mode_landclass') # outputs a Path()
else:
    land_path = Path(land_path) # make sure a user-specified path is a Path()
    
    
# --- Find where the intersection needs to go
# Intersected shapefile path and name
intersect_path = read_from_control(controlFolder/controlFile,'intersect_land_path')
intersect_name = read_from_control(controlFolder/controlFile,'intersect_land_name')

# Specify default path if needed
if intersect_path == 'default':
    intersect_path = make_default_path('shapefiles/catchment_intersection/with_modis') # outputs a Path()
else:
    intersect_path = Path(intersect_path) # make sure a user-specified path is a Path()
    
# Make the folder if it doesn't exist
intersect_path.mkdir(parents=True, exist_ok=True)


# Function to get NoData value from raster
def get_nodata_value(raster_path):
    with rasterio.open(raster_path) as src:
        nodata = src.nodatavals[0]
        if nodata is None:
            # If no NoData value is set, use a default value
            nodata = -9999
        return nodata

# --- Perform zonal statistics
# Load the shapefile
catchment_gdf = gpd.read_file(catchment_path / catchment_name)

# Open the raster file
try:
    raster_path = land_path / land_name
    nodata_value = get_nodata_value(raster_path)
    
    with rasterio.open(raster_path) as src:
        affine = src.transform
        land_data = src.read(1)  # Assuming the data is in the first band

    # Perform zonal statistics
    stats = zonal_stats(catchment_gdf, land_data, affine=affine, stats=['count'], 
                        categorical=True, nodata=nodata_value)

    # Convert the results to a DataFrame
    result_df = pd.DataFrame(stats)

    # Replace NaN (null) values with 0
    result_df = result_df.fillna(0)

    # Function to rename columns
    def rename_column(x):
        if x == 'count':
            return x
        try:
            return f'IGBP_{int(float(x))}'
        except ValueError:
            return x

    # Rename columns to match the desired format with integer IGBP classes
    result_df = result_df.rename(columns=rename_column)

    # Ensure all values are integers
    for col in result_df.columns:
        if col != 'count':  # Assuming 'count' is not a land class column
            result_df[col] = result_df[col].astype(int)

    # Merge the results with the original GeoDataFrame
    catchment_gdf = catchment_gdf.join(result_df)

    # Save the result
    catchment_gdf.to_file(intersect_path / intersect_name)

    print("Processing completed successfully. Results saved.")
    
except Exception as e:
    print(f"Error processing raster: {e}")
    import traceback
    traceback.print_exc()

# --- Code provenance
# Generates a basic log file in the domain folder and copies the control file and itself there.

# Set the log path and file name
logPath = intersect_path
log_suffix = '_catchment_modis_intersect_log.txt'

# Create a log folder
logFolder = '_workflow_log'
Path( logPath / logFolder ).mkdir(parents=True, exist_ok=True)

# Copy this script
thisFile = '3_find_HRU_land_classes.py'
copyfile(thisFile, logPath / logFolder / thisFile);

# Get current date and time
now = datetime.now()

# Create a log file 
logFile = now.strftime('%Y%m%d') + log_suffix
with open( logPath / logFolder / logFile, 'w') as file:
    
    lines = ['Log generated by ' + thisFile + ' on ' + now.strftime('%Y/%m/%d %H:%M:%S') + '\n',
             'Counted the occurrence of IGBP land classes within each HRU.']
    for txt in lines:
        file.write(txt) 
