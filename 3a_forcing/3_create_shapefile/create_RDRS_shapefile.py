#!/usr/bin/env python
# coding: utf-8

import os
import geopandas as gpd
import xarray as xr
import numpy as np
from pathlib import Path
from shutil import copyfile
from datetime import datetime   
from shapely.geometry import Polygon
import rasterio
import rasterstats
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from utils.control_file_utils import read_from_control, make_default_path

# ---- Control file handling
controlFolder = Path('../../0_control_files')
controlFile = 'control_active.txt'

# --- Find location of merged RDRS data
mergePath = read_from_control(controlFolder/controlFile,'forcing_merged_path')
if mergePath == 'default':
    mergePath = make_default_path(controlFolder, controlFile,'forcing/2_merged_data')
else: 
    mergePath = Path(mergePath)

# --- Find where the shapefile needs to go
shapePath = read_from_control(controlFolder/controlFile,'forcing_shape_path')
if shapePath == 'default':
    shapePath = make_default_path(controlFolder, controlFile,'shapefiles/forcing')
else: 
    shapePath = Path(shapePath)

# Find name of the new shapefile
shapeName = read_from_control(controlFolder/controlFile,'forcing_rdrs_shape_name')

# Find the names of the latitude and longitude fields
field_lat = read_from_control(controlFolder/controlFile,'forcing_shape_lat_name')
field_lon = read_from_control(controlFolder/controlFile,'forcing_shape_lon_name')

# --- Read the source file to find the grid spacing
for file in os.listdir(mergePath):
    if file.endswith('.nc') and file.startswith('RDRS_monthly_'):
        forcing_file = file
        break

# Read the RDRS file
with xr.open_dataset(mergePath / forcing_file) as ds:
    rlat = ds.rlat.values
    rlon = ds.rlon.values
    lat = ds.lat.values
    lon = ds.lon.values

# Create lists to store the data
geometries = []
ids = []
lats = []
lons = []

for i in range(len(rlat)):
    for j in range(len(rlon)):
        # Get the corners of the grid cell in rotated coordinates
        rlat_corners = [rlat[i], rlat[i], rlat[i+1] if i+1 < len(rlat) else rlat[i], rlat[i+1] if i+1 < len(rlat) else rlat[i]]
        rlon_corners = [rlon[j], rlon[j+1] if j+1 < len(rlon) else rlon[j], rlon[j+1] if j+1 < len(rlon) else rlon[j], rlon[j]]
        
        # Convert rotated coordinates to lat/lon
        lat_corners = [lat[i,j], lat[i,j+1] if j+1 < len(rlon) else lat[i,j], 
                       lat[i+1,j+1] if i+1 < len(rlat) and j+1 < len(rlon) else lat[i,j], 
                       lat[i+1,j] if i+1 < len(rlat) else lat[i,j]]
        lon_corners = [lon[i,j], lon[i,j+1] if j+1 < len(rlon) else lon[i,j], 
                       lon[i+1,j+1] if i+1 < len(rlat) and j+1 < len(rlon) else lon[i,j], 
                       lon[i+1,j] if i+1 < len(rlat) else lon[i,j]]
        
        # Create polygon
        poly = Polygon(zip(lon_corners, lat_corners))
        
        # Append to lists
        geometries.append(poly)
        ids.append(i * len(rlon) + j)
        lats.append(lat[i,j])
        lons.append(lon[i,j])

# Create the GeoDataFrame
gdf = gpd.GeoDataFrame({
    'geometry': geometries,
    'ID': ids,
    field_lat: lats,
    field_lon: lons,
}, crs='EPSG:4326')

# --- Add elevation data from DEM
dem_path = make_default_path(controlFolder, controlFile,'parameters/dem/5_elevation/elevation.tif')

# Calculate zonal statistics (mean elevation) for each grid cell
zs = rasterstats.zonal_stats(gdf, dem_path, stats=['mean'])

# Add mean elevation to the GeoDataFrame
gdf['elev_m'] = [item['mean'] for item in zs]

# drop columns that are on the edge and dont have elevation data
gdf.dropna(subset=['elev_m'], inplace=True)

# Save the shapefile
shapePath.mkdir(parents=True, exist_ok=True)
gdf.to_file(shapePath / shapeName)

# --- Code provenance
logFolder = '_workflow_log'
Path(shapePath / logFolder).mkdir(parents=True, exist_ok=True)

thisFile = 'create_RDRS_shapefile.py'
copyfile(thisFile, shapePath / logFolder / thisFile)

now = datetime.now()

logFile = now.strftime('%Y%m%d') + '_RDRS_shapefile_log.txt'
with open(shapePath / logFolder / logFile, 'w') as file:
    lines = [
        f"Log generated by {thisFile} on {now.strftime('%Y/%m/%d %H:%M:%S')}\n",
        'Created RDRS regular latitude/longitude grid shapefile with mean elevation from DEM for each grid cell.'
    ]
    for txt in lines:
        file.write(txt + '\n')

print("RDRS shapefile creation complete.")