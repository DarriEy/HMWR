#!/usr/bin/env python
# coding: utf-8

'''
SUMMA workflow: make folder structure
Makes the initial folder structure for a given control file. All other files in the workflow will look for the file `control_active.txt` during their execution. This script:

1. Copies the specified control file into `control_active.txt`;
2. Prepares a folder structure using the settings in `control_active.txt`.
3. Creates a copy of itself to be stored in the new folder structure.

The destination folders are referred to as "domain folders".
'''

# Specify the control file to use
sourceFile  = 'control_active.txt'

# --- Do not change below this line.

# Modules
import os
from pathlib import Path
from shutil import copyfile
from datetime import datetime

# --- Copy the control file into `control_active.txt`
# Easy access to control file folder
controlFolder = Path('../0_control_files')

# Store the name of the 'active' file in a variable
controlFile = 'control_active.txt'

# Copy
#copyfile( controlFolder/sourceFile, controlFolder/controlFile );

# --- Create the main domain folders
# Function to extract a given setting from the control file
def read_from_control( file, setting ):
    
    # Open 'control_active.txt' and ...
    with open(file) as contents:
        for line in contents:
            
            # ... find the line with the requested setting
            if setting in line:
                break
    
    # Extract the setting's value
    substring = line.split('|',1)[1]      # Remove the setting's name (split into 2 based on '|', keep only 2nd part)
    substring = substring.split('#',1)[0] # Remove comments, does nothing if no '#' is found
    substring = substring.strip()         # Remove leading and trailing whitespace, tabs, newlines
       
    # Return this value    
    return substring
    
# Find the path where the domain folders need to go
# Immediately store as a 'Path' to avoid issues with '/' and '\' on different operating systems
rootPath = Path( read_from_control(controlFolder/controlFile,'root_path') )

# Find the domain name
domainName = read_from_control(controlFolder/controlFile,'domain_name')

# Create the domain folder inside 'root'
domainFolder = 'domain_' + domainName
Path( rootPath / domainFolder ).mkdir(parents=True, exist_ok=True)


# --- Make the shapefile folders
# Find the catchment shapefile folder in 'control_active'
catchmentShapeFolder = read_from_control(controlFolder/controlFile,'catchment_shp_path')
networkShapeFolder = read_from_control(controlFolder/controlFile,'river_network_shp_path')
riverBasinFolder =  read_from_control(controlFolder/controlFile,'river_basin_shp_path')
pourPointFolder =  read_from_control(controlFolder/controlFile,'pour_point_shp_path')

# Specify the default paths if required
if catchmentShapeFolder == 'default':
    catchmentShapeFolder = 'shapefiles/catchment'
if networkShapeFolder == 'default':
    networkShapeFolder = 'shapefiles/river_network'
if riverBasinFolder == 'default':
    riverBasinFolder = 'shapefiles/river_basins'
if pourPointFolder == 'default':
    pourPointFolder = 'shapefiles/pour_point'

# Try to make the shapefile folders; does nothing if the folder already exists
Path( rootPath / domainFolder / catchmentShapeFolder ).mkdir(parents=True, exist_ok=True)
Path( rootPath / domainFolder / networkShapeFolder ).mkdir(parents=True, exist_ok=True)
Path( rootPath / domainFolder / riverBasinFolder ).mkdir(parents=True, exist_ok=True)
Path( rootPath / domainFolder / pourPointFolder ).mkdir(parents=True, exist_ok=True)


# --- Code provenance
# Generates a basic log file in the domain folder and copies the control file and itself there.
# Create a log folder
logFolder = '_workflow_log'
Path( rootPath / domainFolder / logFolder ).mkdir(parents=True, exist_ok=True)

# Copy the control file
copyfile(controlFolder / sourceFile, rootPath / domainFolder / logFolder / sourceFile);

# Copy this script
thisFile = 'make_folder_structure.py'
copyfile(thisFile, rootPath / domainFolder / logFolder / thisFile);

# Get current date and time
now = datetime.now()

# Create a log file 
logFile = now.strftime('%Y%m%d') + '_log.txt'
with open(rootPath / domainFolder / logFolder / logFile, 'w') as file:
    
    lines = ['Log generated by ' + thisFile + ' on ' + now.strftime('%Y/%m/%d %H:%M:%S') + '\n',
             'Generated folder structure using ' + sourceFile]
    for txt in lines:
        file.write(txt) 