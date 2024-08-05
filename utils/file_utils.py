# file_utils.py

import os
from pathlib import Path
from shutil import copyfile
import logging

def make_folder_structure(config, logger):
    """Create folder structure for the domain."""
    logger.info("Creating folder structure")
    
    # Create the domain folder
    domain_folder = Path(config.root_path) / f"domain_{config.domain_name}"
    domain_folder.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created domain folder: {domain_folder}")

    # Create shapefile folders
    shapefile_folders = {
        'catchment': config.catchment_shp_path,
        'river_network': config.river_network_shp_path,
        'river_basins': config.river_basin_shp_path,
        'pour_point': config.pour_point_shp_path
    }

    for folder_name, folder_path in shapefile_folders.items():
        if folder_path is None or folder_path == 'default':
            folder_path = f"shapefiles/{folder_name}"
        
        try:
            full_path = domain_folder / folder_path
            full_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created shapefile folder: {full_path}")
        except Exception as e:
            logger.error(f"Failed to create shapefile folder for {folder_name}: {str(e)}")

    # Create log folder
    log_folder = domain_folder / '_workflow_log'
    log_folder.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created log folder: {log_folder}")

    # Copy control file
    control_folder = Path('../0_control_files')
    source_control_file = control_folder / config.source_control_file
    dest_control_file = log_folder / config.source_control_file
    copyfile(source_control_file, dest_control_file)
    logger.info(f"Copied control file from {source_control_file} to {dest_control_file}")

    # Copy preprocessor script
    script_path = Path(__file__).resolve().parent.parent / '1_preprocessor/preProcessor.py'
    dest_script_path = log_folder / 'preprocessor.py'
    copyfile(script_path, dest_script_path)
    logger.info(f"Copied preprocessor script to {dest_script_path}")

    logger.info("Folder structure created successfully")

    return domain_folder