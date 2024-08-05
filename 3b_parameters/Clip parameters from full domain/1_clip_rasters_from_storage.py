import sys
from pathlib import Path
import rasterio
from rasterio.mask import mask
from shapely.geometry import box
import geopandas as gpd

#####################################
### --- Control file handling --- ###
#####################################

# Easy access to control file folder
controlFolder = Path('../../0_control_files')

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
root_path = Path(read_from_control(controlFolder/controlFile, 'root_path'))
domain_name = read_from_control(controlFolder/controlFile, 'domain_name')
full_domain_name = read_from_control(controlFolder/controlFile, 'full_domain_name')
forcing_raw_space = read_from_control(controlFolder/controlFile, 'forcing_raw_space')

# Parse coordinates and add padding
coords = [float(coord) for coord in forcing_raw_space.split('/')]
padding = 0.3
min_lon, min_lat, max_lon, max_lat = coords[1]-padding, coords[2]-padding, coords[3]+padding, coords[0]+padding


def clip_raster(input_raster, output_raster, min_lon, min_lat, max_lon, max_lat):
    # Create a bounding box
    bbox = box(min_lon, min_lat, max_lon, max_lat)
    
    # Convert the bbox to a GeoDataFrame
    geo = gpd.GeoDataFrame({'geometry': bbox}, index=[0], crs="EPSG:4326")
    
    # Open the input raster
    with rasterio.open(input_raster) as src:
        # Reproject the geometry to match the raster's CRS
        geo = geo.to_crs(src.crs)
        
        # Perform the clip
        out_image, out_transform = mask(src, geo.geometry, crop=True)
        
        # Copy the metadata
        out_meta = src.meta.copy()
        
        # Update the metadata
        out_meta.update({"driver": "GTiff",
                         "height": out_image.shape[1],
                         "width": out_image.shape[2],
                         "transform": out_transform})
    
    # Write the clipped raster to a new file
    with rasterio.open(output_raster, "w", **out_meta) as dest:
        dest.write(out_image)

# Define paths
dem_full_domain = root_path / f'domain_{full_domain_name}' / 'parameters' / 'dem' / '5_elevation' / 'elevation.tif'
land_class_full_domain = root_path / f'domain_{full_domain_name}' / 'parameters' / 'landclass' / '7_mode_land_class' / 'land_classes.tif'
soil_class_full_domain = root_path / f'domain_{full_domain_name}' / 'parameters' / 'soilclass' / '2_soil_classes_domain' / 'soil_classes.tif'

dem_sub_domain = make_default_path(Path('parameters') / 'dem' / '5_elevation' / 'elevation.tif')
land_class_sub_domain = make_default_path(Path('parameters') / 'landclass' / '7_mode_landclass' / 'land_classes.tif')
soil_class_sub_domain = make_default_path(Path('parameters') / 'soilclass' / '2_soil_classes_domain' / 'soil_classes.tif')

# Ensure output directories exist
dem_sub_domain.parent.mkdir(parents=True, exist_ok=True)
land_class_sub_domain.parent.mkdir(parents=True, exist_ok=True)
soil_class_sub_domain.parent.mkdir(parents=True, exist_ok=True)

# Clip rasters
clip_raster(str(dem_full_domain), str(dem_sub_domain), min_lon, min_lat, max_lon, max_lat)
clip_raster(str(land_class_full_domain), str(land_class_sub_domain), min_lon, min_lat, max_lon, max_lat)
clip_raster(str(soil_class_full_domain), str(soil_class_sub_domain), min_lon, min_lat, max_lon, max_lat)