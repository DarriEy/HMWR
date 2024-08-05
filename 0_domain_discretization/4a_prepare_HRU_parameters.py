import geopandas as gpd
from pathlib import Path
from pyproj import CRS

#####################################
### --- Control file handling --- ###
#####################################

# Easy access to control file folder
controlFolder = Path('../0_control_files')

# Store the name of the 'active' file in a variable
controlFile = 'control_active.txt'

# Function to extract a given setting from the control file
def read_from_control(file, setting):
    with open(file) as contents:
        for line in contents:
            if setting in line and not line.startswith('#'):
                return line.split('|', 1)[1].split('#', 1)[0].strip()
    return None

# Function to specify a default path
def make_default_path(suffix):
    rootPath = Path(read_from_control(controlFolder/controlFile, 'root_path'))
    domainName = read_from_control(controlFolder/controlFile, 'domain_name')
    return rootPath / f'domain_{domainName}' / suffix

#########################################
### --- Main: Calculate Parameters ---###
#########################################

def main():
    # Read necessary parameters from the control file
    domain_name = read_from_control(controlFolder/controlFile, 'domain_name')
    catchment_shp_path = read_from_control(controlFolder/controlFile, 'catchment_shp_path')
    catchment_shp_name = read_from_control(controlFolder/controlFile, 'catchment_shp_name')
    
    # Construct file paths
    if catchment_shp_path == 'default':
        input_shapefile = make_default_path(f'shapefiles/catchment/{catchment_shp_name}')
    else:
        input_shapefile = Path(catchment_shp_path) / catchment_shp_name
    
    output_shapefile = input_shapefile
    projected_crs = CRS.from_epsg(32606)

    print(f"Input Shapefile: {input_shapefile}")
    print(f"Output Shapefile: {output_shapefile}")

    # Read the shapefile
    gdf = gpd.read_file(input_shapefile)
    original_crs = gdf.crs

    gdf = gdf.to_crs(projected_crs)

    # Calculate area in square meters
    gdf['HRU_area'] = gdf.geometry.area

    # Calculate centroids
    gdf['center_lon'] = gdf.geometry.centroid.x
    gdf['center_lat'] = gdf.geometry.centroid.y

    # Calculate centroids
    gdf['GRU_ID'] = gdf['COMID']
    gdf['HRU_ID'] = gdf['hruNo']

    # Display the results
    gdf = gdf[['GRU_ID', 'HRU_ID','HRU_area', 'center_lon', 'center_lat', 'geometry']]
    gdf = gdf.to_crs(original_crs)
    print(gdf)
    # Ensure output directory exists
    output_shapefile.parent.mkdir(parents=True, exist_ok=True)

    # Save the updated GeoDataFrame to a new shapefile
    gdf.to_file(output_shapefile)
    print(f"Updated shapefile saved to: {output_shapefile}")

if __name__ == "__main__":
    main()