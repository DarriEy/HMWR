import geopandas as gpd
import sys
import pandas as pd
import rasterio
from rasterio.mask import mask
import numpy as np
from shapely.geometry import shape, Polygon
from shapely.ops import unary_union, polygonize
import matplotlib.pyplot as plt
from pyproj import CRS
from pathlib import Path
from shapely.errors import GEOSException
from shapely.validation import make_valid
from shapely.errors import TopologicalError
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from shapely.geometry import Polygon, MultiPolygon, LineString, Point

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.control_file_utils import read_from_control, make_default_path

# Easy access to control file and folder folder
controlFolder = Path('../0_control_files')
controlFile = 'control_active.txt'


########################################
### --- Discretisation Functions --- ###
########################################

def read_and_prepare_data(gru_shapefile, dem_raster, elevation_band_size):
    gru_gdf = gpd.read_file(gru_shapefile).to_crs("EPSG:4326")
    if 'gruId' not in gru_gdf.columns:
        gru_gdf['gruId'] = gru_gdf.index.astype(str)

    # Apply make_valid to input shapefile
    gru_gdf['geometry'] = gru_gdf['geometry'].apply(lambda geom: make_valid(geom))

    with rasterio.open(dem_raster) as src:
        # Mask the DEM with the GRU shapefile
        out_image, out_transform = rasterio.mask.mask(src, gru_gdf.geometry, crop=True)
        out_image = out_image[0]  # Get the first band
        valid_data = out_image[out_image != src.nodata]
        
        min_elevation = np.floor(np.min(valid_data))
        max_elevation = np.ceil(np.max(valid_data))

    elevation_thresholds = np.arange(min_elevation, max_elevation + elevation_band_size, elevation_band_size)
    
    print(f"Elevation range: {min_elevation} to {max_elevation}")
    print(f"Elevation thresholds: {elevation_thresholds}")
    
    return gru_gdf, elevation_thresholds

def find_neighbors(geometry, gdf, current_index, buffer_distance=1e-6):
    buffered = geometry.buffer(buffer_distance)
    return gdf[gdf.geometry.intersects(buffered) & (gdf.index != current_index)]

def to_polygon(geom):
    if isinstance(geom, MultiPolygon):
        if len(geom.geoms) == 1:
            return geom.geoms[0]
        else:
            return geom.buffer(0)
    return geom

def clean_geometries(geometry):
    if geometry.is_empty:
        return None
    if isinstance(geometry, (Polygon, MultiPolygon)):
        if geometry.is_valid:
            return geometry
        else:
            return geometry.buffer(0)
    elif isinstance(geometry, LineString):
        return Polygon(geometry).buffer(0)
    elif isinstance(geometry, Point):
        return geometry.buffer(1e-6)
    else:
        try:
            return MultiPolygon(polygonize(geometry)).buffer(0)
        except:
            return None

def merge_small_hrus(hru_gdf, min_hru_size):
    print(f"Starting with {len(hru_gdf)} HRUs")
    print(f"Minimum HRU size: {min_hru_size} km²")

    # Project to UTM for accurate area calculations
    utm_crs = hru_gdf.estimate_utm_crs()
    hru_gdf_utm = hru_gdf.to_crs(utm_crs)
    
    # Calculate areas
    hru_gdf_utm['area'] = hru_gdf_utm.geometry.area / 1_000_000  # Convert m² to km²
    
    print(f"HRUs below threshold before merging: {len(hru_gdf_utm[hru_gdf_utm['area'] < min_hru_size])}")
    
    # Sort HRUs by area
    hru_gdf_utm = hru_gdf_utm.sort_values('area')
    
    merged_count = 0
    problematic_hrus = []
    while True:
        small_hrus = hru_gdf_utm[hru_gdf_utm['area'] < min_hru_size]
        if len(small_hrus) == 0:
            break
        
        for idx, small_hru in small_hrus.iterrows():
            neighbors = find_neighbors(small_hru.geometry, hru_gdf_utm, idx)
            if len(neighbors) > 0:
                largest_neighbor = neighbors.loc[neighbors['area'].idxmax()]
                merged_geometry = unary_union([make_valid(small_hru.geometry), make_valid(largest_neighbor.geometry)])
                merged_geometry = clean_geometries(merged_geometry)
                
                # Update the largest neighbor's geometry and area
                hru_gdf_utm.at[largest_neighbor.name, 'geometry'] = merged_geometry
                hru_gdf_utm.at[largest_neighbor.name, 'area'] = merged_geometry.area / 1_000_000
                
                # Remove the small HRU
                hru_gdf_utm = hru_gdf_utm.drop(idx)
                merged_count += 1
            else:
                print(f"No neighbors found for HRU {idx} (area: {small_hru['area']:.6f} km²)")
                problematic_hrus.append(small_hru)
        
        # Recalculate areas and sort
        hru_gdf_utm['area'] = hru_gdf_utm.geometry.area / 1_000_000
        hru_gdf_utm = hru_gdf_utm.sort_values('area')
    
    print(f"Merged {merged_count} HRUs")
    
    # Update HRU numbers and IDs
    hru_gdf_utm = hru_gdf_utm.reset_index(drop=True)
    hru_gdf_utm['hruNo'] = range(1, len(hru_gdf_utm) + 1)
    hru_gdf_utm['hruId'] = hru_gdf_utm['gruId'] + '_' + hru_gdf_utm['elevClass'].astype(str)

    # Clean all geometries
    hru_gdf_utm['geometry'] = hru_gdf_utm['geometry'].apply(clean_geometries)
    
    # Additional cleanup steps
    hru_gdf_utm['geometry'] = hru_gdf_utm['geometry'].buffer(0.1).buffer(-0.1)  # Close small gaps
    hru_gdf_utm['geometry'] = hru_gdf_utm['geometry'].simplify(1)  # Simplify geometries slightly
    
    # Remove any remaining line geometries
    hru_gdf_utm = hru_gdf_utm[hru_gdf_utm['geometry'].apply(lambda geom: isinstance(geom, (Polygon, MultiPolygon)))]
    
    hru_gdf_utm = hru_gdf_utm[hru_gdf_utm['geometry'].notnull()]

    # Project back to WGS84
    hru_gdf_merged = hru_gdf_utm.to_crs(hru_gdf.crs)
    hru_gdf_merged['geometry'] = hru_gdf_merged['geometry'].apply(lambda geom: make_valid(geom))
    
    print(f"Finished with {len(hru_gdf_merged)} HRUs")
    print(f"HRUs below threshold after merging: {len(hru_gdf_merged[hru_gdf_merged['area'] < min_hru_size])}")
    
    return hru_gdf_merged

def create_hrus(row, dem_raster, elevation_thresholds):
    with rasterio.open(dem_raster) as dem_src:
        out_image, out_transform = mask(dem_src, [row.geometry], crop=True, all_touched=True)
        out_image = out_image[0]
        valid_data = out_image[out_image != dem_src.nodata]
        
        # Get GRU attributes (excluding 'geometry')
        gru_attributes = row.drop('geometry').to_dict()
        
        if len(valid_data) == 0:
            return [{
                'geometry': row.geometry,
                'gruNo': row.name,
                'gruId': row['gruId'],
                'elevClass': 1,
                'avg_elevation': np.nan,
                **gru_attributes  # Include all GRU attributes except geometry
            }]
        
        hrus = []
        for i in range(len(elevation_thresholds) - 1):
            lower, upper = elevation_thresholds[i:i+2]
            class_mask = (out_image >= lower) & (out_image < upper)
            if np.any(class_mask):
                shapes = rasterio.features.shapes(class_mask.astype(np.uint8), mask=class_mask, transform=out_transform)
                class_polys = [shape(shp) for shp, _ in shapes]
                
                if class_polys:
                    merged_poly = unary_union(class_polys).intersection(row.geometry)
                    if not merged_poly.is_empty:
                        geoms = [merged_poly] if isinstance(merged_poly, Polygon) else merged_poly.geoms
                        for geom in geoms:
                            if isinstance(geom, Polygon):
                                hrus.append({
                                    'geometry': geom,
                                    'gruNo': row.name,
                                    'gruId': row['gruId'],
                                    'elevClass': i + 1,
                                    'avg_elevation': np.mean(out_image[class_mask]),
                                    **gru_attributes  # Include all GRU attributes except geometry
                                })
        
        return hrus if hrus else [{
            'geometry': row.geometry,
            'gruNo': row.name,
            'gruId': row['gruId'],
            'elevClass': 1,
            'avg_elevation': np.mean(valid_data),
            **gru_attributes  # Include all GRU attributes except geometry
        }]

def process_hrus(gru_gdf, dem_raster, elevation_thresholds):
    all_hrus = []
    
    # Use half of available CPU cores
    num_cores = max(1, multiprocessing.cpu_count() // 2)
    
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        future_to_row = {executor.submit(create_hrus, row, dem_raster, elevation_thresholds): row for _, row in gru_gdf.iterrows()}
        for future in as_completed(future_to_row):
            all_hrus.extend(future.result())

    hru_gdf = gpd.GeoDataFrame(all_hrus, crs=gru_gdf.crs)
    
    # Project to UTM for area calculation
    utm_crs = hru_gdf.estimate_utm_crs()
    hru_gdf_utm = hru_gdf.to_crs(utm_crs)
    hru_gdf_utm['area'] = hru_gdf_utm.area / 1_000_000  # Convert m² to km²

    # Ensure 'area' and 'avg_elevation' are float64
    hru_gdf_utm['area'] = hru_gdf_utm['area'].astype('float64')
    hru_gdf_utm['avg_elevation'] = hru_gdf_utm['avg_elevation'].astype('float64')

    hru_gdf_utm['hruNo'] = range(1, len(hru_gdf_utm) + 1)
    hru_gdf_utm['hruId'] = hru_gdf_utm['gruId'] + '_' + hru_gdf_utm['elevClass'].astype(str)
    
    # Calculate centroids in UTM
    centroids_utm = hru_gdf_utm.geometry.centroid
    
    # Convert centroids back to WGS84
    centroids_wgs84 = centroids_utm.to_crs(CRS.from_epsg(4326))
    
    hru_gdf_utm['cent_lon'] = centroids_wgs84.x
    hru_gdf_utm['cent_lat'] = centroids_wgs84.y
    
    # Convert the final GeoDataFrame back to WGS84 for consistency
    hru_gdf = hru_gdf_utm.to_crs(CRS.from_epsg(4326))
    
    return hru_gdf

def plot_hrus(hru_gdf, output_file):
    fig, ax = plt.subplots(figsize=(15, 15))
    hru_gdf.plot(column='elevClass', cmap='terrain', legend=True, ax=ax)
    ax.set_title('HRUs by Elevation Class')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()  # Close the plot to free up memory

#######################################
### --- Main: Generate and HRUs --- ###
#######################################

def main():
    # Read necessary parameters from the control file
    root_path = Path(read_from_control(controlFolder/controlFile, 'root_path'))
    full_domain_name = read_from_control(controlFolder/controlFile, 'full_domain_name')
    domain_name = read_from_control(controlFolder/controlFile, 'domain_name')
    catchment_shp_path = read_from_control(controlFolder/controlFile, 'catchment_shp_path')
    catchment_shp_name = read_from_control(controlFolder/controlFile, 'catchment_shp_name')
    parameter_dem_tif_name = read_from_control(controlFolder/controlFile, 'parameter_dem_tif_name')
    
    # Construct file paths
    gru_shapefile = make_default_path(controlFolder, controlFile,f'shapefiles/river_basins/{domain_name}_basins.shp')
    dem_raster = root_path / f'domain_{full_domain_name}' / 'parameters' / 'dem' / '5_elevation' / parameter_dem_tif_name
    
    if catchment_shp_path == 'default':
        output_shapefile = make_default_path(controlFolder, controlFile,f'shapefiles/catchment/{catchment_shp_name}')
    else:
        output_shapefile = Path(catchment_shp_path) / catchment_shp_name
    
    output_plot = make_default_path(controlFolder, controlFile,f'plots/catchment/{domain_name}_HRUs_elevation.png')
    
    elevation_band_size = float(read_from_control(controlFolder/controlFile, 'elevation_band_size'))
    min_hru_size = float(read_from_control(controlFolder/controlFile, 'min_hru_size'))

    print(f"GRU Shapefile: {gru_shapefile}")
    print(f"DEM Raster: {dem_raster}")
    print(f"Output Shapefile: {output_shapefile}")
    print(f"Output Plot: {output_plot}")
    print(f"Elevation Band Size: {elevation_band_size}")
    print(f"Minimum HRU Size: {min_hru_size}")

    # Check if input files exist
    if not gru_shapefile.exists():
        raise FileNotFoundError(f"Input GRU shapefile not found: {gru_shapefile}")
    if not dem_raster.exists():
        raise FileNotFoundError(f"Input DEM raster not found: {dem_raster}")

    gru_gdf, elevation_thresholds = read_and_prepare_data(gru_shapefile, dem_raster, elevation_band_size)
    hru_gdf = process_hrus(gru_gdf, dem_raster, elevation_thresholds)

    if hru_gdf is not None and not hru_gdf.empty:
        print(hru_gdf)
        
        hru_gdf = merge_small_hrus(hru_gdf, min_hru_size)
        
        # Ensure all geometries are valid polygons
        hru_gdf['geometry'] = hru_gdf['geometry'].apply(clean_geometries)
        hru_gdf = hru_gdf[hru_gdf['geometry'].notnull()]
        
        # Remove any columns that might cause issues with shapefiles
        columns_to_keep = ['geometry', 'gruNo', 'gruId', 'elevClass', 'avg_elevation', 'area', 'hruNo', 'hruId', 'cent_lon', 'cent_lat','COMID']
        hru_gdf = hru_gdf[columns_to_keep]
        
        # Rename columns to ensure they're not too long for shapefiles
        hru_gdf = hru_gdf.rename(columns={
            'avg_elevation': 'avg_elev',
            'hru_to_seg': 'hru_seg'
        })

        # Final check for valid polygons
        valid_polygons = []
        for idx, row in hru_gdf.iterrows():
            geom = row['geometry']
            if isinstance(geom, (Polygon, MultiPolygon)) and geom.is_valid:
                valid_polygons.append(row)
            else:
                print(f"Removing invalid geometry for HRU {idx}")
        
        hru_gdf = gpd.GeoDataFrame(valid_polygons, crs=hru_gdf.crs)

        # Save as Shapefile
        hru_gdf['geometry'] = to_polygon(hru_gdf['geometry'])     
        
        hru_gdf.to_file(output_shapefile)
        print(f"HRU Shapefile created with {len(hru_gdf)} HRUs and saved to {output_shapefile}")

        plot_hrus(hru_gdf, output_plot)
        print(f"HRU plot saved to {output_plot}")
    else:
        print("Error: No valid HRUs were created. Check your input data and parameters.")

if __name__ == "__main__":
    main()