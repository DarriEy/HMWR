import geopandas as gpd
import os
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
from shapely.geometry import Polygon, MultiPolygon, LineString, Point, box
import pvlib
from rasterio.features import rasterize
from scipy.interpolate import griddata

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.control_file_utils import read_from_control, make_default_path

# Easy access to control file folder
controlFolder = Path('../0_control_files')
controlFile = 'control_active.txt'

def plot_hrus(hru_gdf, output_file):
    fig, ax = plt.subplots(figsize=(15, 15))
    hru_gdf.plot(column='radiationClass', cmap='viridis', legend=True, ax=ax)
    ax.set_title('HRUs by Radiation Class')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

def calculate_slope_aspect(dem):
    # Calculate gradients
    dy, dx = np.gradient(dem)
    
    # Calculate slope
    slope = np.arctan(np.sqrt(dx*dx + dy*dy))
    
    # Calculate aspect
    aspect = np.arctan2(-dx, dy)
    
    return slope, aspect

def calculate_annual_radiation(lat, lon, elev, slope, aspect):    
    # Create a DatetimeIndex for the entire year (daily)
    times = pd.date_range(start='2019-01-01', end='2019-12-31', freq='D')
    
    # Create location object
    location = pvlib.location.Location(latitude=lat, longitude=lon, altitude=elev)
    
    # Calculate solar position
    solar_position = location.get_solarposition(times=times)
    
    # Calculate clear sky radiation
    clearsky = location.get_clearsky(times=times)
    
    # Calculate surface tilt and azimuth
    surface_tilt = np.degrees(slope)
    surface_azimuth = np.degrees(aspect)
    
    # Calculate total irradiance
    try:
        total_irrad = pvlib.irradiance.get_total_irradiance(
            surface_tilt, surface_azimuth,
            solar_position['apparent_zenith'], solar_position['azimuth'],
            clearsky['dni'], clearsky['ghi'], clearsky['dhi']
        )
    except Exception as e:
        print(f"Error in get_total_irradiance calculation: {e}")
        return np.nan
    
    # Calculate total incident radiation
    total_radiation = total_irrad['poa_global']
    
    # Return annual sum
    result = total_radiation.sum()
    return result

def calculate_radiation_for_dem(dem, lat, lon):
    # Calculate slope and aspect
    slope, aspect = calculate_slope_aspect(dem)
    
    print(f"DEM shape: {dem.shape}")
    print(f"Slope shape: {slope.shape}")
    print(f"Aspect shape: {aspect.shape}")
    print(f"DEM min: {dem.min()}, max: {dem.max()}")
    print(f"Slope min: {slope.min()}, max: {slope.max()}")
    print(f"Aspect min: {aspect.min()}, max: {aspect.max()}")
    
    # Create a mesh of lat and lon values
    lats = np.linspace(lat - 0.5, lat + 0.5, dem.shape[0])
    lons = np.linspace(lon - 0.5, lon + 0.5, dem.shape[1])
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    
    print(f"lat_grid shape: {lat_grid.shape}")
    print(f"lon_grid shape: {lon_grid.shape}")
    
    # Ensure all arrays have the same shape
    min_shape = min(dem.shape, slope.shape, aspect.shape, lat_grid.shape, lon_grid.shape)
    dem = dem[:min_shape[0], :min_shape[1]]
    slope = slope[:min_shape[0], :min_shape[1]]
    aspect = aspect[:min_shape[0], :min_shape[1]]
    lat_grid = lat_grid[:min_shape[0], :min_shape[1]]
    lon_grid = lon_grid[:min_shape[0], :min_shape[1]]
    
    # Calculate radiation for each point
    radiation = np.zeros_like(dem)
    for i in range(min_shape[0]):
        for j in range(min_shape[1]):
            try:
                rad = calculate_annual_radiation(lat_grid[i,j], lon_grid[i,j], dem[i,j], slope[i,j], aspect[i,j])
                radiation[i,j] = rad
            except Exception as e:
                print(f"Error calculating radiation for i={i}, j={j}, lat={lat_grid[i,j]}, lon={lon_grid[i,j]}, elev={dem[i,j]}, slope={slope[i,j]}, aspect={aspect[i,j]}: {e}")
                radiation[i,j] = np.nan
        
        # Print progress
        if i % 100 == 0:
            print(f"Processed {i} out of {min_shape[0]} rows")
    
    return radiation

def read_radiation_raster(radiation_raster_path):
    with rasterio.open(radiation_raster_path) as src:
        radiation = src.read(1)
        transform = src.transform
        crs = src.crs
    return radiation, transform, crs

def define_radiation_classes(radiation, num_classes):
    min_radiation = np.min(radiation)
    max_radiation = np.max(radiation)
    radiation_thresholds = np.linspace(min_radiation, max_radiation, num_classes + 1)
    return radiation_thresholds

def create_hrus(row, radiation_raster, radiation_thresholds):
    with rasterio.open(radiation_raster) as src:
        out_image, out_transform = mask(src, [row.geometry], crop=True, all_touched=True)
        out_image = out_image[0]
        valid_data = out_image[out_image != src.nodata]
        
        gru_attributes = row.drop('geometry').to_dict()
        
        if len(valid_data) == 0:
            return [{
                'geometry': row.geometry,
                'gruNo': row.name,
                'gruId': row['gruId'],
                'radiationClass': 1,
                'avg_radiation': np.nan,
                **gru_attributes
            }]
        
        hrus = []
        for i in range(len(radiation_thresholds) - 1):
            lower, upper = radiation_thresholds[i:i+2]
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
                                    'radiationClass': i + 1,
                                    'avg_radiation': np.mean(out_image[class_mask]),
                                    **gru_attributes
                                })
        
        return hrus if hrus else [{
            'geometry': row.geometry,
            'gruNo': row.name,
            'gruId': row['gruId'],
            'radiationClass': 1,
            'avg_radiation': np.mean(valid_data),
            **gru_attributes
        }]

def process_hrus(gru_gdf, radiation_raster, radiation_thresholds):
    all_hrus = []
    
    num_cores = max(1, multiprocessing.cpu_count() // 2)
    
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        future_to_row = {executor.submit(create_hrus, row, radiation_raster, radiation_thresholds): row for _, row in gru_gdf.iterrows()}
        for future in as_completed(future_to_row):
            all_hrus.extend(future.result())

    hru_gdf = gpd.GeoDataFrame(all_hrus, crs=gru_gdf.crs)
    
    utm_crs = hru_gdf.estimate_utm_crs()
    hru_gdf_utm = hru_gdf.to_crs(utm_crs)
    hru_gdf_utm['area'] = hru_gdf_utm.area / 1_000_000  # Convert m² to km²

    hru_gdf_utm['area'] = hru_gdf_utm['area'].astype('float64')
    hru_gdf_utm['avg_radiation'] = hru_gdf_utm['avg_radiation'].astype('float64')

    hru_gdf_utm['hruNo'] = range(1, len(hru_gdf_utm) + 1)
    hru_gdf_utm['hruId'] = hru_gdf_utm['gruId'] + '_' + hru_gdf_utm['radiationClass'].astype(str)
    
    centroids_utm = hru_gdf_utm.geometry.centroid
    centroids_wgs84 = centroids_utm.to_crs(CRS.from_epsg(4326))
    
    hru_gdf_utm['cent_lon'] = centroids_wgs84.x
    hru_gdf_utm['cent_lat'] = centroids_wgs84.y
    
    hru_gdf = hru_gdf_utm.to_crs(CRS.from_epsg(4326))
    
    return hru_gdf

def merge_small_hrus(hru_gdf, min_hru_size):
    print(f"Starting with {len(hru_gdf)} HRUs")
    print(f"Minimum HRU size: {min_hru_size} km²")

    # Project to UTM for accurate area calculations
    utm_crs = hru_gdf.estimate_utm_crs()
    hru_gdf_utm = hru_gdf.to_crs(utm_crs)
    
    # Calculate areas
    hru_gdf_utm['area'] = hru_gdf_utm.geometry.area / 1_000_000  # Convert m² to km²
    
    print(f"HRUs below threshold before merging: {len(hru_gdf_utm[hru_gdf_utm['area'] < min_hru_size])}")
    
    merged_count = 0
    while True:
        small_hrus = hru_gdf_utm[hru_gdf_utm['area'] < min_hru_size].sort_values('area')
        if len(small_hrus) == 0:
            break
        
        for idx, small_hru in small_hrus.iterrows():
            neighbors = find_neighbors(small_hru.geometry, hru_gdf_utm, idx)
            if len(neighbors) > 0:
                # Merge with the largest neighbor
                largest_neighbor = neighbors.loc[neighbors['area'].idxmax()]
            else:
                # If no neighbors, find the nearest HRU
                distances = hru_gdf_utm.geometry.distance(small_hru.geometry)
                nearest_idx = distances[distances > 0].idxmin()
                largest_neighbor = hru_gdf_utm.loc[nearest_idx]
            
            merged_geometry = unary_union([make_valid(small_hru.geometry), make_valid(largest_neighbor.geometry)])
            merged_geometry = clean_geometries(merged_geometry)
            
            # Update the neighbor's attributes
            hru_gdf_utm.at[largest_neighbor.name, 'geometry'] = merged_geometry
            hru_gdf_utm.at[largest_neighbor.name, 'area'] = merged_geometry.area / 1_000_000
            
            # Update the radiation class (use area-weighted average)
            total_area = small_hru['area'] + largest_neighbor['area']
            weighted_class = (small_hru['radiationClass'] * small_hru['area'] + 
                              largest_neighbor['radiationClass'] * largest_neighbor['area']) / total_area
            hru_gdf_utm.at[largest_neighbor.name, 'radiationClass'] = round(weighted_class)
            
            # Remove the small HRU
            hru_gdf_utm = hru_gdf_utm.drop(idx)
            merged_count += 1
            
            # Break the loop to recalculate areas
            break
        
        # Recalculate areas
        hru_gdf_utm['area'] = hru_gdf_utm.geometry.area / 1_000_000
    
    print(f"Merged {merged_count} HRUs")
    
    # Update HRU numbers and IDs
    hru_gdf_utm = hru_gdf_utm.reset_index(drop=True)
    hru_gdf_utm['hruNo'] = range(1, len(hru_gdf_utm) + 1)
    hru_gdf_utm['hruId'] = hru_gdf_utm['gruId'] + '_' + hru_gdf_utm['radiationClass'].astype(str)

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

def find_neighbors(geometry, gdf, current_index, buffer_distance=1e-2):
    buffered = geometry.buffer(buffer_distance)
    return gdf[gdf.geometry.intersects(buffered) & (gdf.index != current_index)]

def define_radiation_classes(radiation, num_classes):
    min_radiation = np.nanmin(radiation[np.isfinite(radiation)])
    max_radiation = np.nanmax(radiation[np.isfinite(radiation)])
    if not np.isfinite(min_radiation) or not np.isfinite(max_radiation):
        raise ValueError("Invalid radiation values detected. Please check the radiation calculations.")
    radiation_thresholds = np.linspace(min_radiation, max_radiation, num_classes + 1)
    return radiation_thresholds

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
    dem_raster = root_path / f'domain_{domain_name}' / 'parameters' / 'dem' / '5_elevation' / parameter_dem_tif_name
    radiation_raster = root_path / f'domain_{domain_name}' / 'parameters' / 'radiation' / 'annual_radiation.tif'
    
    # Check if radiation raster exists
    if not radiation_raster.exists():
        print("Annual radiation raster not found. Calculating radiation...")
        
        # Read the DEM raster
        with rasterio.open(dem_raster) as src:
            dem = src.read(1)
            transform = src.transform
            crs = src.crs
            
            # Get the bounds of the raster
            bounds = src.bounds
        
        print(f"DEM shape: {dem.shape}")
        print(f"DEM bounds: {bounds}")
        
        # Calculate the center coordinates of the DEM
        center_lat = (bounds.bottom + bounds.top) / 2
        center_lon = (bounds.left + bounds.right) / 2
        
        print(f"Center coordinates: lat={center_lat}, lon={center_lon}")

        # Calculate radiation
        radiation = calculate_radiation_for_dem(dem, center_lat, center_lon)
        
        # Save the radiation raster
        radiation_raster.parent.mkdir(parents=True, exist_ok=True)
        
        with rasterio.open(radiation_raster, 'w', driver='GTiff',
                           height=radiation.shape[0], width=radiation.shape[1],
                           count=1, dtype=radiation.dtype,
                           crs=crs, transform=transform) as dst:
            dst.write(radiation, 1)

        print(f"Annual radiation raster saved to {radiation_raster}")
    else:
        print(f"Annual radiation raster found at {radiation_raster}")
    
    # Read the radiation raster
    radiation, transform, crs = read_radiation_raster(radiation_raster)
    
    # Read the GRU shapefile
    gru_gdf = gpd.read_file(gru_shapefile).to_crs("EPSG:4326")
    if 'gruId' not in gru_gdf.columns:
        gru_gdf['gruId'] = gru_gdf.index.astype(str)
    
    # Apply make_valid to input shapefile
    gru_gdf['geometry'] = gru_gdf['geometry'].apply(lambda geom: make_valid(geom))

    # Read radiation_class_number from control file
    radiation_class_number = int(read_from_control(controlFolder/controlFile, 'radiation_class_number'))
    
    try:
        radiation_thresholds = define_radiation_classes(radiation, radiation_class_number)
        print(f"Radiation range: {np.nanmin(radiation[np.isfinite(radiation)])} to {np.nanmax(radiation[np.isfinite(radiation)])}")
        print(f"Radiation thresholds: {radiation_thresholds}")
    except ValueError as e:
        print(f"Error in radiation calculations: {e}")
        print("Please check the radiation raster and calculations.")
        return  # Exit the main function if there's an error
    
    # Generate HRUs
    hru_gdf = process_hrus(gru_gdf, radiation_raster, radiation_thresholds)
    hru_gdf['geometry'] = hru_gdf['geometry'].apply(lambda geom: make_valid(geom))

    # Merge small HRUs
    min_hru_size = float(read_from_control(controlFolder/controlFile, 'min_hru_size'))
    hru_gdf = merge_small_hrus(hru_gdf, min_hru_size)
    
    # Save the HRU shapefile
    output_shapefile = make_default_path(controlFolder, controlFile,f'shapefiles/catchment/{domain_name}_HRUs_radiation.shp')
    hru_gdf.to_file(output_shapefile)
    print(f"HRU Shapefile created with {len(hru_gdf)} HRUs and saved to {output_shapefile}")
    
    # Plot HRUs
    output_plot = make_default_path(controlFolder, controlFile,f'plots/catchment/{domain_name}_HRUs_radiation.png')
    plot_hrus(hru_gdf, output_plot)
    print(f"HRU plot saved to {output_plot}")

if __name__ == "__main__":
    main()