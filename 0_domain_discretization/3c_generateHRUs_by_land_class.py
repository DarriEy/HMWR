import geopandas as gpd
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
from shapely.geometry import Polygon, MultiPolygon, LineString, Point, MultiLineString


sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.control_file_utils import read_from_control, make_default_path

# Easy access to control file folder
controlFolder = Path('../0_control_files')
controlFile = 'control_active.txt'

def read_and_prepare_data(gru_shapefile, land_raster):
    gru_gdf = gpd.read_file(gru_shapefile).to_crs("EPSG:4326")
    if 'gruId' not in gru_gdf.columns:
        gru_gdf['gruId'] = gru_gdf.index.astype(str)

    gru_gdf['geometry'] = gru_gdf['geometry'].apply(lambda geom: make_valid(geom))

    with rasterio.open(land_raster) as src:
        out_image, out_transform = rasterio.mask.mask(src, gru_gdf.geometry, crop=True)
        out_image = out_image[0]  # Get the first band
        unique_land_classes = np.unique(out_image[out_image != src.nodata])
    
    print(f"Unique land classes: {unique_land_classes}")
    
    return gru_gdf, unique_land_classes

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
    if geometry is None or geometry.is_empty:
        return None
    if isinstance(geometry, (Polygon, MultiPolygon)):
        if geometry.is_valid:
            return geometry
        else:
            cleaned = geometry.buffer(0)
            if cleaned.is_empty:
                return None
            return cleaned
    elif isinstance(geometry, (LineString, MultiLineString)):
        return geometry.buffer(1e-6)
    elif isinstance(geometry, Point):
        return geometry.buffer(1e-6)
    else:
        try:
            return MultiPolygon(polygonize(geometry)).buffer(0)
        except:
            return None

def ensure_full_coverage(hru_gdf, gru_gdf):
    gru_union = gru_gdf.geometry.unary_union
    hru_union = hru_gdf.geometry.unary_union
    gaps = gru_union.difference(hru_union)
    
    if not gaps.is_empty:
        if isinstance(gaps, Polygon):
            gaps = MultiPolygon([gaps])
        
        for gap in gaps.geoms:
            nearest_hru = hru_gdf.geometry.distance(gap).idxmin()
            hru_gdf.at[nearest_hru, 'geometry'] = hru_gdf.at[nearest_hru, 'geometry'].union(gap)
    
    hru_gdf['area'] = hru_gdf.geometry.area / 1_000_000  # Recalculate areas
    return hru_gdf


def merge_small_hrus(hru_gdf, min_hru_size):
    print(f"Starting with {len(hru_gdf)} HRUs")
    print(f"Minimum HRU size: {min_hru_size} km²")

    utm_crs = hru_gdf.estimate_utm_crs()
    hru_gdf_utm = hru_gdf.to_crs(utm_crs)
    
    hru_gdf_utm['area'] = hru_gdf_utm.geometry.area / 1_000_000  # Convert m² to km²
    
    print(f"HRUs below threshold before merging: {len(hru_gdf_utm[hru_gdf_utm['area'] < min_hru_size])}")
    
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
                # Prioritize merging with neighbors of the same land class
                same_land_neighbors = neighbors[neighbors['landClass'] == small_hru['landClass']]
                if len(same_land_neighbors) > 0:
                    largest_neighbor = same_land_neighbors.loc[same_land_neighbors['area'].idxmax()]
                else:
                    largest_neighbor = neighbors.loc[neighbors['area'].idxmax()]
                
                merged_geometry = unary_union([make_valid(small_hru.geometry), make_valid(largest_neighbor.geometry)])
                merged_geometry = clean_geometries(merged_geometry)
                
                hru_gdf_utm.at[largest_neighbor.name, 'geometry'] = merged_geometry
                hru_gdf_utm.at[largest_neighbor.name, 'area'] = merged_geometry.area / 1_000_000
                
                # If merging with a different land class, update the land class of the larger HRU
                if largest_neighbor['landClass'] != small_hru['landClass']:
                    hru_gdf_utm.at[largest_neighbor.name, 'landClass'] = small_hru['landClass']
                
                hru_gdf_utm = hru_gdf_utm.drop(idx)
                merged_count += 1
            else:
                # If no neighbors, merge with the closest HRU
                distances = hru_gdf_utm.geometry.distance(small_hru.geometry)
                closest_hru = hru_gdf_utm.loc[distances.idxmin()]
                merged_geometry = unary_union([make_valid(small_hru.geometry), make_valid(closest_hru.geometry)])
                merged_geometry = clean_geometries(merged_geometry)
                
                hru_gdf_utm.at[closest_hru.name, 'geometry'] = merged_geometry
                hru_gdf_utm.at[closest_hru.name, 'area'] = merged_geometry.area / 1_000_000
                
                if closest_hru['landClass'] != small_hru['landClass']:
                    hru_gdf_utm.at[closest_hru.name, 'landClass'] = small_hru['landClass']
                
                hru_gdf_utm = hru_gdf_utm.drop(idx)
                merged_count += 1
        
        hru_gdf_utm['area'] = hru_gdf_utm.geometry.area / 1_000_000
        hru_gdf_utm = hru_gdf_utm.sort_values('area')
    
    print(f"Merged {merged_count} HRUs")
    
    hru_gdf_utm = hru_gdf_utm.reset_index(drop=True)
    hru_gdf_utm['hruNo'] = range(1, len(hru_gdf_utm) + 1)
    hru_gdf_utm['hruId'] = hru_gdf_utm['gruId'] + '_' + hru_gdf_utm['landClass'].astype(str)

    hru_gdf_utm['geometry'] = hru_gdf_utm['geometry'].apply(clean_geometries)
    
    hru_gdf_utm['geometry'] = hru_gdf_utm['geometry'].buffer(0.1).buffer(-0.1)
    hru_gdf_utm['geometry'] = hru_gdf_utm['geometry'].simplify(1)
    
    hru_gdf_utm = hru_gdf_utm[hru_gdf_utm['geometry'].apply(lambda geom: isinstance(geom, (Polygon, MultiPolygon)))]
    
    hru_gdf_utm = hru_gdf_utm[hru_gdf_utm['geometry'].notnull()]

    hru_gdf_merged = hru_gdf_utm.to_crs(hru_gdf.crs)
    hru_gdf_merged['geometry'] = hru_gdf_merged['geometry'].apply(lambda geom: make_valid(geom))
    
    print(f"Finished with {len(hru_gdf_merged)} HRUs")
    print(f"HRUs below threshold after merging: {len(hru_gdf_merged[hru_gdf_merged['area'] < min_hru_size])}")
    
    return hru_gdf_merged

def create_hrus(row, land_raster, unique_land_classes):
    with rasterio.open(land_raster) as src:
        out_image, out_transform = mask(src, [row.geometry], crop=True, all_touched=True)
        out_image = out_image[0]
        
        gru_attributes = row.drop('geometry').to_dict()
        
        hrus = []
        for land_class in unique_land_classes:
            class_mask = (out_image == land_class)
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
                                    'landClass': int(land_class),
                                    **gru_attributes
                                })
        
        return hrus if hrus else [{
            'geometry': row.geometry,
            'gruNo': row.name,
            'gruId': row['gruId'],
            'landClass': -9999,  # No-data value
            **gru_attributes
        }]

def process_hrus(gru_gdf, land_raster, unique_land_classes):
    all_hrus = []
    
    num_cores = max(1, multiprocessing.cpu_count() // 2)
    
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        future_to_row = {executor.submit(create_hrus, row, land_raster, unique_land_classes): row for _, row in gru_gdf.iterrows()}
        for future in as_completed(future_to_row):
            all_hrus.extend(future.result())

    hru_gdf = gpd.GeoDataFrame(all_hrus, crs=gru_gdf.crs)
    
    utm_crs = hru_gdf.estimate_utm_crs()
    hru_gdf_utm = hru_gdf.to_crs(utm_crs)
    hru_gdf_utm['area'] = hru_gdf_utm.area / 1_000_000  # Convert m² to km²

    hru_gdf_utm['area'] = hru_gdf_utm['area'].astype('float64')
    hru_gdf_utm['hruNo'] = range(1, len(hru_gdf_utm) + 1)
    hru_gdf_utm['hruId'] = hru_gdf_utm['gruId'] + '_' + hru_gdf_utm['landClass'].astype(str)
    
    centroids_utm = hru_gdf_utm.geometry.centroid
    centroids_wgs84 = centroids_utm.to_crs(CRS.from_epsg(4326))
    
    hru_gdf_utm['cent_lon'] = centroids_wgs84.x
    hru_gdf_utm['cent_lat'] = centroids_wgs84.y
    
    hru_gdf = hru_gdf_utm.to_crs(CRS.from_epsg(4326))
    
    return hru_gdf

def plot_hrus(hru_gdf, output_file):
    fig, ax = plt.subplots(figsize=(15, 15))
    hru_gdf.plot(column='landClass', cmap='tab20', legend=True, ax=ax)
    ax.set_title('Land-based HRUs')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

def validate_coverage(hru_gdf, gru_gdf, tolerance=1e-6):
    gru_area = gru_gdf.geometry.unary_union.area
    hru_area = hru_gdf.geometry.unary_union.area
    coverage_ratio = hru_area / gru_area
    print(f"Coverage ratio: {coverage_ratio:.6f}")
    assert abs(1 - coverage_ratio) < tolerance, f"Coverage is not complete. Ratio: {coverage_ratio:.6f}"


def main():
    # Read necessary parameters from the control file
    root_path = Path(read_from_control(controlFolder/controlFile, 'root_path'))
    full_domain_name = read_from_control(controlFolder/controlFile, 'full_domain_name')
    domain_name = read_from_control(controlFolder/controlFile, 'domain_name')
    catchment_shp_path = read_from_control(controlFolder/controlFile, 'catchment_shp_path')
    catchment_shp_name = read_from_control(controlFolder/controlFile, 'catchment_shp_name')
    parameter_land_tif_name = read_from_control(controlFolder/controlFile, 'parameter_land_tif_name')
    
    # Construct file paths
    gru_shapefile = make_default_path(controlFolder, controlFile,f'shapefiles/river_basins/{domain_name}_basins.shp')
    land_raster = root_path / f'domain_{full_domain_name}' / 'parameters' / 'landclass' / '7_mode_land_class' / parameter_land_tif_name
    
    if catchment_shp_path == 'default':
        output_shapefile = make_default_path(controlFolder, controlFile,f'shapefiles/catchment/{domain_name}_land_based_HRUs.shp')
    else:
        output_shapefile = Path(catchment_shp_path) / f'{domain_name}_land_based_HRUs.shp'
    
    output_plot = make_default_path(controlFolder, controlFile,f'plots/catchment/{domain_name}_land_based_HRUs.png')
    
    min_hru_size = float(read_from_control(controlFolder/controlFile, 'min_hru_size'))

    print(f"GRU Shapefile: {gru_shapefile}")
    print(f"Land Raster: {land_raster}")
    print(f"Output Shapefile: {output_shapefile}")
    print(f"Output Plot: {output_plot}")
    print(f"Minimum HRU Size: {min_hru_size}")

    # Check if input files exist
    if not gru_shapefile.exists():
        raise FileNotFoundError(f"Input GRU shapefile not found: {gru_shapefile}")
    if not land_raster.exists():
        raise FileNotFoundError(f"Input land raster not found: {land_raster}")

    gru_gdf, unique_land_classes = read_and_prepare_data(gru_shapefile, land_raster)
    hru_gdf = process_hrus(gru_gdf, land_raster, unique_land_classes)

    if hru_gdf is not None and not hru_gdf.empty:
        print(hru_gdf)
        
        hru_gdf = merge_small_hrus(hru_gdf, min_hru_size)
        hru_gdf = ensure_full_coverage(hru_gdf, gru_gdf)
        #validate_coverage(hru_gdf, gru_gdf)
        hru_gdf['geometry'] = hru_gdf['geometry'].apply(clean_geometries)
        hru_gdf = hru_gdf[hru_gdf['geometry'].notnull()]
        
        hru_gdf['hru_type'] = 'land_based'
        
        columns_to_keep = ['geometry', 'gruNo', 'gruId', 'landClass', 'area', 'hruNo', 'hruId', 'cent_lon', 'cent_lat', 'COMID', 'hru_type']
        hru_gdf = hru_gdf[columns_to_keep]
        
        valid_polygons = []
        for idx, row in hru_gdf.iterrows():
            geom = row['geometry']
            if isinstance(geom, (Polygon, MultiPolygon)) and geom.is_valid:
                valid_polygons.append(row)
            else:
                print(f"Removing invalid geometry for HRU {idx}")
        
        hru_gdf = gpd.GeoDataFrame(valid_polygons, crs=hru_gdf.crs)

        hru_gdf['geometry'] = to_polygon(hru_gdf['geometry'])     
        hru_gdf['geometry'] = hru_gdf['geometry'].apply(clean_geometries)
        hru_gdf = hru_gdf[hru_gdf['geometry'].notnull()]
        hru_gdf = hru_gdf[hru_gdf['geometry'].is_valid]

        hru_gdf.to_file(output_shapefile)
        print(f"Land-based HRU Shapefile created with {len(hru_gdf)} HRUs and saved to {output_shapefile}")

        plot_hrus(hru_gdf, output_plot)
        print(f"Land-based HRU plot saved to {output_plot}")
    else:
        print("Error: No valid HRUs were created. Check your input data and parameters.")

if __name__ == "__main__":
    main()