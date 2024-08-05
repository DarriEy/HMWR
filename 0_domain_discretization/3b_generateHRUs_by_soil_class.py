import geopandas as gpd
import pandas as pd
import rasterio
from rasterio.mask import mask
import numpy as np
from shapely.geometry import shape, Polygon, MultiPolygon, mapping
from shapely.ops import unary_union
import matplotlib.pyplot as plt
from pathlib import Path
from rasterio.features import rasterize
from rasterio.enums import Resampling
from tqdm import tqdm
from rasterio.warp import transform_geom
from rasterstats import zonal_stats
from scipy.ndimage import median_filter
from pyproj import CRS
from geopandas import GeoDataFrame
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.control_file_utils import read_from_control, make_default_path

# Easy access to control file folder
controlFolder = Path('../0_control_files')
controlFile = 'control_active.txt'

def plot_hrus(hru_gdf, output_file):
    fig, ax = plt.subplots(figsize=(15, 15))
    hru_gdf.plot(column='soilClass', cmap='viridis', legend=True, ax=ax)
    ax.set_title('Soil-based HRUs')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

def rasterize_vector(invector, infield, infield_dtype, refraster, outraster):
    in_gdf = gpd.read_file(invector)    
    
    with rasterio.open(refraster) as src:                 
        ref_mask = src.read_masks(1)
        meta = src.meta.copy()
        nodatavals = src.nodatavals
    meta.update(count=1, dtype=infield_dtype, compress='lzw') 
    
    with rasterio.open(outraster, 'w+', **meta) as out:
        out_arr = out.read(1)
        shapes = ((geom,value) for geom, value in zip(in_gdf['geometry'], in_gdf[infield]))  
        burned = rasterio.features.rasterize(shapes=shapes, out=out_arr, fill=nodatavals, transform=out.transform)
        burned_ma = np.ma.masked_array(burned, ref_mask==0)
        out.write(burned_ma,1)    
    return

def classify_raster(inraster, bound_raster, classif_trigger, bins, class_outraster, value_outraster):
    with rasterio.open(inraster) as ff:
        data  = ff.read(1)
        data_mask = ff.read_masks(1)
        out_meta = ff.meta.copy()

    with rasterio.open(bound_raster) as ff:
        bounds  = ff.read(1)
        bounds_mask = ff.read_masks(1)
    unique_bounds = np.unique(bounds[bounds_mask!=0])

    data_class = data.copy()
    data_value = data.copy()

    if isinstance(bins, str) and bins == 'median':
        for bound in unique_bounds:
            smask = bounds == bound            
            smin, smedian, smax = np.min(data[smask]), np.median(data[smask]), np.max(data[smask])
            if (classif_trigger is None) or ((classif_trigger is not None) and ((smax-smin)>=classif_trigger)):
                bins = [smin,smedian,smax]
            else:
                bins = [smin,smax]
            (hist,bin_edges) = np.histogram(data[smask],bins=bins)            
            for ibin in np.arange(len(bin_edges)-1):
                if ibin != (len(bin_edges)-2):
                    smask = (bounds == bound) & (data >= bin_edges[ibin]) & (data < bin_edges[ibin+1])
                else: 
                    smask = (bounds == bound) & (data >= bin_edges[ibin]) & (data <= bin_edges[ibin+1])
                data_class[smask] = ibin+1
                data_value[smask] = np.mean(data[smask])     

    data_class = data_class.astype('int32') 
    out_meta.update(count=1, dtype='int32', compress='lzw')
    with rasterio.open(class_outraster, 'w', **out_meta) as outf:
        data_class_ma = np.ma.masked_array(data_class,data_mask==0)
        outf.write(data_class_ma, 1)    

    data_value = data_value.astype('float64') 
    out_meta.update(count=1, dtype='float64', compress='lzw')
    with rasterio.open(value_outraster, 'w', **out_meta) as outf:
        data_value_ma = np.ma.masked_array(data_value,data_mask==0) 
        outf.write(data_value_ma, 1)
    return

def fill_empty_spaces(hru_gdf, domain_gdf, min_area=0.001):
    # Create a union of all HRUs
    hru_union = unary_union(hru_gdf.geometry)
    
    # Find empty spaces
    domain = domain_gdf.geometry.unary_union
    empty_spaces = domain.difference(hru_union)
    
    # Convert empty spaces to GeoDataFrame
    empty_gdf = gpd.GeoDataFrame(geometry=gpd.GeoSeries(list(empty_spaces) if isinstance(empty_spaces, MultiPolygon) else [empty_spaces]))
    empty_gdf = empty_gdf[empty_gdf.area >= min_area]  # Filter out very small spaces
    
    for idx, empty_space in empty_gdf.iterrows():
        # Find adjacent HRUs
        adjacent_hrus = hru_gdf[hru_gdf.geometry.touches(empty_space.geometry)]
        
        if not adjacent_hrus.empty:
            # Determine dominant adjacent HRU (by area)
            dominant_hru = adjacent_hrus.loc[adjacent_hrus.area.idxmax()]
            
            # Expand dominant HRU to include empty space
            new_geometry = dominant_hru.geometry.union(empty_space.geometry)
            hru_gdf.loc[dominant_hru.name, 'geometry'] = new_geometry
    
    # Recalculate areas
    hru_gdf['area'] = hru_gdf.geometry.area
    
    return hru_gdf

def fill_gaps_in_hrus(hru_gdf, gru_gdf, buffer_distance=1):
    # Ensure the HRUs and GRUs are in the same CRS
    if hru_gdf.crs != gru_gdf.crs:
        gru_gdf = gru_gdf.to_crs(hru_gdf.crs)
    
    # Reset index to ensure unique identifiers
    hru_gdf = hru_gdf.reset_index(drop=True)
    
    for idx, gru in gru_gdf.iterrows():
        print(f"Processing GRU {idx + 1}/{len(gru_gdf)}")
        
        # Get HRUs within this GRU
        gru_hrus = hru_gdf[hru_gdf['gruId'] == gru['gruId']]
        
        if gru_hrus.empty:
            print(f"No HRUs found for GRU {gru['gruId']}")
            continue
        
        # Find gaps within this GRU
        hru_union = unary_union(gru_hrus.geometry)
        gaps = gru.geometry.difference(hru_union)
        
        if gaps.is_empty:
            continue
        
        # Process each gap
        for gap in gaps.geoms if hasattr(gaps, 'geoms') else [gaps]:
            # Use a small buffer to find nearby HRUs
            gap_buffer = gap.buffer(buffer_distance)
            nearby_hrus = gru_hrus[gru_hrus.geometry.intersects(gap_buffer)]
            
            if nearby_hrus.empty:
                print(f"No nearby HRUs found for a gap in GRU {gru['gruId']}. Increasing buffer.")
                buffer_multiplier = 2
                while nearby_hrus.empty and buffer_multiplier <= 10:
                    gap_buffer = gap.buffer(buffer_distance * buffer_multiplier)
                    nearby_hrus = gru_hrus[gru_hrus.geometry.intersects(gap_buffer)]
                    buffer_multiplier += 1
                
                if nearby_hrus.empty:
                    print(f"Still no nearby HRUs found. Creating a new HRU for this gap.")
                    new_hru = gru_hrus.iloc[0].copy()
                    new_hru.geometry = gap
                    new_hru['area'] = gap.area
                    new_hru['soilClass'] = gru_hrus['soilClass'].mode().iloc[0]  # Use the most common soil class
                    hru_gdf = hru_gdf.append(new_hru, ignore_index=True)
                    continue
            
            # Choose the largest nearby HRU to fill the gap
            largest_hru_idx = nearby_hrus.area.idxmax()
            
            # Expand the largest HRU to include the gap
            new_geometry = nearby_hrus.loc[largest_hru_idx, 'geometry'].union(gap)
            
            # Update the HRU in the main dataframe
            hru_gdf.loc[largest_hru_idx, 'geometry'] = new_geometry
    
    # Recalculate areas
    hru_gdf['area'] = hru_gdf.geometry.area
    
    # Remove any potential slivers or tiny polygons
    hru_gdf = hru_gdf[hru_gdf.area > np.finfo(float).eps]
    
    return hru_gdf

def smooth_raster(raster_path, output_path, filter_size=4):
    with rasterio.open(raster_path) as src:
        soil_data = src.read(1)
        smoothed_soil_data = median_filter(soil_data, size=filter_size)
        
        profile = src.profile
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(smoothed_soil_data, 1)
    
    return output_path


def eliminate_small_hrus(hru_gdf, min_area_km2, area_field='area', max_iterations=100):
    min_area = min_area_km2 * 1e6  # Convert km² to m²
    iteration = 0
    total_hrus = len(hru_gdf)
    previous_hru_count = total_hrus + 1

    while iteration < max_iterations:
        small_hrus = hru_gdf[hru_gdf[area_field] < min_area]
        print(f"Iteration {iteration}: {len(small_hrus)} small HRUs to eliminate out of {len(hru_gdf)} total HRUs")
        
        if len(small_hrus) == 0 or len(hru_gdf) <= len(hru_gdf['gruId'].unique()):
            break
        
        if len(hru_gdf) == previous_hru_count:
            print("No changes in the last iteration. Stopping elimination process.")
            break
        
        previous_hru_count = len(hru_gdf)
        
        for idx, small_hru in small_hrus.iterrows():
            neighbors = hru_gdf[(hru_gdf.geometry.touches(small_hru.geometry)) & (hru_gdf['gruId'] == small_hru['gruId'])]
            neighbors = neighbors[neighbors.index != idx]
            
            if len(neighbors) > 0:
                same_soil_neighbors = neighbors[neighbors['soilClass'] == small_hru['soilClass']]
                if len(same_soil_neighbors) > 0:
                    merge_candidate_idx = same_soil_neighbors[area_field].idxmax()
                else:
                    merge_candidate_idx = neighbors[area_field].idxmax()
                
                merged_geometry = small_hru.geometry.union(hru_gdf.loc[merge_candidate_idx, 'geometry'])
                hru_gdf.loc[merge_candidate_idx, 'geometry'] = merged_geometry
                hru_gdf.loc[merge_candidate_idx, area_field] = merged_geometry.area
                hru_gdf = hru_gdf.drop(idx)
                break
        
        hru_gdf[area_field] = hru_gdf.geometry.area
        iteration += 1
    
    print(f"Small HRU elimination completed after {iteration} iterations")
    print(f"Reduced from {total_hrus} to {len(hru_gdf)} HRUs")
    return hru_gdf

def fill_gaps_in_hrus(hru_gdf, gru_gdf, buffer_distance=0.1):
    # Ensure the HRUs and GRUs are in the same CRS
    if hru_gdf.crs != gru_gdf.crs:
        gru_gdf = gru_gdf.to_crs(hru_gdf.crs)
    
    # Reset index to ensure unique identifiers
    hru_gdf = hru_gdf.reset_index(drop=True)
    
    for idx, gru in gru_gdf.iterrows():
        print(f"Processing GRU {idx + 1}/{len(gru_gdf)}")
        
        # Get HRUs within this GRU
        gru_hrus = hru_gdf[hru_gdf['gruId'] == gru['gruId']]
        
        if gru_hrus.empty:
            print(f"No HRUs found for GRU {gru['gruId']}")
            continue
        
        # Find gaps within this GRU
        hru_union = unary_union(gru_hrus.geometry)
        gaps = gru.geometry.difference(hru_union)
        
        if gaps.is_empty:
            continue
        
        # Process each gap
        for gap in gaps.geoms if hasattr(gaps, 'geoms') else [gaps]:
            # Use a small buffer to find nearby HRUs
            gap_buffer = gap.buffer(buffer_distance)
            nearby_hrus = gru_hrus[gru_hrus.geometry.intersects(gap_buffer)]
            
            if nearby_hrus.empty:
                print(f"No nearby HRUs found for a gap in GRU {gru['gruId']}")
                continue
            
            # Choose the largest nearby HRU to fill the gap
            largest_hru_idx = nearby_hrus.area.idxmax()
            
            # Expand the largest HRU to include the gap
            new_geometry = nearby_hrus.loc[largest_hru_idx, 'geometry'].union(gap)
            
            # Update the HRU in the main dataframe
            hru_gdf.loc[largest_hru_idx, 'geometry'] = new_geometry
    
    # Recalculate areas
    hru_gdf['area'] = hru_gdf.geometry.area
    
    # Remove any potential slivers or tiny polygons
    hru_gdf = hru_gdf[hru_gdf.area > np.finfo(float).eps]
    
    return hru_gdf

def create_hrus_for_gru(gru, soil_raster, min_area, gru_crs):
    gru_geometry = [gru.geometry]
    with rasterio.open(soil_raster) as src:
        out_image, out_transform = mask(src, gru_geometry, crop=True, all_touched=True)
        out_meta = src.meta.copy()
        out_meta.update({"height": out_image.shape[1], "width": out_image.shape[2], "transform": out_transform})

    unique_soils = np.unique(out_image[out_image != src.nodata])
    print(f"Unique soil classes in GRU: {unique_soils}")
    hrus = []

    for soil_class in unique_soils:
        soil_mask = out_image[0] == soil_class
        hru_shapes = rasterio.features.shapes(soil_mask.astype('uint8'), mask=soil_mask, transform=out_transform)
        
        for geom, _ in hru_shapes:
            hru = gru.copy()
            hru_geom = shape(geom).intersection(gru.geometry)
            if not hru_geom.is_empty:
                hru.geometry = hru_geom
                hru['soilClass'] = int(soil_class)
                hru['gruId'] = gru['gruId'] if 'gruId' in gru.index else gru.name
                hrus.append(hru)

    # Create a temporary GeoDataFrame and set its CRS
    temp_gdf = GeoDataFrame(hrus, crs=gru_crs)
    temp_gdf = reproject_to_utm(temp_gdf)
    
    # Filter based on area
    hrus = [hru for hru, area in zip(hrus, temp_gdf.geometry.area) if area >= min_area]
    
    print(f"Created {len(hrus)} HRUs for this GRU")

    # If no HRUs were created, create a single HRU for the entire GRU
    if not hrus:
        print("No HRUs created, using entire GRU as a single HRU")
        hru = gru.copy()
        hru['soilClass'] = int(unique_soils[0]) if len(unique_soils) > 0 else 0
        hru['gruId'] = gru['gruId'] if 'gruId' in gru.index else gru.name
        hrus.append(hru)

    return hrus

def reproject_to_utm(gdf):
    # Check if the GDF is already in a projected CRS
    if gdf.crs and gdf.crs.is_projected:
        return gdf

    # Get the centroid of the entire dataset
    centroid = gdf.geometry.unary_union.centroid
    
    # Determine the UTM zone
    utm_zone = int((centroid.x + 180) / 6) + 1
    hemisphere = 'north' if centroid.y >= 0 else 'south'
    
    # Create the UTM CRS
    utm_crs = CRS.from_dict({'proj': 'utm', 'zone': utm_zone, 'hemisphere': hemisphere})
    
    # Reproject the GeoDataFrame
    return gdf.to_crs(utm_crs)

def ensure_complete_hru_coverage(hru_gdf, gru_gdf):
    # Ensure the HRUs and GRUs are in the same CRS
    if hru_gdf.crs != gru_gdf.crs:
        gru_gdf = gru_gdf.to_crs(hru_gdf.crs)
    
    new_hrus = []
    
    for idx, gru in gru_gdf.iterrows():
        print(f"Processing GRU {idx + 1}/{len(gru_gdf)}")
        
        # Get HRUs within this GRU
        gru_hrus = hru_gdf[hru_gdf['gruId'] == gru['gruId']]
        
        if gru_hrus.empty:
            print(f"No HRUs found for GRU {gru['gruId']}. Creating new HRU.")
            new_hru = gpd.GeoDataFrame({'geometry': [gru.geometry], 'gruId': [gru['gruId']], 'soilClass': [0]}, crs=gru_gdf.crs)
            new_hrus.append(new_hru)
            continue
        
        # Find uncovered areas within this GRU
        hru_union = unary_union(gru_hrus.geometry)
        uncovered = gru.geometry.difference(hru_union)
        
        if not uncovered.is_empty:
            print(f"Found uncovered area in GRU {gru['gruId']}. Creating new HRU(s).")
            if isinstance(uncovered, (Polygon, MultiPolygon)):
                uncovered = [uncovered]
            for geom in uncovered:
                new_hru = gpd.GeoDataFrame({
                    'geometry': [geom],
                    'gruId': [gru['gruId']],
                    'soilClass': [gru_hrus['soilClass'].mode().iloc[0]]  # Use most common soil class
                }, crs=gru_gdf.crs)
                new_hrus.append(new_hru)
    
    # Combine original HRUs with new ones
    if new_hrus:
        new_hrus_gdf = gpd.GeoDataFrame(pd.concat(new_hrus, ignore_index=True), crs=hru_gdf.crs)
        hru_gdf = pd.concat([hru_gdf, new_hrus_gdf], ignore_index=True)
    
    # Recalculate areas
    hru_gdf['area'] = hru_gdf.geometry.area
    
    return hru_gdf

def dissolve_same_class_hrus(hru_gdf):
    return hru_gdf.dissolve(by=['gruId', 'soilClass']).reset_index()

def clean_hrus(hru_gdf, min_area=1):  # min_area in square meters
    hru_gdf['area'] = hru_gdf.geometry.area
    small_hrus = hru_gdf[hru_gdf['area'] < min_area]
    
    for idx, small_hru in small_hrus.iterrows():
        neighbors = hru_gdf[hru_gdf.geometry.touches(small_hru.geometry) & (hru_gdf.index != idx)]
        if not neighbors.empty:
            largest_neighbor = neighbors.loc[neighbors['area'].idxmax()]
            hru_gdf.loc[largest_neighbor.name, 'geometry'] = largest_neighbor.geometry.union(small_hru.geometry)
            hru_gdf = hru_gdf.drop(idx)
    
    hru_gdf['area'] = hru_gdf.geometry.area
    return hru_gdf

def main():
    # Read necessary parameters from the control file
    root_path = Path(read_from_control(controlFolder/controlFile, 'root_path'))
    domain_name = read_from_control(controlFolder/controlFile, 'domain_name')
    catchment_shp_path = read_from_control(controlFolder/controlFile, 'catchment_shp_path')
    catchment_shp_name = read_from_control(controlFolder/controlFile, 'catchment_shp_name')
    parameter_soil_tif_name = read_from_control(controlFolder/controlFile, 'parameter_soil_tif_name')
    
    # Construct file paths
    gru_shapefile = make_default_path(controlFolder, controlFile, f'shapefiles/river_basins/{domain_name}_basins.shp')
    soil_raster = root_path / f'domain_{domain_name}' / 'parameters' / 'soilclass' / '2_soil_classes_domain' / parameter_soil_tif_name
    
    if catchment_shp_path == 'default':
        output_shapefile = make_default_path(controlFolder, controlFile, f'shapefiles/catchment/{domain_name}_soil_based_HRUs.shp')
    else:
        output_shapefile = Path(catchment_shp_path) / f'{domain_name}_soil_based_HRUs.shp'
    
    output_plot = make_default_path(controlFolder, controlFile, f'plots/catchment/{domain_name}_soil_based_HRUs.png')
    
    min_hru_size = float(read_from_control(controlFolder/controlFile, 'min_hru_size'))

     # Read the GRU shapefile
    gru_gdf = gpd.read_file(gru_shapefile)
    print(f"GRU shapefile CRS: {gru_gdf.crs}")
    print(f"Number of GRUs: {len(gru_gdf)}")
    print(f"GRU bounding box: {gru_gdf.total_bounds}")

    # Read the soil raster
    with rasterio.open(soil_raster) as src:
        soil_data = src.read(1)
        soil_meta = src.meta
        print(f"Soil raster CRS: {src.crs}")
        print(f"Soil raster bounds: {src.bounds}")
        print(f"Soil raster shape: {soil_data.shape}")

        # Reproject GRU geometries to match the soil raster CRS
        gru_gdf = gru_gdf.to_crs(src.crs)
        print(f"Reprojected GRU shapefile CRS: {gru_gdf.crs}")
        print(f"Reprojected GRU bounding box: {gru_gdf.total_bounds}")

    # Smooth the soil raster
    smoothed_soil_raster = smooth_raster(soil_raster, soil_raster.parent / 'smoothed_soil.tif')

    # Ensure gruId is present in gru_gdf
    if 'gruId' not in gru_gdf.columns:
        gru_gdf['gruId'] = gru_gdf.index

    # Create HRUs based on soil classes
    hru_list = []
    for idx, gru in gru_gdf.iterrows():
        print(f"\nProcessing GRU {idx + 1}/{len(gru_gdf)}")
        gru_hrus = create_hrus_for_gru(gru, smoothed_soil_raster, min_hru_size * 1e6, gru_gdf.crs)  # Convert km² to m²
        hru_list.extend(gru_hrus)

    # Create GeoDataFrame from HRU list and set its CRS
    hru_gdf = GeoDataFrame(hru_list, crs=gru_gdf.crs)
    
    # Reproject to UTM for accurate area calculation
    hru_gdf = reproject_to_utm(hru_gdf)
    
    # Calculate area for each HRU
    hru_gdf['area'] = hru_gdf.geometry.area

    print(f"Number of HRUs before elimination: {len(hru_gdf)}")

    # Eliminate small HRUs
    hru_gdf = eliminate_small_hrus(hru_gdf, min_hru_size)

    print(f"Number of HRUs after elimination: {len(hru_gdf)}")

    # After creating and processing HRUs
    print("Filling gaps in HRUs...")
    hru_gdf = ensure_complete_hru_coverage(hru_gdf, gru_gdf)
    hru_gdf = dissolve_same_class_hrus(hru_gdf)
    hru_gdf = clean_hrus(hru_gdf)
    
    # Reset index and assign unique HRU IDs
    hru_gdf = hru_gdf.reset_index(drop=True)
    hru_gdf['hruId'] = hru_gdf.index + 1

    # Save the final HRU shapefile
    hru_gdf.to_file(output_shapefile)
    print(f"Final HRU Shapefile: {output_shapefile}")

 

    # Plot the HRUs
    plot_hrus(hru_gdf, output_plot)
    print(f"HRU Plot: {output_plot}")

    # Print summary statistics
    print("\nHRU Summary:")
    print(f"Total number of HRUs: {len(hru_gdf)}")
    print("\nSoil class distribution:")
    soil_class_counts = hru_gdf['soilClass'].value_counts()
    for soil_class, count in soil_class_counts.items():
        print(f"  Soil class {soil_class}: {count} HRUs")

if __name__ == "__main__":
    main()