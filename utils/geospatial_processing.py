import geopandas as gpd # type: ignore
from pathlib import Path
import rasterio # type: ignore
from rasterio.mask import mask # type: ignore
import numpy as np # type: ignore
from pyproj import CRS # type: ignore
import pysheds # type: ignore
from pysheds.grid import Grid # type: ignore
from rasterio.features import shapes # type: ignore
import logging
import geopandas as gpd # type: ignore
import networkx as nx # type: ignore
from shapely.geometry import Point, Polygon, MultiPolygon, LineString, shape, mapping # type: ignore
import math
from shapely.ops import unary_union, polygonize # type: ignore
from shapely.validation import make_valid # type: ignore
import matplotlib.pyplot as plt # type: ignore
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import pvlib # type: ignore
import pandas as pd # type: ignore
import os 
import whitebox # type: ignore
import inspect
from rasterio import features # type: ignore

def create_pour_point_shapefile(config, logger):
    logger.info("Creating pour point shapefile")
    
    if config.pour_point_coords.lower() == 'default':
        logger.info("Using user-provided pour point shapefile")
        return None
    
    try:
        lat, lon = map(float, config.pour_point_coords.split('/'))
        point = Point(lon, lat)
        gdf = gpd.GeoDataFrame({'geometry': [point]}, crs="EPSG:4326")
        
        if config.pour_point_shp_path == 'default':
            output_path = Path(config.root_path) / f"domain_{config.domain_name}" / "shapefiles" / "pour_point"
        else:
            output_path = Path(config.pour_point_shp_path)
        
        if config.pour_point_shp_name == 'default':
            config.pour_point_shp_name = f"{config.domain_name}_pourPoint.shp"
        
        output_path.mkdir(parents=True, exist_ok=True)
        output_file = output_path / config.pour_point_shp_name
        
        gdf.to_file(output_file)
        logger.info(f"Pour point shapefile created successfully: {output_file}")
        return output_file
    except ValueError:
        logger.error("Invalid pour point coordinates format. Expected 'lat/lon'.")
    except Exception as e:
        logger.error(f"Error creating pour point shapefile: {str(e)}")
    
    return None

def run_whitebox_tool(tool_name, arguments):
    wbt = whitebox.WhiteboxTools()
    
    try:
        tool_function = getattr(wbt, tool_name)
        
        # Get the function signature
        sig = inspect.signature(tool_function)
        
        # Prepare arguments
        args = []
        kwargs = {}
        for arg in arguments:
            if '=' in arg:
                key, value = arg.split('=')
                key = key.lstrip('-')  # Remove leading dashes
                if key in sig.parameters:
                    kwargs[key] = value
                else:
                    print(f"Warning: Argument '{key}' not found in function signature for {tool_name}")
            else:
                args.append(arg)
        
        # Print function signature and provided arguments for debugging
        print(f"Function signature for {tool_name}: {sig}")
        print(f"Provided args: {args}")
        print(f"Provided kwargs: {kwargs}")
        
        # Call the function with the prepared arguments
        result = tool_function(*args, **kwargs)
        
        # Print the result for debugging
        print(f"Result from {tool_name}: {result}")
        
        # Check if the result is an error code
        if isinstance(result, int) and result != 0:
            raise RuntimeError(f"WhiteboxTools {tool_name} failed with return code {result}")
        
        return result
    except AttributeError as e:
        raise RuntimeError(f"WhiteboxTools doesn't have a tool named {tool_name}: {str(e)}")
    except Exception as e:
        raise RuntimeError(f"WhiteboxTools {tool_name} failed: {str(e)}")

def visualize_raster(raster_path, title, logger):
    try:
        with rasterio.open(raster_path) as src:
            data = src.read(1)
            
        plt.figure(figsize=(10, 10))
        plt.imshow(data, cmap='viridis')
        plt.colorbar(label='Value')
        plt.title(title)
        plt.axis('off')
        
        output_dir = Path(raster_path).parent / "visualizations"
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f"{title.replace(' ', '_').lower()}.png"
        plt.savefig(output_path)
        plt.close()
        
        logger.info(f"Visualization saved: {output_path}")
    except Exception as e:
        logger.error(f"Failed to visualize {title}: {str(e)}")

def validate_dem(dem_path, logger):
    try:
        with rasterio.open(dem_path) as src:
            data = src.read(1)
            if np.all(data == src.nodata):
                raise ValueError("DEM contains only no-data values")
            if np.all(data == 0):
                raise ValueError("DEM contains only zero values")
            logger.info(f"DEM statistics - Min: {np.min(data)}, Max: {np.max(data)}, Mean: {np.mean(data)}")
    except Exception as e:
        logger.error(f"Failed to validate DEM: {str(e)}")
        raise

def delineate_grus(config, logger):
    logger.info("Delineating GRUs, sub-basins, and river network")
    
    try:
        dem_raster = Path(config.root_path) / f"domain_{config.domain_name}" / "parameters" / "dem" / "5_elevation" / config.parameter_dem_tif_name
        pour_point_shapefile = config.pour_point_shapefile_path
        output_gru_shapefile = Path(config.root_path) / f"domain_{config.domain_name}" / "shapefiles" / "river_basins" / f"{config.domain_name}_delineated_GRUs.shp"
        output_subbasin_shapefile = Path(config.root_path) / f"domain_{config.domain_name}" / "shapefiles" / "river_basins" / f"{config.domain_name}_delineated_subbasins.shp"
        output_river_shapefile = Path(config.root_path) / f"domain_{config.domain_name}" / "shapefiles" / "river_network" / f"{config.domain_name}_river_network.shp"
        
        # Set the working directory
        work_dir = Path(config.root_path) / f"domain_{config.domain_name}" / "temp"
        work_dir.mkdir(parents=True, exist_ok=True)
        
        # Set Whitebox working directory
        wbt = whitebox.WhiteboxTools()
        wbt.set_working_dir(str(work_dir))
        
        # Define output files
        filled_dem = work_dir / "filled_dem.tif"
        breached_dem = work_dir / "breached_dem.tif"
        flow_dir = work_dir / "flow_dir.tif"
        flow_acc = work_dir / "flow_acc.tif"
        streams = work_dir / "streams.tif"
        stream_order = work_dir / "stream_order.tif"
        subbasins = work_dir / "subbasins.tif"
        snap_pour_points = work_dir / "snap_pour_points.shp"
        
        validate_dem(dem_raster, logger)

        logger.info("Filling depressions...")
        run_whitebox_tool("fill_depressions", [f"{dem_raster}", f"output={filled_dem}", "fix_flats=True"])
        
        # Visualize filled DEM
        visualize_raster(filled_dem, "Filled DEM", logger)

        logger.info("Breaching depressions...")
        run_whitebox_tool("breach_depressions", [f"{filled_dem}", f"output={breached_dem}", "max_depth=100", "max_length=100"])
        
        # Visualize breached DEM
        visualize_raster(breached_dem, "Breached DEM", logger)

        logger.info("Calculating flow direction...")
        run_whitebox_tool("d8_pointer", [f"{breached_dem}", f"output={flow_dir}"])
        
        # Visualize flow direction
        visualize_raster(flow_dir, "Flow Direction", logger)

        logger.info("Calculating flow accumulation...")
        run_whitebox_tool("d8_flow_accumulation", [f"{flow_dir}", f"output={flow_acc}", "out_type=cells"])
        
        # Visualize flow accumulation
        visualize_raster(flow_acc, "Flow Accumulation", logger)

        # Add a step to check flow accumulation statistics
        logger.info("Checking flow accumulation statistics...")
        flow_acc_stats = run_whitebox_tool("raster_summary_stats", [f"{flow_acc}"])
        logger.info(f"Flow accumulation statistics: {flow_acc_stats}")
        
        # If flow_acc_stats is 0 or very low, try an alternative method
        if isinstance(flow_acc_stats, int) and flow_acc_stats == 0:
            logger.warning("Flow accumulation failed, trying an alternative method...")
            run_whitebox_tool("fill_depressions", [f"{dem_raster}", f"output={filled_dem}", "fix_flats=True", "flat_increment=0.01"])
            run_whitebox_tool("d8_pointer", [f"{filled_dem}", f"output={flow_dir}"])
            run_whitebox_tool("d8_flow_accumulation", [f"{flow_dir}", f"output={flow_acc}", "out_type=cells"])
            flow_acc_stats = run_whitebox_tool("raster_summary_stats", [f"{flow_acc}"])
            logger.info(f"New flow accumulation statistics: {flow_acc_stats}")
            
            # Visualize new flow accumulation
            visualize_raster(flow_acc, "New Flow Accumulation", logger)

        # If we still don't have valid statistics, raise an error
        if isinstance(flow_acc_stats, int) and flow_acc_stats == 0:
            raise RuntimeError(f"Failed to obtain valid flow accumulation statistics: {flow_acc_stats}")

        logger.info("Extracting streams...")
        run_whitebox_tool("extract_streams", [f"{flow_acc}", f"output={streams}", "threshold=1000"])
        
        logger.info("Calculating stream order...")
        run_whitebox_tool("strahler_stream_order", [f"{streams}", f"{flow_dir}", f"output={stream_order}"])
        
        logger.info("Delineating sub-basins...")
        run_whitebox_tool("subbasins", [f"{flow_dir}", f"{streams}", f"output={subbasins}"])
        
        logger.info("Converting sub-basins to vector...")
        run_whitebox_tool("raster_to_vector_polygons", [f"{subbasins}", f"output={output_subbasin_shapefile}"])
        
        logger.info("Extracting vector stream network...")
        run_whitebox_tool("raster_streams_to_vector", [f"{streams}", f"{flow_dir}", f"output={output_river_shapefile}"])
        
        logger.info("Snapping pour points...")
        if os.path.exists(pour_point_shapefile) and os.path.getsize(pour_point_shapefile) > 0:
            if os.path.exists(streams) and os.path.getsize(streams) > 0:
                run_whitebox_tool("jenson_snap_pour_points", [f"{pour_point_shapefile}", f"{streams}", f"output={snap_pour_points}", "snap_dist=1000"])
            else:
                logger.error(f"Streams file is missing or empty: {streams}")
        else:
            logger.error(f"Pour points file is missing or empty: {pour_point_shapefile}")
        
        logger.info("Delineating main watershed...")
        run_whitebox_tool("watershed", [f"{flow_dir}", f"{snap_pour_points}", f"output={output_gru_shapefile}"])
        
        logger.info(f"Delineated GRU shapefile saved to: {output_gru_shapefile}")
        logger.info(f"Delineated sub-basin shapefile saved to: {output_subbasin_shapefile}")
        logger.info(f"River network shapefile saved to: {output_river_shapefile}")
        
        return output_gru_shapefile, output_subbasin_shapefile, output_river_shapefile

    except Exception as e:
        logger.error(f"An error occurred during GRU delineation: {str(e)}", exc_info=True)
        return None, None, None

def subset_merit_hydrofabrics(config, logger):
    logger.info("Subsetting MERIT Hydrofabrics")
    output_basins_path = Path(config.root_path) / f"domain_{config.domain_name}" / "shapefiles" / "river_basins" / f"{config.domain_name}_basins.shp"
    output_rivers_path = Path(config.root_path) / f"domain_{config.domain_name}" / "shapefiles" / "river_network" / f"{config.domain_name}_network.shp"

    if not output_basins_path.exists():
        try:
            fulldom_basins = config.root_path / f"domain_{config.full_domain_name}" / 'shapefiles' / 'catchment' / config.fullDom_basins_name
            fulldom_rivers = config.root_path / f"domain_{config.full_domain_name}" / 'shapefiles' / 'river_network' / config.fullDom_rivers_name

            basins = gpd.read_file(fulldom_basins).set_crs('epsg:4326')
            rivers = gpd.read_file(fulldom_rivers).set_crs('epsg:4326')

            pour_point_shp = gpd.read_file(config.pour_point_shapefile_path)

            if pour_point_shp.crs != basins.crs:
                pour_point_shp = pour_point_shp.to_crs(basins.crs)
            
            containing_basin = gpd.sjoin(pour_point_shp, basins, how='left', predicate='within')
            if containing_basin.empty:
                raise ValueError("No basin contains the given pour point.")
            downstream_basin_id = containing_basin.iloc[0]['COMID']

            river_graph = nx.DiGraph()
            for _, row in rivers.iterrows():
                current_basin = row['COMID']
                for up_col in ['up1', 'up2', 'up3']:
                    upstream_basin = row[up_col]
                    if upstream_basin != -9999:
                        river_graph.add_edge(upstream_basin, current_basin)

            upstream_basin_ids = nx.ancestors(river_graph, downstream_basin_id)
            upstream_basin_ids.add(downstream_basin_id)

            subset_basins = basins[basins['COMID'].isin(upstream_basin_ids)].copy()
            subset_rivers = rivers[rivers['COMID'].isin(upstream_basin_ids)].copy()

            subset_basins['hru_to_seg'] = subset_basins['COMID']
            subset_basins = prepare_basin_attributes(subset_basins)

            subset_rivers = prepare_river_attributes(subset_rivers)

            subset_basins.to_file(output_basins_path)
            subset_rivers.to_file(output_rivers_path)

            logger.info(f"Subset basins shapefile saved to: {output_basins_path}")
            logger.info(f"Subset rivers shapefile saved to: {output_rivers_path}")

        except Exception as e:
            logger.error(f"Error subsetting MERIT Hydrofabrics: {str(e)}")
    else:
        logger.info('Output shapefile already exists, carrying on')
    
    return output_basins_path, output_rivers_path

def subset_TDX_hydrofabrics(config, logger):
    logger.info("Subsetting TDX Hydrofabrics")
    output_basins_path = Path(config.root_path) / f"domain_{config.domain_name}" / "shapefiles" / "river_basins" / f"{config.domain_name}_TDX_basins.shp"
    output_rivers_path = Path(config.root_path) / f"domain_{config.domain_name}" / "shapefiles" / "river_network" / f"{config.domain_name}_TDX_network.shp"

    if not output_basins_path.exists():
        try:
            fulldom_basins = config.root_path / f"domain_{config.full_domain_name}" / 'shapefiles' / 'catchment' / config.fullDom_basins_name
            fulldom_rivers = config.root_path / f"domain_{config.full_domain_name}" / 'shapefiles' / 'river_network' / config.fullDom_rivers_name

            basins = gpd.read_file(fulldom_basins).set_crs('epsg:4326')
            rivers = gpd.read_file(fulldom_rivers).set_crs('epsg:4326')

            pour_point_shp = gpd.read_file(config.pour_point_shapefile_path)

            containing_basins = gpd.sjoin(pour_point_shp, basins, how='left', predicate='within')
            if containing_basins.empty:
                raise ValueError("No basin contains the given pour point.")

            river_graph = nx.DiGraph()
            for _, row in rivers.iterrows():
                current_basin = row['LINKNO']
                for up_col in ['USLINKNO1', 'USLINKNO2']:
                    upstream_basin = row[up_col]
                    if upstream_basin != -9999:
                        river_graph.add_edge(upstream_basin, current_basin)

            all_upstream_basin_ids = set()
            for _, point in containing_basins.iterrows():
                basin_id = point['streamID']
                upstream_basin_ids = nx.ancestors(river_graph, basin_id)
                upstream_basin_ids.add(basin_id)
                all_upstream_basin_ids.update(upstream_basin_ids)

            subset_basins = basins[basins['streamID'].isin(all_upstream_basin_ids)].copy()
            subset_rivers = rivers[rivers['LINKNO'].isin(all_upstream_basin_ids)].copy()

            subset_basins['hru_to_seg'] = subset_basins['streamID']
            subset_basins = prepare_basin_attributes(subset_basins)

            subset_rivers['COMID'] = subset_rivers['LINKNO']
            subset_rivers['NextDownID'] = subset_rivers['DSLINKNO']
            subset_rivers['slope'] = subset_rivers['Slope']
            subset_rivers = prepare_river_attributes(subset_rivers)

            subset_basins.to_file(output_basins_path)
            subset_rivers.to_file(output_rivers_path)

            logger.info(f"Subset TDX basins shapefile saved to: {output_basins_path}")
            logger.info(f"Subset TDX rivers shapefile saved to: {output_rivers_path}")

        except Exception as e:
            logger.error(f"Error subsetting TDX Hydrofabrics: {str(e)}")
    else:
        logger.info('Output shapefile already exists, carrying on')
    
    return output_basins_path, output_rivers_path

def subset_NWS_hydrofabrics(config, logger):
    logger.info("Subsetting NWS Hydrofabrics")
    output_basins_path = Path(config.root_path) / f"domain_{config.domain_name}" / "shapefiles" / "river_basins" / f"{config.domain_name}_basins_NWS.shp"
    output_rivers_path = Path(config.root_path) / f"domain_{config.domain_name}" / "shapefiles" / "river_network" / f"{config.domain_name}_network_NWS.shp"

    if not output_basins_path.exists():
        try:
            basins_path = Path(config.root_path) / f"domain_{config.full_domain_name}" / "shapefiles" / "river_basins" / config.fullDom_basins_name
            rivers_path = Path(config.root_path) / f"domain_{config.full_domain_name}" / "shapefiles" / "river_network" / config.fullDom_rivers_name

            basins = gpd.read_file(basins_path)
            rivers = gpd.read_file(rivers_path)

            pour_point_shp = gpd.read_file(config.pour_point_shapefile_path)

            if pour_point_shp.crs != basins.crs:
                pour_point_shp = pour_point_shp.to_crs(basins.crs)

            containing_basin = gpd.sjoin(pour_point_shp, basins, how='left', predicate='within')
            if containing_basin.empty:
                raise ValueError("No basin contains the given pour point.")
            downstream_basin_id = containing_basin.iloc[0]['COMID']

            river_graph = nx.DiGraph()
            for _, row in rivers.iterrows():
                current_basin = row['COMID']
                upstream_basin = row['toCOMID']
                if upstream_basin != 0:  # Assuming 0 indicates no upstream basin
                    river_graph.add_edge(current_basin, upstream_basin)

            upstream_basin_ids = set()
            stack = [downstream_basin_id]
            while stack:
                current = stack.pop()
                upstream_basin_ids.add(current)
                stack.extend([n for n in river_graph.predecessors(current) if n not in upstream_basin_ids])

            subset_basins = basins[basins['COMID'].isin(upstream_basin_ids)].copy()
            subset_rivers = rivers[rivers['COMID'].isin(upstream_basin_ids)].copy()

            subset_basins['hru_to_seg'] = subset_basins['COMID']
            subset_basins = prepare_basin_attributes(subset_basins)

            subset_rivers['NextDownID'] = subset_rivers['toCOMID']
            subset_rivers = prepare_river_attributes(subset_rivers)

            subset_basins.to_file(output_basins_path)
            subset_rivers.to_file(output_rivers_path)

            logger.info(f"Subset NWS basins shapefile saved to: {output_basins_path}")
            logger.info(f"Subset NWS rivers shapefile saved to: {output_rivers_path}")

        except Exception as e:
            logger.error(f"Error subsetting NWS Hydrofabrics: {str(e)}")
    else:
        logger.info('Output shapefile already exists, carrying on')
    
    return output_basins_path, output_rivers_path

def find_bounding_box_coordinates(config, logger):
    logger.info("Finding bounding box coordinates")

    basins_path = Path(config.root_path) / f"domain_{config.domain_name}" / "shapefiles" / "river_basins" / f"{config.river_basin_shp_name}"
    try:
        shp = gpd.read_file(str(basins_path))
        
        bounding_box = shp.total_bounds

        lon_range = bounding_box[2] - bounding_box[0]
        lat_range = bounding_box[3] - bounding_box[1]
        
        buffered_box = [
            bounding_box[0] - 0.1 * lon_range,  # lon_min
            bounding_box[1] - 0.1 * lat_range,  # lat_min
            bounding_box[2] + 0.1 * lon_range,  # lon_max
            bounding_box[3] + 0.1 * lat_range   # lat_max
        ]

        rounded_coords, lat, lon = round_bounding_box(buffered_box)
        
        update_control_file_bounding_box(config, rounded_coords)

        logger.info(f"Bounding box coordinates (with 10% buffer): {rounded_coords}")

        return rounded_coords

    except Exception as e:
        logger.error(f"Error finding bounding box coordinates: {str(e)}", exc_info=True)
        return None

def prepare_basin_attributes(gdf):
    gdf['GRU_ID'] = gdf.index + 1
    gdf['area'] = gdf.to_crs(gdf.estimate_utm_crs()).area / 1e6  # area in km²
    return gdf[['GRU_ID', 'hru_to_seg', 'area', 'geometry']]

def prepare_river_attributes(gdf):
    gdf['length'] = gdf.to_crs(gdf.estimate_utm_crs()).length  # length in meters
    return gdf[['COMID', 'NextDownID', 'slope', 'length', 'geometry']]

def round_bounding_box(coords):
    lon = [coords[0], coords[2]]
    lat = [coords[1], coords[3]]

    rounded_lon = [math.floor(lon[0] * 100) / 100, math.ceil(lon[1] * 100) / 100]
    rounded_lat = [math.floor(lat[0] * 100) / 100, math.ceil(lat[1] * 100) / 100]

    control_string = f"{rounded_lat[1]}/{rounded_lon[0]}/{rounded_lat[0]}/{rounded_lon[1]}"

    return control_string, rounded_lat, rounded_lon

def update_control_file_bounding_box(config, bounding_box):
    control_file_path = Path(config.root_path) / f"domain_{config.domain_name}" / "_workflow_log" / config.source_control_file
    
    try:
        with open(control_file_path, 'r') as file:
            lines = file.readlines()

        for i, line in enumerate(lines):
            if line.strip().startswith('forcing_raw_space'):
                lines[i] = f"forcing_raw_space            | {bounding_box}                                     # Bounding box of the shapefile: lat_max/lon_min/lat_min/lon_max. Will be converted to ERA5 download coordinates in script. Order and use of '/' to separate values is mandatory.\n"
                break

        with open(control_file_path, 'w') as file:
            file.writelines(lines)

    except Exception as e:
        raise Exception(f"Error updating control file with new bounding box: {str(e)}")
    
def generate_hrus(config, logger):
    """
    Generate HRUs based on the specified attribute in the control file.
    """
    logger.info("Generating HRUs")

    generate_hrus_setting = config.domain_discretisation.lower()

    if generate_hrus_setting == 'none':
        logger.info("Using GRUs as HRUs (no further discretization)")
        _use_grus_as_hrus(config, logger)
    elif generate_hrus_setting == 'elevation':
        logger.info("Generating HRUs based on elevation")
        _generate_elevation_based_hrus(config, logger)
    elif generate_hrus_setting == 'soil_class':
        logger.info("Generating HRUs based on soil class")
        _generate_soil_class_based_hrus(config, logger)
    elif generate_hrus_setting == 'land_class':
        logger.info("Generating HRUs based on land class")
        _generate_land_class_based_hrus(config, logger)
    elif generate_hrus_setting == 'radiation_class':
        logger.info("Generating HRUs based on radiation")
        _generate_radiation_based_hrus(config, logger)
    else:
        logger.error(f"Invalid generate_hrus setting: {generate_hrus_setting}")
        raise ValueError(f"Invalid generate_hrus setting: {generate_hrus_setting}")

def _use_grus_as_hrus(config, logger):
    """
    Use GRUs as HRUs without further discretization.
    """
    gru_shapefile = Path(config.root_path) / f"domain_{config.domain_name}" / "shapefiles" / "river_basins" / f"{config.domain_name}_basins.shp"
    hru_output_shapefile = Path(config.root_path) / f"domain_{config.domain_name}" / "shapefiles" / "catchment" / f"{config.domain_name}_HRUs.shp"

    gru_gdf = gpd.read_file(gru_shapefile)
    gru_gdf['hruId'] = gru_gdf['COMID']
    gru_gdf['hruNo'] = range(1, len(gru_gdf) + 1)
    gru_gdf['area'] = gru_gdf.to_crs(gru_gdf.estimate_utm_crs()).area / 1e6  # area in km²
    gru_gdf['hru_type'] = 'GRU'

    # Save as HRU shapefile
    gru_gdf.to_file(hru_output_shapefile)
    logger.info(f"GRUs saved as HRUs to {hru_output_shapefile}")
    
def prepare_hru_parameters(config, logger):
    """
    Prepare HRU parameters for SUMMA by calculating area and centroid coordinates.
    """
    logger.info("Preparing HRU parameters for SUMMA")

    try:
        # Construct file paths
        if config.catchment_shp_path == 'default':
            input_shapefile = Path(config.root_path) / f"domain_{config.domain_name}" / "shapefiles" / "catchment" / config.catchment_shp_name
        else:
            input_shapefile = Path(config.catchment_shp_path) / config.catchment_shp_name
        
        output_shapefile = input_shapefile
        projected_crs = CRS.from_epsg(32606)  # UTM Zone 6N, adjust if needed for your area

        logger.info(f"Input Shapefile: {input_shapefile}")
        logger.info(f"Output Shapefile: {output_shapefile}")

        # Read the shapefile
        gdf = gpd.read_file(input_shapefile)
        original_crs = gdf.crs

        # Project to UTM for area calculation
        gdf = gdf.to_crs(projected_crs)

        # Calculate area in square meters
        gdf['HRU_area'] = gdf.geometry.area

        # Calculate centroids
        centroids = gdf.geometry.centroid
        gdf['center_lon'] = centroids.x
        gdf['center_lat'] = centroids.y

        # Set GRU_ID and HRU_ID
        gdf['GRU_ID'] = gdf['COMID']
        if 'hruNo' in gdf.columns:
            gdf['HRU_ID'] = gdf['hruNo']
        else:
            gdf['HRU_ID'] = gdf['COMID']

        # Select and order columns
        gdf = gdf[['GRU_ID', 'HRU_ID', 'HRU_area', 'center_lon', 'center_lat', 'geometry']]

        # Project back to original CRS
        gdf = gdf.to_crs(original_crs)

        # Ensure output directory exists
        output_shapefile.parent.mkdir(parents=True, exist_ok=True)

        # Save the updated GeoDataFrame to a new shapefile
        gdf.to_file(output_shapefile)
        logger.info(f"Updated shapefile with HRU parameters saved to: {output_shapefile}")

    except Exception as e:
        logger.error(f"An error occurred while preparing HRU parameters: {str(e)}", exc_info=True)

def _read_and_prepare_data(gru_shapefile, dem_raster, elevation_band_size):
    gru_gdf = gpd.read_file(gru_shapefile).to_crs("EPSG:4326")
    if 'gruId' not in gru_gdf.columns:
        gru_gdf['gruId'] = gru_gdf.index.astype(str)

    gru_gdf['geometry'] = gru_gdf['geometry'].apply(lambda geom: geom if geom.is_valid else geom.buffer(0))

    with rasterio.open(dem_raster) as src:
        out_image, out_transform = rasterio.mask.mask(src, gru_gdf.geometry, crop=True)
        out_image = out_image[0]  # Get the first band
        valid_data = out_image[out_image != src.nodata]
        
        min_elevation = np.floor(np.min(valid_data))
        max_elevation = np.ceil(np.max(valid_data))

    elevation_thresholds = np.arange(min_elevation, max_elevation + elevation_band_size, elevation_band_size)
    
    return gru_gdf, elevation_thresholds

def _generate_elevation_based_hrus(config, logger):
    """
    Generate HRUs based on elevation.
    """
    logger.info("Generating elevation-based HRUs")

    try:
        # Read necessary parameters
        gru_shapefile = Path(config.root_path) / f"domain_{config.domain_name}" / "shapefiles" / "river_basins" / f"{config.domain_name}_basins.shp"
        dem_raster = Path(config.root_path) / f"domain_{config.domain_name}" / "parameters" / "dem" / "5_elevation" / config.parameter_dem_tif_name
        output_shapefile = Path(config.root_path) / f"domain_{config.domain_name}" / "shapefiles" / "catchment" / f"{config.domain_name}_HRUs_elevation.shp"
        output_plot = Path(config.root_path) / f"domain_{config.domain_name}" / "plots" / "catchment" / f"{config.domain_name}_HRUs_elevation.png"
        elevation_band_size = float(config.elevation_band_size)
        min_hru_size = float(config.min_hru_size)

        logger.info(f"GRU Shapefile: {gru_shapefile}")
        logger.info(f"DEM Raster: {dem_raster}")
        logger.info(f"Output Shapefile: {output_shapefile}")
        logger.info(f"Output Plot: {output_plot}")
        logger.info(f"Elevation Band Size: {elevation_band_size}")
        logger.info(f"Minimum HRU Size: {min_hru_size}")

        # Check if input files exist
        if not gru_shapefile.exists():
            raise FileNotFoundError(f"Input GRU shapefile not found: {gru_shapefile}")
        if not dem_raster.exists():
            raise FileNotFoundError(f"Input DEM raster not found: {dem_raster}")

        gru_gdf, elevation_thresholds = _read_and_prepare_data(gru_shapefile, dem_raster, elevation_band_size)
        hru_gdf = _process_hrus(gru_gdf, dem_raster, elevation_thresholds)

        if hru_gdf is not None and not hru_gdf.empty:
            logger.info(f"Initial HRUs created: {len(hru_gdf)}")

            hru_gdf = _merge_small_hrus(hru_gdf, min_hru_size, logger)
        
            # Ensure all geometries are valid polygons
            hru_gdf['geometry'] = hru_gdf['geometry'].apply(_clean_geometries)
            hru_gdf = hru_gdf[hru_gdf['geometry'].notnull()]
            
            # Remove any columns that might cause issues with shapefiles
            columns_to_keep = ['geometry', 'gruNo', 'gruId', 'elevClass', 'avg_elevation', 'area', 'hruNo', 'hruId', 'cent_lon', 'cent_lat', 'COMID']
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
                    logger.warning(f"Removing invalid geometry for HRU {idx}")
            
            hru_gdf = gpd.GeoDataFrame(valid_polygons, crs=hru_gdf.crs)

            # Save as Shapefile
            hru_gdf['geometry'] = hru_gdf['geometry'].apply(_to_polygon)     
            
            hru_gdf.to_file(output_shapefile)
            logger.info(f"HRU Shapefile created with {len(hru_gdf)} HRUs and saved to {output_shapefile}")

            _plot_hrus(hru_gdf, output_plot, 'elevClass', 'Elevation-based HRUs')
            logger.info(f"HRU plot saved to {output_plot}")
        else:
            logger.error("No valid HRUs were created. Check your input data and parameters.")

    except Exception as e:
        logger.error(f"An error occurred during elevation-based HRU generation: {str(e)}", exc_info=True)

def _process_hrus(gru_gdf, dem_raster, elevation_thresholds):
    all_hrus = []
    
    # Use half of available CPU cores
    num_cores = max(1, multiprocessing.cpu_count() // 2)
    
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        future_to_row = {executor.submit(_create_hrus, row, dem_raster, elevation_thresholds): row for _, row in gru_gdf.iterrows()}
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
    centroids_wgs84 = centroids_utm.to_crs(rasterio.crs.CRS.from_epsg(4326))
    
    hru_gdf_utm['cent_lon'] = centroids_wgs84.x
    hru_gdf_utm['cent_lat'] = centroids_wgs84.y
    
    # Convert the final GeoDataFrame back to WGS84 for consistency
    hru_gdf = hru_gdf_utm.to_crs(rasterio.crs.CRS.from_epsg(4326))
    
    return hru_gdf

def _create_hrus(row, dem_raster, elevation_thresholds):
    with rasterio.open(dem_raster) as dem_src:
        out_image, out_transform = rasterio.mask.mask(dem_src, [row.geometry], crop=True, all_touched=True)
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
                shapes = features.shapes(class_mask.astype(np.uint8), mask=class_mask, transform=out_transform)
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

def _merge_small_hrus(hru_gdf, min_hru_size, logger):
    # Project to UTM for accurate area calculations
    hru_gdf.set_crs(epsg=4326, inplace=True)
    utm_crs = hru_gdf.estimate_utm_crs()
    
    hru_gdf_utm = hru_gdf.to_crs(utm_crs)
    
    # Clean geometries before processing
    hru_gdf_utm['geometry'] = hru_gdf_utm['geometry'].apply(_clean_geometries)
    hru_gdf_utm = hru_gdf_utm[hru_gdf_utm['geometry'].notnull()]
    
    # Preserve original domain boundary
    try:
        original_boundary = unary_union(hru_gdf_utm.geometry)
    except Exception as e:
        logger.error(f"Error creating original boundary: {str(e)}")
        original_boundary = None
    
    # Calculate areas
    hru_gdf_utm['area'] = hru_gdf_utm.geometry.area / 1_000_000  # Convert m² to km²
    
    # Sort HRUs by area
    hru_gdf_utm = hru_gdf_utm.sort_values('area')
    
    merged_count = 0
    while True:
        small_hrus = hru_gdf_utm[hru_gdf_utm['area'] < min_hru_size]
        if len(small_hrus) == 0:
            break
        
        progress = False
        for idx, small_hru in small_hrus.iterrows():
            try:
                small_hru_geom = _clean_geometries(small_hru.geometry)
                if small_hru_geom is None:
                    logger.warning(f"Invalid geometry for HRU {idx}. Skipping.")
                    hru_gdf_utm = hru_gdf_utm.drop(idx)
                    continue

                neighbors = _find_neighbors(small_hru_geom, hru_gdf_utm, idx, buffer_distance=1e-2)
                if len(neighbors) > 0:
                    largest_neighbor = neighbors.loc[neighbors['area'].idxmax()]
                    largest_neighbor_geom = _clean_geometries(largest_neighbor.geometry)
                    if largest_neighbor_geom is None:
                        logger.warning(f"Invalid geometry for largest neighbor of HRU {idx}. Skipping.")
                        continue

                    merged_geometry = unary_union([small_hru_geom, largest_neighbor_geom])
                    merged_geometry = _simplify_geometry(merged_geometry)
                    
                    # Update the largest neighbor's geometry and area
                    hru_gdf_utm.at[largest_neighbor.name, 'geometry'] = merged_geometry
                    hru_gdf_utm.at[largest_neighbor.name, 'area'] = merged_geometry.area / 1_000_000
                    
                    # Remove the small HRU
                    hru_gdf_utm = hru_gdf_utm.drop(idx)
                    merged_count += 1
                    progress = True
                else:
                    # If no neighbors found, find the nearest HRU and merge
                    distances = hru_gdf_utm.geometry.distance(small_hru_geom)
                    nearest_hru = hru_gdf_utm.loc[distances.idxmin()]
                    if nearest_hru.name != idx:
                        nearest_hru_geom = _clean_geometries(nearest_hru.geometry)
                        if nearest_hru_geom is None:
                            logger.warning(f"Invalid geometry for nearest HRU of {idx}. Skipping.")
                            continue
                        
                        merged_geometry = unary_union([small_hru_geom, nearest_hru_geom])
                        merged_geometry = _simplify_geometry(merged_geometry)
                        
                        # Update the nearest HRU's geometry and area
                        hru_gdf_utm.at[nearest_hru.name, 'geometry'] = merged_geometry
                        hru_gdf_utm.at[nearest_hru.name, 'area'] = merged_geometry.area / 1_000_000
                        
                        # Remove the small HRU
                        hru_gdf_utm = hru_gdf_utm.drop(idx)
                        merged_count += 1
                        progress = True
            except Exception as e:
                logger.error(f"Error merging HRU {idx}: {str(e)}")
                logger.error(f"Small HRU geometry: {small_hru.geometry.wkt}")
                if len(neighbors) > 0:
                    logger.error(f"Largest neighbor geometry: {largest_neighbor.geometry.wkt}")
        
        if not progress:
            break
        
        # Recalculate areas and sort
        hru_gdf_utm['area'] = hru_gdf_utm.geometry.area / 1_000_000
        hru_gdf_utm = hru_gdf_utm.sort_values('area')
    
    # Fill gaps
    if original_boundary is not None:
        try:
            gaps = original_boundary.difference(unary_union(hru_gdf_utm.geometry))
            if not gaps.is_empty:
                for gap in gaps.geoms:
                    nearest_hru = hru_gdf_utm.geometry.distance(gap).idxmin()
                    hru_gdf_utm.at[nearest_hru, 'geometry'] = _clean_geometries(unary_union([hru_gdf_utm.at[nearest_hru, 'geometry'], gap]))
        except Exception as e:
            logger.error(f"Error filling gaps: {str(e)}")
    
    # Update HRU numbers and IDs
    hru_gdf_utm = hru_gdf_utm.reset_index(drop=True)
    hru_gdf_utm['hruNo'] = range(1, len(hru_gdf_utm) + 1)
    hru_gdf_utm['hruId'] = hru_gdf_utm['gruId'] + '_' + hru_gdf_utm['elevClass'].astype(str)
    
    # Recalculate final areas
    hru_gdf_utm['area'] = hru_gdf_utm.geometry.area / 1_000_000
    
    # Project back to WGS84
    hru_gdf_merged = hru_gdf_utm.to_crs(hru_gdf.crs)
    
    return hru_gdf_merged

def _clean_geometries(geometry):
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

def _find_neighbors(geometry, gdf, current_index, buffer_distance=1e-2):
    try:
        # Simplify the geometry
        simplified_geom = _simplify_geometry(geometry)
        buffered = simplified_geom.buffer(buffer_distance)
        
        # Find neighbors using spatial index for efficiency
        possible_matches_index = list(gdf.sindex.intersection(buffered.bounds))
        possible_matches = gdf.iloc[possible_matches_index]
        precise_matches = possible_matches[possible_matches.intersects(buffered)]
        
        return precise_matches[precise_matches.index != current_index]
    except Exception as e:
        print(f"Error finding neighbors for HRU {current_index}: {str(e)}")
        return gpd.GeoDataFrame()

def _simplify_geometry(geom, tolerance=0.0001):
    """Simplify the geometry to reduce complexity while preserving shape."""
    if geom.is_empty:
        return geom
    if geom.geom_type == 'MultiPolygon':
        parts = [part for part in geom.geoms if part.is_valid and not part.is_empty]
        if len(parts) == 0:
            return Polygon()
        return MultiPolygon(parts).simplify(tolerance, preserve_topology=True)
    return geom.simplify(tolerance, preserve_topology=True)

def _to_polygon(geom):
    if isinstance(geom, MultiPolygon):
        if len(geom.geoms) == 1:
            return geom.geoms[0]
        else:
            return geom.buffer(0)
    return geom

def _plot_hrus(hru_gdf, output_file, class_column, title):
    fig, ax = plt.subplots(figsize=(15, 15))
    
    if class_column == 'radiationClass':
        # Use a sequential colormap for radiation
        hru_gdf.plot(column='avg_radiation', cmap='viridis', legend=True, ax=ax)
    else:
        # Use a qualitative colormap for other class types
        hru_gdf.plot(column=class_column, cmap='viridis', legend=True, ax=ax)
    
    ax.set_title(title)
    plt.axis('off')
    plt.tight_layout()
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

def _generate_soil_class_based_hrus(config, logger):
    """
    Generate HRUs based on soil class.
    """
    logger.info("Generating soil class-based HRUs")

    try:
        # Read necessary parameters
        gru_shapefile = Path(config.root_path) / f"domain_{config.domain_name}" / "shapefiles" / "river_basins" / f"{config.domain_name}_basins.shp"
        soil_raster = Path(config.root_path) / f"domain_{config.domain_name}" / "parameters" / "soilclass" / "2_soil_classes_domain" / config.parameter_soil_tif_name
        output_shapefile = Path(config.root_path) / f"domain_{config.domain_name}" / "shapefiles" / "catchment" / f"{config.domain_name}_HRUs_soil.shp"
        output_plot = Path(config.root_path) / f"domain_{config.domain_name}" / "plots" / "catchment" / f"{config.domain_name}_HRUs_soil.png"

        min_hru_size = float(config.min_hru_size)

        logger.info(f"GRU Shapefile: {gru_shapefile}")
        logger.info(f"Soil Raster: {soil_raster}")
        logger.info(f"Output Shapefile: {output_shapefile}")
        logger.info(f"Output Plot: {output_plot}")
        logger.info(f"Minimum HRU Size: {min_hru_size}")

        # Check if input files exist
        if not gru_shapefile.exists():
            raise FileNotFoundError(f"Input GRU shapefile not found: {gru_shapefile}")
        if not soil_raster.exists():
            raise FileNotFoundError(f"Input soil raster not found: {soil_raster}")

        gru_gdf, unique_soil_classes = _read_and_prepare_soil_data(gru_shapefile, soil_raster)
        hru_gdf = _process_soil_hrus(gru_gdf, soil_raster, unique_soil_classes)

        if hru_gdf is not None and not hru_gdf.empty:
            logger.info(f"Initial HRUs created: {len(hru_gdf)}")

            hru_gdf = _merge_small_hrus(hru_gdf, min_hru_size)
        
            # Ensure all geometries are valid polygons
            hru_gdf['geometry'] = hru_gdf['geometry'].apply(_clean_geometries)
            hru_gdf = hru_gdf[hru_gdf['geometry'].notnull()]
            
            # Remove any columns that might cause issues with shapefiles
            columns_to_keep = ['geometry', 'gruNo', 'gruId', 'soilClass', 'area', 'hruNo', 'hruId', 'cent_lon', 'cent_lat', 'COMID']
            hru_gdf = hru_gdf[columns_to_keep]

            # Final check for valid polygons
            valid_polygons = []
            for idx, row in hru_gdf.iterrows():
                geom = row['geometry']
                if isinstance(geom, (Polygon, MultiPolygon)) and geom.is_valid:
                    valid_polygons.append(row)
                else:
                    logger.warning(f"Removing invalid geometry for HRU {idx}")
            
            hru_gdf = gpd.GeoDataFrame(valid_polygons, crs=hru_gdf.crs)

            # Save as Shapefile
            hru_gdf['geometry'] = hru_gdf['geometry'].apply(_to_polygon)     
            
            hru_gdf.to_file(output_shapefile)
            logger.info(f"Soil-based HRU Shapefile created with {len(hru_gdf)} HRUs and saved to {output_shapefile}")

            _plot_hrus(hru_gdf, output_plot, 'soilClass', 'Soil-based HRUs')
            logger.info(f"Soil-based HRU plot saved to {output_plot}")
        else:
            logger.error("No valid HRUs were created. Check your input data and parameters.")

    except Exception as e:
        logger.error(f"An error occurred during soil-based HRU generation: {str(e)}", exc_info=True)

def _read_and_prepare_soil_data(gru_shapefile, soil_raster):
    gru_gdf = gpd.read_file(gru_shapefile).to_crs("EPSG:4326")
    if 'gruId' not in gru_gdf.columns:
        gru_gdf['gruId'] = gru_gdf.index.astype(str)

    gru_gdf['geometry'] = gru_gdf['geometry'].apply(lambda geom: make_valid(geom))

    with rasterio.open(soil_raster) as src:
        unique_soil_classes = np.unique(src.read(1)[src.read(1) != src.nodata])
    
    return gru_gdf, unique_soil_classes

def _process_soil_hrus(gru_gdf, soil_raster, unique_soil_classes):
    all_hrus = []
    
    num_cores = max(1, multiprocessing.cpu_count() // 2)
    
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        future_to_row = {executor.submit(_create_soil_hrus, row, soil_raster, unique_soil_classes): row for _, row in gru_gdf.iterrows()}
        for future in as_completed(future_to_row):
            all_hrus.extend(future.result())

    hru_gdf = gpd.GeoDataFrame(all_hrus, crs=gru_gdf.crs)
    
    return _postprocess_hrus(hru_gdf)

def _create_soil_hrus(row, soil_raster, unique_soil_classes):
    with rasterio.open(soil_raster) as src:
        out_image, out_transform = mask(src, [row.geometry], crop=True, all_touched=True)
        out_image = out_image[0]
        
        gru_attributes = row.drop('geometry').to_dict()
        
        hrus = []
        for soil_class in unique_soil_classes:
            class_mask = (out_image == soil_class)
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
                                    'soilClass': int(soil_class),
                                    **gru_attributes
                                })
        
        return hrus if hrus else [{
            'geometry': row.geometry,
            'gruNo': row.name,
            'gruId': row['gruId'],
            'soilClass': -9999,  # No-data value
            **gru_attributes
        }]

def _postprocess_hrus(hru_gdf):
    # Project to UTM for area calculation
    utm_crs = hru_gdf.estimate_utm_crs()
    hru_gdf_utm = hru_gdf.to_crs(utm_crs)
    hru_gdf_utm['area'] = hru_gdf_utm.area / 1_000_000  # Convert m² to km²

    hru_gdf_utm['area'] = hru_gdf_utm['area'].astype('float64')
    if 'avg_radiation' in hru_gdf_utm.columns:
        hru_gdf_utm['avg_radiation'] = hru_gdf_utm['avg_radiation'].astype('float64')

    hru_gdf_utm['hruNo'] = range(1, len(hru_gdf_utm) + 1)
    
    # Determine the class column name
    class_column = next((col for col in ['elevClass', 'soilClass', 'landClass', 'radiationClass'] if col in hru_gdf_utm.columns), None)
    if class_column:
        hru_gdf_utm['hruId'] = hru_gdf_utm['gruId'] + '_' + hru_gdf_utm[class_column].astype(str)
    else:
        hru_gdf_utm['hruId'] = hru_gdf_utm['gruId'] + '_' + hru_gdf_utm['hruNo'].astype(str)
    
    centroids_utm = hru_gdf_utm.geometry.centroid
    centroids_wgs84 = centroids_utm.to_crs(CRS.from_epsg(4326))
    
    hru_gdf_utm['cent_lon'] = centroids_wgs84.x
    hru_gdf_utm['cent_lat'] = centroids_wgs84.y
    
    hru_gdf = hru_gdf_utm.to_crs(CRS.from_epsg(4326))
    
    return hru_gdf

def _generate_land_class_based_hrus(config, logger):
    """
    Generate HRUs based on land class.
    """
    logger.info("Generating land class-based HRUs")

    try:
        # Read necessary parameters
        gru_shapefile = Path(config.root_path) / f"domain_{config.domain_name}" / "shapefiles" / "river_basins" / f"{config.domain_name}_basins.shp"
        land_raster = Path(config.root_path) / f"domain_{config.full_domain_name}" / "parameters" / "landclass" / "7_mode_land_class" / config.parameter_land_tif_name
        output_shapefile = Path(config.root_path) / f"domain_{config.domain_name}" / "shapefiles" / "catchment" / f"{config.domain_name}_HRUs_land.shp"
        output_plot = Path(config.root_path) / f"domain_{config.domain_name}" / "plots" / "catchment" / f"{config.domain_name}_HRUs_land.png"

        min_hru_size = float(config.min_hru_size)

        logger.info(f"GRU Shapefile: {gru_shapefile}")
        logger.info(f"Land Raster: {land_raster}")
        logger.info(f"Output Shapefile: {output_shapefile}")
        logger.info(f"Output Plot: {output_plot}")
        logger.info(f"Minimum HRU Size: {min_hru_size}")

        # Check if input files exist
        if not gru_shapefile.exists():
            raise FileNotFoundError(f"Input GRU shapefile not found: {gru_shapefile}")
        if not land_raster.exists():
            raise FileNotFoundError(f"Input land raster not found: {land_raster}")

        gru_gdf, unique_land_classes = _read_and_prepare_land_data(gru_shapefile, land_raster)
        hru_gdf = _process_land_hrus(gru_gdf, land_raster, unique_land_classes)

        if hru_gdf is not None and not hru_gdf.empty:
            logger.info(f"Initial HRUs created: {len(hru_gdf)}")

            hru_gdf = _merge_small_hrus(hru_gdf, min_hru_size)
        
            # Ensure all geometries are valid polygons
            hru_gdf['geometry'] = hru_gdf['geometry'].apply(_clean_geometries)
            hru_gdf = hru_gdf[hru_gdf['geometry'].notnull()]
            
            # Remove any columns that might cause issues with shapefiles
            columns_to_keep = ['geometry', 'gruNo', 'gruId', 'landClass', 'area', 'hruNo', 'hruId', 'cent_lon', 'cent_lat', 'COMID']
            hru_gdf = hru_gdf[columns_to_keep]

            # Final check for valid polygons
            valid_polygons = []
            for idx, row in hru_gdf.iterrows():
                geom = row['geometry']
                if isinstance(geom, (Polygon, MultiPolygon)) and geom.is_valid:
                    valid_polygons.append(row)
                else:
                    logger.warning(f"Removing invalid geometry for HRU {idx}")
            
            hru_gdf = gpd.GeoDataFrame(valid_polygons, crs=hru_gdf.crs)

            # Save as Shapefile
            hru_gdf['geometry'] = hru_gdf['geometry'].apply(_to_polygon)     
            
            hru_gdf.to_file(output_shapefile)
            logger.info(f"Land-based HRU Shapefile created with {len(hru_gdf)} HRUs and saved to {output_shapefile}")

            _plot_hrus(hru_gdf, output_plot, 'landClass', 'Land-based HRUs')
            logger.info(f"Land-based HRU plot saved to {output_plot}")
        else:
            logger.error("No valid HRUs were created. Check your input data and parameters.")

    except Exception as e:
        logger.error(f"An error occurred during land-based HRU generation: {str(e)}", exc_info=True)

def _read_and_prepare_land_data(gru_shapefile, land_raster):
    gru_gdf = gpd.read_file(gru_shapefile).to_crs("EPSG:4326")
    if 'gruId' not in gru_gdf.columns:
        gru_gdf['gruId'] = gru_gdf.index.astype(str)

    gru_gdf['geometry'] = gru_gdf['geometry'].apply(lambda geom: make_valid(geom))

    with rasterio.open(land_raster) as src:
        unique_land_classes = np.unique(src.read(1)[src.read(1) != src.nodata])
    
    return gru_gdf, unique_land_classes

def _process_land_hrus(gru_gdf, land_raster, unique_land_classes):
    all_hrus = []
    
    num_cores = max(1, multiprocessing.cpu_count() // 2)
    
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        future_to_row = {executor.submit(_create_land_hrus, row, land_raster, unique_land_classes): row for _, row in gru_gdf.iterrows()}
        for future in as_completed(future_to_row):
            all_hrus.extend(future.result())

    hru_gdf = gpd.GeoDataFrame(all_hrus, crs=gru_gdf.crs)
    
    return _postprocess_hrus(hru_gdf)

def _create_land_hrus(row, land_raster, unique_land_classes):
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

def _generate_radiation_based_hrus(config, logger):
    """
    Generate HRUs based on radiation.
    """
    logger.info("Generating radiation-based HRUs")

    try:
        # Read necessary parameters
        gru_shapefile = Path(config.root_path) / f"domain_{config.domain_name}" / "shapefiles" / "river_basins" / f"{config.domain_name}_basins.shp"
        dem_raster = Path(config.root_path) / f"domain_{config.domain_name}" / "parameters" / "dem" / "5_elevation" / config.parameter_dem_tif_name
        radiation_raster = Path(config.root_path) / f"domain_{config.domain_name}" / "parameters" / "radiation" / "annual_radiation.tif"
        output_shapefile = Path(config.root_path) / f"domain_{config.domain_name}" / "shapefiles" / "catchment" / f"{config.domain_name}_HRUs_radiation.shp"
        output_plot = Path(config.root_path) / f"domain_{config.domain_name}" / "plots" / "catchment" / f"{config.domain_name}_HRUs_radiation.png"

        min_hru_size = float(config.min_hru_size)
        radiation_class_number = int(config.radiation_class_number)

        logger.info(f"GRU Shapefile: {gru_shapefile}")
        logger.info(f"DEM Raster: {dem_raster}")
        logger.info(f"Radiation Raster: {radiation_raster}")
        logger.info(f"Output Shapefile: {output_shapefile}")
        logger.info(f"Output Plot: {output_plot}")
        logger.info(f"Minimum HRU Size: {min_hru_size}")
        logger.info(f"Number of Radiation Classes: {radiation_class_number}")

        # Check if input files exist
        if not gru_shapefile.exists():
            raise FileNotFoundError(f"Input GRU shapefile not found: {gru_shapefile}")
        if not dem_raster.exists():
            raise FileNotFoundError(f"Input DEM raster not found: {dem_raster}")

        # Generate or load radiation raster
        if not radiation_raster.exists():
            logger.info("Annual radiation raster not found. Calculating radiation...")
            radiation_raster = _calculate_annual_radiation(dem_raster, radiation_raster)
        else:
            logger.info(f"Annual radiation raster found at {radiation_raster}")

        gru_gdf, radiation_thresholds = _read_and_prepare_radiation_data(gru_shapefile, radiation_raster, radiation_class_number)
        hru_gdf = _process_radiation_hrus(gru_gdf, radiation_raster, radiation_thresholds)

        if hru_gdf is not None and not hru_gdf.empty:
            logger.info(f"Initial HRUs created: {len(hru_gdf)}")

            hru_gdf = _merge_small_hrus(hru_gdf, min_hru_size)
        
            # Ensure all geometries are valid polygons
            hru_gdf['geometry'] = hru_gdf['geometry'].apply(_clean_geometries)
            hru_gdf = hru_gdf[hru_gdf['geometry'].notnull()]
            
            # Remove any columns that might cause issues with shapefiles
            columns_to_keep = ['geometry', 'gruNo', 'gruId', 'radiationClass', 'avg_radiation', 'area', 'hruNo', 'hruId', 'cent_lon', 'cent_lat', 'COMID']
            hru_gdf = hru_gdf[columns_to_keep]

            # Final check for valid polygons
            valid_polygons = []
            for idx, row in hru_gdf.iterrows():
                geom = row['geometry']
                if isinstance(geom, (Polygon, MultiPolygon)) and geom.is_valid:
                    valid_polygons.append(row)
                else:
                    logger.warning(f"Removing invalid geometry for HRU {idx}")
            
            hru_gdf = gpd.GeoDataFrame(valid_polygons, crs=hru_gdf.crs)

            # Save as Shapefile
            hru_gdf['geometry'] = hru_gdf['geometry'].apply(_to_polygon)     
            
            hru_gdf.to_file(output_shapefile)
            logger.info(f"Radiation-based HRU Shapefile created with {len(hru_gdf)} HRUs and saved to {output_shapefile}")

            _plot_hrus(hru_gdf, output_plot, 'radiationClass', 'Radiation-based HRUs')
            logger.info(f"Radiation-based HRU plot saved to {output_plot}")
        else:
            logger.error("No valid HRUs were created. Check your input data and parameters.")

    except Exception as e:
        logger.error(f"An error occurred during radiation-based HRU generation: {str(e)}", exc_info=True)

def _calculate_annual_radiation(dem_raster: Path, radiation_raster: Path) -> Path:
    """
    Calculate annual radiation based on DEM and save it as a new raster.
    """
    with rasterio.open(dem_raster) as src:
        dem = src.read(1)
        transform = src.transform
        crs = src.crs
        bounds = src.bounds
    
    center_lat = (bounds.bottom + bounds.top) / 2
    center_lon = (bounds.left + bounds.right) / 2

    # Calculate slope and aspect
    dy, dx = np.gradient(dem)
    slope = np.arctan(np.sqrt(dx*dx + dy*dy))
    aspect = np.arctan2(-dx, dy)

    # Create a DatetimeIndex for the entire year (daily)
    times = pd.date_range(start='2019-01-01', end='2019-12-31', freq='D')
    
    # Create location object
    location = pvlib.location.Location(latitude=center_lat, longitude=center_lon, altitude=np.mean(dem))
    
    # Calculate solar position
    solar_position = location.get_solarposition(times=times)
    
    # Calculate clear sky radiation
    clearsky = location.get_clearsky(times=times)
    
    # Initialize the radiation array
    radiation = np.zeros_like(dem)

    # Calculate radiation for each pixel
    for i in range(dem.shape[0]):
        for j in range(dem.shape[1]):
            surface_tilt = np.degrees(slope[i, j])
            surface_azimuth = np.degrees(aspect[i, j])
            
            total_irrad = pvlib.irradiance.get_total_irradiance(
                surface_tilt, surface_azimuth,
                solar_position['apparent_zenith'], solar_position['azimuth'],
                clearsky['dni'], clearsky['ghi'], clearsky['dhi']
            )
            
            radiation[i, j] = total_irrad['poa_global'].sum()

    # Save the radiation raster
    radiation_raster.parent.mkdir(parents=True, exist_ok=True)
    
    with rasterio.open(radiation_raster, 'w', driver='GTiff',
                    height=radiation.shape[0], width=radiation.shape[1],
                    count=1, dtype=radiation.dtype,
                    crs=crs, transform=transform) as dst:
        dst.write(radiation, 1)

    return radiation_raster

def _read_and_prepare_radiation_data(gru_shapefile, radiation_raster, num_classes):
    gru_gdf = gpd.read_file(gru_shapefile).to_crs("EPSG:4326")
    if 'gruId' not in gru_gdf.columns:
        gru_gdf['gruId'] = gru_gdf.index.astype(str)

    gru_gdf['geometry'] = gru_gdf['geometry'].apply(lambda geom: make_valid(geom))

    with rasterio.open(radiation_raster) as src:
        radiation = src.read(1)
        radiation = radiation[radiation != src.nodata]
        radiation_thresholds = np.linspace(np.min(radiation), np.max(radiation), num_classes + 1)
    
    return gru_gdf, radiation_thresholds

def _process_radiation_hrus(gru_gdf, radiation_raster, radiation_thresholds):
    all_hrus = []
    
    num_cores = max(1, multiprocessing.cpu_count() // 2)
    
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        future_to_row = {executor.submit(_create_radiation_hrus, row, radiation_raster, radiation_thresholds): row for _, row in gru_gdf.iterrows()}
        for future in as_completed(future_to_row):
            all_hrus.extend(future.result())

    hru_gdf = gpd.GeoDataFrame(all_hrus, crs=gru_gdf.crs)
    
    return _postprocess_hrus(hru_gdf)

def _create_radiation_hrus(row, radiation_raster, radiation_thresholds):
    with rasterio.open(radiation_raster) as src:
        out_image, out_transform = mask(src, [row.geometry], crop=True, all_touched=True)
        out_image = out_image[0]
        
        gru_attributes = row.drop('geometry').to_dict()
        
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
            'avg_radiation': np.mean(out_image),
            **gru_attributes
        }]