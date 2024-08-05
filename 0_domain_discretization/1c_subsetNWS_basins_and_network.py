import sys
from pathlib import Path
import geopandas as gpd
import networkx as nx
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.control_file_utils import read_from_control, make_default_path

# Easy access to control file folder
controlFolder = Path('../0_control_files')
controlFile = 'control_active.txt'

# Read necessary paths and settings
root_path = Path(read_from_control(controlFolder/controlFile, 'root_path'))
domain_name = read_from_control(controlFolder/controlFile, 'domain_name')
full_domain_name = read_from_control(controlFolder/controlFile, 'full_domain_name')

# Define input file paths
basins_path = root_path / f'domain_{full_domain_name}' / 'shapefiles' / 'river_basins' / 'Hawaii_nhd_catchment.shp'
rivers_path = root_path / f'domain_{full_domain_name}' / 'shapefiles' / 'river_network' / 'Hawaii_nhd_flowline.shp'

# Read pour point path from control file
pour_point_shp_path = read_from_control(controlFolder/controlFile, 'pour_point_shp_path')
pour_point_shp_name = read_from_control(controlFolder/controlFile, 'pour_point_shp_name')

if pour_point_shp_path == 'default':
    pour_point_path = make_default_path(controlFolder, controlFile, 'shapefiles/pour_point')
else:
    pour_point_path = Path(pour_point_shp_path)

if pour_point_shp_name == 'default':
    pour_point_file = f'{domain_name}_pourPoint.shp'
else:
    pour_point_file = pour_point_shp_name

pour_point_path = pour_point_path / pour_point_file

# Load the basin and river network shapefiles
basins = gpd.read_file(basins_path)
rivers = gpd.read_file(rivers_path)

# Load the pour point shapefile
pour_point_shp = gpd.read_file(pour_point_path)

# Check if pour point shapefile has a CRS, if not, use the CRS from basins
if pour_point_shp.crs is None:
    print("Pour point shapefile has no CRS. Using CRS from basin shapefile.")
    pour_point_shp = pour_point_shp.set_crs(basins.crs)
else:
    # If pour point has a CRS but it's different from basins, reproject to match basins
    if pour_point_shp.crs != basins.crs:
        print("Reprojecting pour point to match basin CRS.")
        pour_point_shp = pour_point_shp.to_crs(basins.crs)

# Function to find the basin containing the pour point
def find_basin_for_pour_point(pour_point, basins):
    containing_basin = gpd.sjoin(pour_point, basins, how='left', predicate='within')
    if containing_basin.empty:
        raise ValueError("No basin contains the given pour point.")
    return containing_basin.iloc[0]['COMID']

# Assume the pour point shapefile contains a single point
pour_point = pour_point_shp.geometry.iloc[0]
downstream_basin_id = find_basin_for_pour_point(pour_point_shp, basins)
print(f"Downstream basin ID: {downstream_basin_id}")

# Function to build the river network graph
def build_river_graph(rivers):
    G = nx.DiGraph()
    for _, row in rivers.iterrows():
        current_basin = row['COMID']
        upstream_basin = row['toCOMID']
        if upstream_basin != 0:  # Assuming 0 or some other value indicates no upstream basin
            G.add_edge(current_basin, upstream_basin)
    return G

# Build the river network graph
river_graph = build_river_graph(rivers)

# Function to find all upstream basins using networkx
def find_upstream_basins_networkx(basin_id, G):
    upstream_basins = set()
    stack = [basin_id]
    while stack:
        current = stack.pop()
        upstream_basins.add(current)
        stack.extend([n for n in G.predecessors(current) if n not in upstream_basins])
    return upstream_basins

# Find all upstream basins for the downstream basin
upstream_basin_ids = find_upstream_basins_networkx(downstream_basin_id, river_graph)

# Subset the basins and rivers
subset_basins = basins[basins['COMID'].isin(upstream_basin_ids)].copy()
subset_rivers = rivers[rivers['COMID'].isin(upstream_basin_ids)].copy()

# Define the output file paths for the subset domain
output_basins_path = make_default_path(controlFolder, controlFile, f'shapefiles/river_basins/{domain_name}_basins_NWS.shp')
output_rivers_path = make_default_path(controlFolder, controlFile, f'shapefiles/river_network/{domain_name}_network_NWS.shp')

# Ensure output directories exist
output_basins_path.parent.mkdir(parents=True, exist_ok=True)
output_rivers_path.parent.mkdir(parents=True, exist_ok=True)

# Save the subset basins and rivers to shapefiles
subset_basins.to_file(output_basins_path)
subset_rivers.to_file(output_rivers_path)

print(f"Subset basins shapefile saved to: {output_basins_path}")
print(f"Subset rivers shapefile saved to: {output_rivers_path}")