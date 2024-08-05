import geopandas as gpd
import networkx as nx
import os
from pathlib import Path


#####################################
### --- Control file handling --- ###
#####################################

# Easy access to control file folder
controlFolder = Path('../0_control_files')

# Store the name of the 'active' file in a variable
controlFile = 'control_active.txt'

# Function to extract a given setting from the control file
def read_from_control( file, setting ):
    
    # Open 'control_active.txt' and ...
    with open(file) as contents:
        for line in contents:
            
            # ... find the line with the requested setting
            if setting in line and not line.startswith('#'):
                break
    
    # Extract the setting's value
    substring = line.split('|',1)[1]      # Remove the setting's name (split into 2 based on '|', keep only 2nd part)
    substring = substring.split('#',1)[0] # Remove comments, does nothing if no '#' is found
    substring = substring.strip()         # Remove leading and trailing whitespace, tabs, newlines
       
    # Return this value    
    return substring

# Function to specify a default path
def make_default_path(suffix):
    
    # Get the root path
    rootPath = Path(read_from_control(controlFolder/controlFile,'root_path'))
    
    # Get the domain folder
    domainName = read_from_control(controlFolder/controlFile,'domain_name')
    domainFolder = 'domain_' + domainName
    
    # Specify the forcing path
    defaultPath = rootPath / domainFolder / suffix
    
    return defaultPath

#####################################
### --- Subsetting Functions --- ###
#####################################

# Function to find the basin containing each pour point
def find_basin_for_pour_points(pour_points, basins):
    containing_basins = gpd.sjoin(pour_points, basins, how='left', predicate='within')
    if containing_basins.empty:
        raise ValueError("No basin contains the given pour points.")
    return containing_basins

# Function to build the river network graph
def build_river_graph(rivers):
    G = nx.DiGraph()
    for _, row in rivers.iterrows():
        current_basin = row['LINKNO']
        for up_col in ['USLINKNO1', 'USLINKNO2']:
            upstream_basin = row[up_col]
            if upstream_basin != -9999:
                G.add_edge(upstream_basin, current_basin)
    return G

# Function to find all upstream basins using networkx
def find_upstream_basins_networkx(basin_ids, G):
    upstream_basin_ids = set()
    for basin_id in basin_ids:
        upstream_basins = nx.ancestors(G, basin_id)
        upstream_basins.add(basin_id)  # Include the target basin itself
        upstream_basin_ids.update(upstream_basins)
    return upstream_basin_ids

# Function to extract upstream basins and network for each point
def extract_upstream_data(basins, rivers, pour_points, output_dir):
    containing_basins = find_basin_for_pour_points(pour_points, basins)
    
    river_graph = build_river_graph(rivers)
    
    for index, point in containing_basins.iterrows():
        basin_id = point['streamID']
        point_id = point['point_id']  # Assumes pour points have an 'point_id' field
        
        upstream_basin_ids = find_upstream_basins_networkx([basin_id], river_graph)
        
        # Subset the basins and rivers
        subset_basins = basins[basins['streamID'].isin(upstream_basin_ids)]
        subset_rivers = rivers[rivers['LINKNO'].isin(upstream_basin_ids)]
        
        # Define the output file paths
        output_basins_path = os.path.join(output_dir, f"upstream_basins_point_{basin_id}.shp")
        output_rivers_path = os.path.join(output_dir, f"upstream_rivers_point_{basin_id}.shp")
        
        # Save the subset basins and rivers to shapefiles
        subset_basins.to_file(output_basins_path)
        subset_rivers.to_file(output_rivers_path)
        
        print(f"Upstream basins shapefile saved to: {output_basins_path}")
        print(f"Upstream rivers shapefile saved to: {output_rivers_path}")

####################################
### --- Main: Run subsetting --- ###
####################################

def main():
    #Find the required paths
    basins_path = read_from_control(controlFolder/controlFile,'basins_path')  
    rivers_path = read_from_control(controlFolder/controlFile,'rivers_path')
    pour_points_path = read_from_control(controlFolder/controlFile,'pour_points_path') 
    output_dir = read_from_control(controlFolder/controlFile,'output_dir')
    print(basins_path, rivers_path, pour_points_path, output_dir)
    os.makedirs(output_dir, exist_ok=True)

    #Load the basins, rivers and pourpoints shapefiles
    basins = gpd.read_file(basins_path)
    rivers = gpd.read_file(rivers_path)
    pour_points = gpd.read_file(pour_points_path)

    # Ensure the pour points shapefile contains an 'point_id' field
    if 'point_id' not in pour_points.columns:
        pour_points['point_id'] = range(len(pour_points))

    # Extract upstream data for each point
    extract_upstream_data(basins, rivers, pour_points, output_dir)

if __name__ == "__main__":
    main()