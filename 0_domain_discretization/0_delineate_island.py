import numpy as np
import os
from scipy import ndimage
import rasterio
from rasterio import features
import fiona
from shapely.geometry import shape, mapping
from pathlib import Path
from pysheds.grid import Grid
import matplotlib.pyplot as plt
from collections import deque

#####################################
### --- Control file handling --- ###
#####################################

# Easy access to control file folder
controlFolder = Path('../0_control_files')

# Store the name of the 'active' file in a variable
controlFile = 'control_Reunion.txt'

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
### --- Delineation Functions --- ###
#####################################


def flow_directions():
    """Return a dictionary of flow directions and their corresponding offsets."""
    return {
        1: (0, 1),   # E
        2: (1, 1),   # SE
        4: (1, 0),   # S
        8: (1, -1),  # SW
        16: (-1, -1), # W
        32: (-1, 0),  # NW
        64: (-1, 1),  # N
        128: (0, -1)  # NE
    }

def does_it_flow_into_me(flow_dir, i, j):
    """Check if the cell at (i,j) flows into the center cell."""
    directions = flow_directions()
    if flow_dir in directions:
        di, dj = directions[flow_dir]
        return (-di, -dj) == (i, j)
    return False

def delineate_watershed(fdir, start_x, start_y, island_mask):
    """Delineate a watershed starting from (start_x, start_y)."""
    height, width = fdir.shape
    watershed = np.zeros_like(fdir, dtype=bool)
    stack = deque([(start_y, start_x)])
    
    while stack:
        y, x = stack.pop()
        if watershed[y, x] or not island_mask[y, x]:
            continue
        watershed[y, x] = True
        
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == 0 and j == 0:
                    continue
                ny, nx = y + i, x + j
                if 0 <= ny < height and 0 <= nx < width and island_mask[ny, nx]:
                    if does_it_flow_into_me(fdir[ny, nx], i, j):
                        stack.append((ny, nx))
    
    return watershed

def delineate_all_watersheds(fdir, acc, island_mask, min_watershed_size=2):
    """Delineate all watersheds on the island."""
    pour_points = find_pour_points(fdir, acc, island_mask)
    watersheds = np.zeros_like(fdir, dtype=int)
    watershed_id = 1
    
    for y, x in zip(*np.where(pour_points)):
        if watersheds[y, x] == 0:
            watershed = delineate_watershed(fdir, x, y, island_mask)
            if np.sum(watershed) >= min_watershed_size:
                watersheds[watershed] = watershed_id
                watershed_id += 1
    
    return watersheds

def create_island_mask(dem):
    """Create a binary mask of the island from the DEM."""
    mask = dem > 0
    # Use morphological operations to clean up the mask
    mask = ndimage.binary_closing(mask)
    mask = ndimage.binary_opening(mask)
    return mask

def plot_raster(data, title, output_folder, filename, cmap='viridis', vmin=None, vmax=None):
    plt.figure(figsize=(10, 10))
    if data.dtype == bool:
        plt.imshow(data, cmap='binary')
    else:
        plt.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(label='Value')
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, filename), dpi=300, bbox_inches='tight')
    plt.close()

def find_pour_points(fdir, acc, island_mask, coastal_buffer_size=5, min_acc_threshold=100):
    """Find pour points including both coastal and interior watersheds."""
    height, width = fdir.shape
    
    # Identify coastal cells
    coastal = np.zeros_like(island_mask, dtype=bool)
    coastal[coastal_buffer_size:-coastal_buffer_size, coastal_buffer_size:-coastal_buffer_size] = (
        island_mask[coastal_buffer_size:-coastal_buffer_size, coastal_buffer_size:-coastal_buffer_size] & 
        ~island_mask[:-2*coastal_buffer_size, coastal_buffer_size:-coastal_buffer_size]
    )
    
    # Find local maxima in flow accumulation (potential interior pour points)
    acc_max = ndimage.maximum_filter(acc, size=3)
    local_max = (acc == acc_max) & (acc > min_acc_threshold)
    
    # Combine coastal and interior pour points
    pour_points = coastal | local_max
    
    return pour_points

def process_dem(dem_path, output_folder, output_prefix, min_watershed_size=5):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Read the DEM using pysheds
    print("Reading DEM...")
    grid = Grid.from_raster(dem_path)
    dem = grid.read_raster(dem_path)
    plot_raster(dem, "Original DEM", output_folder, "1_original_dem.png")

    # Create island mask
    print("Creating island mask...")
    island_mask = create_island_mask(dem)
    plot_raster(island_mask, "Island Mask", output_folder, "2_island_mask.png", cmap='binary')

    # Apply island mask to DEM
    dem_masked = grid.view(dem)
    dem_masked[~island_mask] = grid.nodata
    plot_raster(dem_masked, "Masked DEM", output_folder, "2_masked_dem.png")

    # Fill depressions
    print("Filling depressions...")
    filled_dem = grid.fill_depressions(dem_masked)
    plot_raster(filled_dem, "Filled DEM", output_folder, "3_filled_dem.png")

    # Calculate flow direction
    print("Calculating flow direction...")
    fdir = grid.flowdir(filled_dem)
    fdir[~island_mask] = grid.nodata
    plot_raster(fdir, "Flow Direction", output_folder, "4_flow_direction.png")

    # Calculate flow accumulation
    print("Calculating flow accumulation...")
    acc = grid.accumulation(fdir)
    acc[~island_mask] = 0
    plot_raster(np.log1p(acc), "Flow Accumulation (log scale)", output_folder, "5_flow_accumulation.png")

    print("Delineating watersheds...")
    watersheds = delineate_all_watersheds(fdir, acc, island_mask, min_watershed_size)
    
    print(f"Watersheds shape: {watersheds.shape}")
    print(f"Unique watershed labels: {np.unique(watersheds)}")
    

    # Remove small watersheds
    unique, counts = np.unique(watersheds, return_counts=True)
    for label, count in zip(unique, counts):
        if count < min_watershed_size and label != 0:
            watersheds[watersheds == label] = 0

    # Relabel watersheds
    watersheds, num_watersheds = ndimage.label(watersheds > 0)
    print(f"Number of watersheds after filtering: {num_watersheds}")
    print(f"Unique watershed labels after filtering: {np.unique(watersheds)}")

    plot_raster(watersheds, "Watersheds", output_folder, "6_watersheds.png", cmap='tab20')

    # Save watersheds as shapefile
    print("Saving watersheds as shapefile...")
    watershed_shapes = features.shapes(watersheds.astype('int32'), mask=watersheds > 0, transform=grid.affine)
    
    watershed_schema = {
        'geometry': 'Polygon',
        'properties': {'id': 'int'}
    }
    
    output_shapefile = os.path.join(output_folder, f"{output_prefix}_watersheds.shp")
    with fiona.open(output_shapefile, 'w', driver='ESRI Shapefile', crs=grid.crs.srs, schema=watershed_schema) as shp:
        for i, (geom, value) in enumerate(watershed_shapes):
            shp.write({
                'geometry': geom,
                'properties': {'id': int(value)}
            })

    print("Processing completed successfully")

#####################################
### --- Main: Run delineation --- ###
#####################################

def main():
    dem_path = read_from_control(controlFolder/controlFile, 'domain_dem_path')
    output_folder = read_from_control(controlFolder/controlFile, 'output_folder')
    output_prefix = read_from_control(controlFolder/controlFile, 'output_prefix')
    min_watershed_size = int(read_from_control(controlFolder/controlFile, 'min_watershed_size'))

    print(f"Processing DEM: {dem_path}")
    print(f"Output folder: {output_folder}")
    print(f"Output prefix: {output_prefix}")
    print(f"Minimum watershed size: {min_watershed_size}")
    
    process_dem(dem_path, output_folder, output_prefix, min_watershed_size)
 
if __name__ == "__main__":
    main()
