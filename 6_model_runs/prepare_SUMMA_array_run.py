import geopandas as gpd
from pathlib import Path
import math
import re

# Control file handling (reused from your example)
controlFolder = Path('../0_control_files')
controlFile = 'control_active.txt'

def read_from_control(file, setting):
    with open(file) as contents:
        for line in contents:
            if setting in line and not line.startswith('#'):
                return line.split('|', 1)[1].split('#', 1)[0].strip()
    return None

def make_default_path(suffix):
    rootPath = Path(read_from_control(controlFolder/controlFile, 'root_path'))
    domainName = read_from_control(controlFolder/controlFile, 'domain_name')
    return rootPath / f'domain_{domainName}' / suffix

# Read necessary settings
root_path = Path(read_from_control(controlFolder/controlFile, 'root_path'))
domain_name = read_from_control(controlFolder/controlFile, 'domain_name')
catchment_shp_path = read_from_control(controlFolder/controlFile, 'catchment_shp_path')
catchment_shp_name = read_from_control(controlFolder/controlFile, 'catchment_shp_name')

# Construct full path to shapefile
if catchment_shp_path == 'default':
    shp_path = make_default_path('shapefiles/catchment') / catchment_shp_name
else:
    shp_path = Path(catchment_shp_path) / catchment_shp_name

# Read the shapefile and count HRUs
gdf = gpd.read_file(shp_path)
gru_max = len(gdf)

# Set gru_count (you might want to read this from the control file or set it manually)
gru_count = int(read_from_control(controlFolder/controlFile, 'gru_count_per_run'))  # Example value, adjust as needed

# Calculate number of array jobs needed
num_arrays = math.ceil(gru_max / gru_count)

# Read the existing .sh script
sh_script_path = Path('1_run_summa_as_array_with_actors_model.sh')
with open(sh_script_path, 'r') as f:
    sh_script = f.read()

# Update the array line
sh_script = re.sub(r'#SBATCH --array=.*', f'#SBATCH --array=0-{num_arrays - 1}', sh_script)

# Update gru_max and gru_count
sh_script = re.sub(r'gru_max=.*', f'gru_max={gru_max}', sh_script)
sh_script = re.sub(r'gru_count=.*', f'gru_count={gru_count}', sh_script)

# Write the updated script back
with open(sh_script_path, 'w') as f:
    f.write(sh_script)

print(f"SLURM script has been updated: {sh_script_path}")
print(f"Total number of HRUs (gru_max): {gru_max}")
print(f"Number of HRUs per job (gru_count): {gru_count}")
print(f"Number of array jobs: {num_arrays}")