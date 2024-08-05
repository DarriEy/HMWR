import os
import glob
import easymore
from pathlib import Path
from shutil import rmtree, copyfile
from datetime import datetime
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from utils.control_file_utils import read_from_control, make_default_path

# --- Control file handling
controlFolder = Path('../../0_control_files')
controlFile = 'control_active.txt'

# --- Get forcing dataset
forcing_dataset = read_from_control(controlFolder/controlFile, 'forcing_dataset').lower()

# --- Set dataset-specific parameters
if forcing_dataset == 'era5':
    forcing_shape_name_key = 'forcing_shape_name'
    lat_name = 'latitude'
    lon_name = 'longitude'
elif forcing_dataset == 'rdrs':
    forcing_shape_name_key = 'forcing_rdrs_shape_name'
    lat_name = 'lat'
    lon_name = 'lon'
elif forcing_dataset == 'carra':
    forcing_shape_name_key = 'forcing_carra_shape_name'
    lat_name = 'latitude'
    lon_name = 'longitude'
else:
    raise ValueError(f"Unsupported forcing dataset: {forcing_dataset}")

# --- Find location of shapefiles
catchment_path = read_from_control(controlFolder/controlFile, 'intersect_dem_path')
catchment_name = read_from_control(controlFolder/controlFile, 'intersect_dem_name')
catchment_path = Path(catchment_path) if catchment_path != 'default' else make_default_path(controlFolder, controlFile, 'shapefiles/catchment_intersection/with_dem')

forcing_shape_path = read_from_control(controlFolder/controlFile, 'forcing_shape_path')
forcing_shape_name = read_from_control(controlFolder/controlFile, forcing_shape_name_key)
forcing_shape_path = Path(forcing_shape_path) if forcing_shape_path != 'default' else make_default_path(controlFolder, controlFile, 'shapefiles/forcing')

# --- Find where the intersection needs to go
intersect_path = read_from_control(controlFolder/controlFile, 'intersect_forcing_path')
intersect_path = Path(intersect_path) if intersect_path != 'default' else make_default_path(controlFolder, controlFile, 'shapefiles/catchment_intersection/with_forcing')
intersect_path.mkdir(parents=True, exist_ok=True)

# --- Find the forcing files
forcing_merged_path = read_from_control(controlFolder/controlFile, 'forcing_merged_path')
forcing_merged_path = Path(forcing_merged_path) if forcing_merged_path != 'default' else make_default_path(controlFolder, controlFile, 'forcing/2_merged_data')

if forcing_dataset == 'rdrs':
    forcing_files = [forcing_merged_path/file for file in os.listdir(forcing_merged_path) if file.startswith('RDRS_monthly_') and file.endswith('.nc')]
elif forcing_dataset == 'carra':
    forcing_files = [forcing_merged_path/file for file in os.listdir(forcing_merged_path) if file.startswith('CARRA_processed_carra_iceland_') and file.endswith('.nc')]
else:
    forcing_files = [forcing_merged_path/file for file in os.listdir(forcing_merged_path) if file.endswith('.nc')]
forcing_files.sort()

# --- Find where the temporary EASYMORE files need to go
forcing_easymore_path = read_from_control(controlFolder/controlFile, 'forcing_easymore_path')
forcing_easymore_path = Path(forcing_easymore_path) if forcing_easymore_path != 'default' else make_default_path(controlFolder, controlFile, 'forcing/3_temp_easymore')
forcing_easymore_path.mkdir(parents=True, exist_ok=True)

# --- Find where the area-weighted forcing needs to go
forcing_basin_path = read_from_control(controlFolder/controlFile, 'forcing_basin_avg_path')
forcing_basin_path = Path(forcing_basin_path) if forcing_basin_path != 'default' else make_default_path(controlFolder, controlFile, 'forcing/3_basin_averaged_data')
forcing_basin_path.mkdir(parents=True, exist_ok=True)

# --- EASYMORE
esmr = easymore.easymore()
esmr.author_name = 'SUMMA public workflow scripts'
esmr.license = 'Copernicus data use license: https://cds.climate.copernicus.eu/api/v2/terms/static/licence-to-use-copernicus-products.pdf'
esmr.case_name = read_from_control(controlFolder/controlFile, 'domain_name') + '_' + forcing_dataset

esmr.source_shp = forcing_shape_path/forcing_shape_name
esmr.source_shp_lat = read_from_control(controlFolder/controlFile, 'forcing_shape_lat_name')
esmr.source_shp_lon = read_from_control(controlFolder/controlFile, 'forcing_shape_lon_name')

esmr.target_shp = catchment_path/catchment_name
esmr.target_shp_ID = read_from_control(controlFolder/controlFile, 'catchment_shp_hruid')
esmr.target_shp_lat = read_from_control(controlFolder/controlFile, 'catchment_shp_lat')
esmr.target_shp_lon = read_from_control(controlFolder/controlFile, 'catchment_shp_lon')

esmr.source_nc = str(forcing_files[0])
esmr.var_names = ['airpres', 'LWRadAtm', 'SWRadAtm', 'pptrate', 'airtemp', 'spechum', 'windspd']
esmr.var_lat = lat_name
esmr.var_lon = lon_name
esmr.var_time = 'time'

esmr.temp_dir = str(forcing_easymore_path) + '/'
esmr.output_dir = str(forcing_basin_path) + '/'

esmr.remapped_dim_id = 'hru'
esmr.remapped_var_id = 'hruId'
esmr.format_list = ['f4']
esmr.fill_value_list = ['-9999']

esmr.save_csv = False
esmr.remap_csv = ''
esmr.sort_ID = False

# Run EASYMORE
esmr.nc_remapper()

# --- Move files to prescribed locations
remap_file = esmr.case_name + '_remapping.csv'
copyfile(esmr.temp_dir + remap_file, intersect_path / remap_file)

for file in glob.glob(esmr.temp_dir + esmr.case_name + '_intersected_shapefile.*'):
    copyfile(file, intersect_path / os.path.basename(file))

# Remove the temporary EASYMORE directory
try:
    rmtree(esmr.temp_dir)
except OSError as e:
    print(f"Error: {e.filename} - {e.strerror}.")



# --- Code provenance
def create_log(path, suffix, script_name):
    log_folder = path / '_workflow_log'
    log_folder.mkdir(parents=True, exist_ok=True)
    copyfile(script_name, log_folder / script_name)
    
    now = datetime.now()
    log_file = now.strftime('%Y%m%d') + suffix
    with open(log_folder / log_file, 'w') as file:
        lines = [
            f"Log generated by {script_name} on {now.strftime('%Y/%m/%d %H:%M:%S')}\n",
            f"Processed {forcing_dataset.upper()} forcing data."
        ]
        file.writelines(line + '\n' for line in lines)

create_log(intersect_path, '_catchment_forcing_intersect_log.txt', '1_make_one_weighted_forcing_file.py')
create_log(forcing_basin_path, '_create_one_weighted_forcing_file_log.txt', '1_make_one_weighted_forcing_file.py')

print(f"Completed processing of {forcing_dataset.upper()} forcing data.")