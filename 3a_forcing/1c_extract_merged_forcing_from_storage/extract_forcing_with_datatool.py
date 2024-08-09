import sys
from pathlib import Path
import subprocess

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.control_file_utils import read_from_control, make_default_path # type: ignore

# Easy access to control file folder
controlFolder = Path('../../0_control_files')
controlFile = 'control_active.txt'

# Read necessary settings from the control file
datatool_account = read_from_control(controlFolder/controlFile, 'datatool_account')
datatool_output_dir = read_from_control(controlFolder/controlFile, 'datatool_output-dir')
datatool_cache = read_from_control(controlFolder/controlFile, 'datatool_cache')
datatool_prefix = read_from_control(controlFolder/controlFile, 'datatool_prefix')
datatool_path = read_from_control(controlFolder/controlFile, 'datatool_path')
domain_name = read_from_control(controlFolder/controlFile, 'domain_name')
forcing_dataset = read_from_control(controlFolder/controlFile, 'forcing_dataset')
forcing_raw_time = read_from_control(controlFolder/controlFile, 'forcing_raw_time')
forcing_raw_space = read_from_control(controlFolder/controlFile, 'forcing_raw_space')
forcing_variables = read_from_control(controlFolder/controlFile, f'forcing_{forcing_dataset}_variables')

# Process the settings
start_date, end_date = forcing_raw_time.split(',')
start_date = f"{start_date}-01-01 00:00:00"
end_date = f"{end_date}-12-31 23:00:00"

lat_max, lon_min, lat_min, lon_max = forcing_raw_space.split('/')
lat_lims = f"{lat_min},{lat_max}"
lon_lims = f"{lon_min},{lon_max}"

if datatool_output_dir == 'default':
    datatool_output_dir = make_default_path(controlFolder, controlFile, Path('forcing') / '1_raw_data')

if datatool_cache == 'default':
    datatool_cache = '$SLURM_TMPDIR'

if datatool_prefix == 'default':
    datatool_prefix = f"{domain_name}_{forcing_dataset}_"

if datatool_path == 'default':
    datatool_path = make_default_path(controlFolder, controlFile, Path('installs') / 'datatool')

# Construct the datatool command
datatool_command = [
    f"{datatool_path}/extract-dataset.sh",
    f"--dataset={forcing_dataset}",
    f"--dataset-dir=/project/6079554/data/meteorological-data/{forcing_dataset}",
    f"--output-dir={datatool_output_dir}",
    f"--start-date={start_date}",
    f"--end-date={end_date}",
    f"--lat-lims={lat_lims}",
    f"--lon-lims={lon_lims}",
    f"--variable={forcing_variables}",
    f"--prefix={datatool_prefix}",
    "--submit-job",
    f"--cache={datatool_cache}",
    f"--account={datatool_account}"
]

print(datatool_command)
# Execute the datatool command
try:
    result = subprocess.run(datatool_command, check=True, capture_output=True, text=True)
    print("Datatool command executed successfully.")
    print("Output:", result.stdout)
except subprocess.CalledProcessError as e:
    print("Error executing datatool command:")
    print("Exit code:", e.returncode)
    print("Error output:", e.stderr)