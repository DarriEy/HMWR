from pathlib import Path
from datetime import datetime

def read_from_control(file, setting):
    """Extract a given setting from the control file."""
    with open(file) as contents:
        for line in contents:
            if setting in line and not line.startswith('#'):
                return line.split('|', 1)[1].split('#', 1)[0].strip()
    return None

def make_default_path(control_folder, control_file, suffix):
    """Specify a default path based on the control file settings."""
    root_path = Path(read_from_control(control_folder/control_file, 'root_path'))
    domain_name = read_from_control(control_folder/control_file, 'domain_name')
    return root_path / f'domain_{domain_name}' / suffix

def parse_time_period(period_str):
    """Parse time period string into datetime objects."""
    start, end = [datetime.strptime(date.strip(), '%Y-%m-%d') for date in period_str.split(',')]
    return start, end

def get_config_path(control_folder, control_file, setting, default_suffix=None, is_folder=False):
    """
    Get a configuration path for a file or folder, using a default if specified.
    
    Args:
    control_folder (Path): Path to the folder containing the control file
    control_file (str): Name of the control file
    setting (str): The setting to read from the control file
    default_suffix (str, optional): The suffix to append to the default path if 'default' is specified
    is_folder (bool): Whether the path is for a folder (True) or file (False)
    
    Returns:
    Path: The configuration path
    """
    path = read_from_control(control_folder/control_file, setting)
    if path == 'default':
        if is_folder:
            root_path = Path(read_from_control(control_folder/control_file, 'root_code_path'))
            return root_path / default_suffix
        else:
            return make_default_path(control_folder, control_file, default_suffix)
    return Path(path)