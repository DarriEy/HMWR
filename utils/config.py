# config.py

from dataclasses import dataclass
from typing import List, Tuple
from datetime import datetime
from pathlib import Path
from mpi4py import MPI # type: ignore
import numpy as np # type: ignore
from utils.calibration_utils import read_param_bounds # type: ignore
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

@dataclass
class Config:
    root_path: str
    domain_name: str
    experiment_id: str
    params_to_calibrate: List[str]
    basin_params_to_calibrate: List[str]
    obs_file_path: str
    sim_reach_ID: str
    filemanager_name: str
    optimization_metrics: List[str]
    pop_size: int
    mizu_control_file: str
    moo_num_iter: int
    nsga2_n_gen: int
    nsga2_n_obj: int
    optimization_metric: str
    algorithm: str
    num_iter: int
    calib_period: Tuple[datetime, datetime]
    eval_period: Tuple[datetime, datetime]
    local_bounds_dict: dict
    basin_bounds_dict: dict
    local_bounds: List[Tuple[float, float]]
    basin_bounds: List[Tuple[float, float]]
    all_bounds: List[Tuple[float, float]]
    all_params: List[str]
    poplsize: int
    swrmsize: int
    ngsize: int
    dds_r: float
    diagnostic_frequency: int

def initialize_config(rank: int, comm: MPI.Comm) -> Config:
    control_folder = Path('../../0_control_files')
    control_file = 'control_active.txt'

    if rank == 0:
        root_path = read_from_control(control_folder/control_file, 'root_path')
        domain_name = read_from_control(control_folder/control_file, 'domain_name')
        experiment_id = read_from_control(control_folder/control_file, 'experiment_id')
        params_to_calibrate = read_from_control(control_folder/control_file, 'params_to_calibrate').split(',')
        basin_params_to_calibrate = read_from_control(control_folder/control_file, 'basin_params_to_calibrate').split(',')
        obs_file_path = read_from_control(control_folder/control_file, 'obs_file_path')
        sim_reach_ID = read_from_control(control_folder/control_file, 'sim_reach_ID')
        filemanager_name = read_from_control(control_folder/control_file, 'settings_summa_filemanager')
        mizu_control_file = read_from_control(control_folder/control_file, 'settings_mizu_control_file')
        optimization_metric = read_from_control(control_folder/control_file, 'optimization_metric')
        algorithm = read_from_control(control_folder/control_file, 'Optimisation_algorithm')
        num_iter = int(read_from_control(control_folder/control_file, 'num_iter'))
        calib_period_str = read_from_control(control_folder/control_file, 'calibration_period')
        eval_period_str = read_from_control(control_folder/control_file, 'evaluation_period')
        calib_period = parse_time_period(calib_period_str)
        eval_period = parse_time_period(eval_period_str)
        optimization_metrics = read_from_control(control_folder/control_file, 'moo_optimization_metrics').split(',')
        pop_size = int(read_from_control(control_folder/control_file, 'moo_pop_size'))
        moo_num_iter = int(read_from_control(control_folder/control_file, 'moo_num_iter'))
        nsga2_n_gen = int(read_from_control(control_folder/control_file, 'nsga2_n_gen'))
        nsga2_n_obj = int(read_from_control(control_folder/control_file, 'nsga2_n_obj'))
        local_parameters_file = get_config_path(control_folder, control_file, 'local_parameters_file', 'settings/SUMMA/localParamInfo.txt')
        basin_parameters_file = get_config_path(control_folder, control_file, 'basin_parameters_file', 'settings/SUMMA/basinParamInfo.txt')
        
        local_bounds_dict = read_param_bounds(local_parameters_file, params_to_calibrate)
        basin_bounds_dict = read_param_bounds(basin_parameters_file, basin_params_to_calibrate)
        local_bounds = [local_bounds_dict[param] for param in params_to_calibrate]
        basin_bounds = [basin_bounds_dict[param] for param in basin_params_to_calibrate]
        all_bounds = local_bounds + basin_bounds
        all_params = params_to_calibrate + basin_params_to_calibrate

        poplsize = int(read_from_control(control_folder/control_file, 'poplsize'))
        swrmsize = int(read_from_control(control_folder/control_file, 'swrmsize'))
        ngsize = int(read_from_control(control_folder/control_file, 'ngsize'))
        dds_r = float(read_from_control(control_folder/control_file, 'dds_r'))
        diagnostic_frequency = int(read_from_control(control_folder/control_file, 'diagnostic_frequency'))

    else:
        root_path = None
        domain_name = None
        experiment_id = None
        params_to_calibrate = None
        basin_params_to_calibrate = None
        obs_file_path = None
        sim_reach_ID = None
        moo_num_iter = None
        nsga2_n_gen = None
        nsga2_n_obj = None
        filemanager_name = None
        mizu_control_file = None
        optimization_metric = None
        algorithm = None
        optimization_metrics = None
        pop_size = None
        num_iter = None
        calib_period = None
        eval_period = None
        local_bounds_dict = None
        basin_bounds_dict = None
        local_bounds = None
        basin_bounds = None
        all_bounds = None
        all_params = None
        poplsize = None
        swrmsize = None
        ngsize = None
        dds_r = None
        diagnostic_frequency = None

    config = Config(
        root_path=comm.bcast(root_path, root=0),
        domain_name=comm.bcast(domain_name, root=0),
        experiment_id=comm.bcast(experiment_id, root=0),
        params_to_calibrate=comm.bcast(params_to_calibrate, root=0),
        basin_params_to_calibrate=comm.bcast(basin_params_to_calibrate, root=0),
        obs_file_path=comm.bcast(obs_file_path, root=0),
        sim_reach_ID=comm.bcast(sim_reach_ID, root=0),
        filemanager_name=comm.bcast(filemanager_name, root=0),
        mizu_control_file=comm.bcast(mizu_control_file, root=0),
        optimization_metric=comm.bcast(optimization_metric, root=0),
        algorithm=comm.bcast(algorithm, root=0),
        num_iter=comm.bcast(num_iter, root=0),
        moo_num_iter=comm.bcast(moo_num_iter, root=0),
        nsga2_n_gen=comm.bcast(nsga2_n_gen, root=0),
        nsga2_n_obj=comm.bcast(nsga2_n_obj, root=0),
        optimization_metrics=comm.bcast(optimization_metrics, root=0),
        pop_size=comm.bcast(pop_size, root=0),
        calib_period=comm.bcast(calib_period, root=0),
        eval_period=comm.bcast(eval_period, root=0),
        local_bounds_dict=comm.bcast(local_bounds_dict, root=0),
        basin_bounds_dict=comm.bcast(basin_bounds_dict, root=0),
        local_bounds=comm.bcast(local_bounds, root=0),
        basin_bounds=comm.bcast(basin_bounds, root=0),
        all_bounds=comm.bcast(all_bounds, root=0),
        all_params=comm.bcast(all_params, root=0),
        poplsize=comm.bcast(poplsize, root=0),
        swrmsize=comm.bcast(swrmsize, root=0),
        ngsize=comm.bcast(ngsize, root=0),
        dds_r=comm.bcast(dds_r, root=0),
        diagnostic_frequency=comm.bcast(diagnostic_frequency,root=0)
    )

    return config

@dataclass
class preConfig:
    root_path: Path
    domain_name: str
    catchment_shp_path: str
    river_network_shp_path: str
    river_basin_shp_path: str
    river_basin_shp_name: str
    pour_point_shp_path: str
    source_control_file: str
    pour_point_coords: str
    domain_subsetting:str
    full_domain_name: str
    experiment_id: str
    parameter_dem_tif_name: str
    stream_order_threshold: int
    fullDom_basins_name: str
    fullDom_rivers_name: str
    elevation_band_size: float
    min_hru_size: float
    radiation_class_number: int
    domain_discretisation: str
    pour_point_shp_name: str

    @classmethod
    def from_control_file(cls, control_file: Path):
        def read_from_control(file, setting):
            with open(file) as contents:
                for line in contents:
                    if setting in line and not line.startswith('#'):
                        return line.split('|', 1)[1].split('#', 1)[0].strip()
            return None

        root_path = Path(read_from_control(control_file, 'root_path'))
        domain_name = read_from_control(control_file, 'domain_name')
        catchment_shp_path = read_from_control(control_file, 'catchment_shp_path')
        river_network_shp_path = read_from_control(control_file, 'river_network_shp_path')
        river_basin_shp_path = read_from_control(control_file, 'river_basin_shp_path')
        pour_point_shp_path = read_from_control(control_file, 'pour_point_shp_path')
        source_control_file = control_file.name
        pour_point_coords = read_from_control(control_file, 'pour_point_coords')
        domain_subsetting = read_from_control(control_file, 'domain_subsetting')
        full_domain_name = read_from_control(control_file, 'full_domain_name')
        experiment_id = read_from_control(control_file, 'experiment_id')
        parameter_dem_tif_name = read_from_control(control_file, 'parameter_dem_tif_name')
        stream_order_threshold = int(read_from_control(control_file, 'stream_order_threshold'))
        fullDom_basins_name = read_from_control(control_file, 'fullDom_basins_name')
        fullDom_rivers_name = read_from_control(control_file, 'fullDom_rivers_name')
        elevation_band_size = int(read_from_control(control_file, 'elevation_band_size'))
        min_hru_size = float(read_from_control(control_file, 'min_hru_size'))
        radiation_class_number = int(read_from_control(control_file, 'radiation_class_number'))
        domain_discretisation = read_from_control(control_file, 'domain_discretisation')
        river_basin_shp_name = read_from_control(control_file, 'river_basin_shp_name')
        pour_point_shp_name = read_from_control(control_file, 'pour_point_shp_name')

        return cls(
            root_path=root_path,
            domain_name=domain_name,
            full_domain_name=full_domain_name,
            experiment_id=experiment_id,
            catchment_shp_path=catchment_shp_path,
            river_network_shp_path=river_network_shp_path,
            river_basin_shp_path=river_basin_shp_path,
            pour_point_shp_path=pour_point_shp_path,
            source_control_file=source_control_file,
            pour_point_coords=pour_point_coords,
            domain_subsetting=domain_subsetting,
            parameter_dem_tif_name=parameter_dem_tif_name,
            stream_order_threshold=stream_order_threshold,
            fullDom_basins_name=fullDom_basins_name,
            fullDom_rivers_name=fullDom_rivers_name,
            elevation_band_size=elevation_band_size,
            min_hru_size=min_hru_size,
            radiation_class_number=radiation_class_number,
            domain_discretisation=domain_discretisation, 
            river_basin_shp_name=river_basin_shp_name,
            pour_point_shp_name=pour_point_shp_name
        )
   