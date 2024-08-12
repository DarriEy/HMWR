# utils/input_data_processing.py

from pathlib import Path
import json
import pandas as pd # type: ignore
import numpy as np
import logging

def prepare_maf_json(config):
    """Prepare the JSON file for the Model Agnostic Framework."""
    maf_config = {
        "exec": {
            "met": str(Path(config.root_path) / "installs/datatool/extract-dataset.sh"),
            "gis": str(Path(config.root_path) / "installs/gistool/extract-gis.sh"),
            "remap": "easymore cli"
        },
        "args": {
            "met": [{
                "dataset": config.forcing_dataset,
                "dataset-dir": f"{config.datatool_dataset_root}/rdrsv2.1/",
                "variable": [
                    "RDRS_v2.1_P_P0_SFC",
                    "RDRS_v2.1_P_HU_09944",
                    "RDRS_v2.1_P_TT_09944",
                    "RDRS_v2.1_P_UVC_09944",
                    "RDRS_v2.1_A_PR0_SFC",
                    "RDRS_v2.1_P_FB_SFC",
                    "RDRS_v2.1_P_FI_SFC"
                ],
                "output-dir": str(Path(config.root_path) / f"domain_{config.domain_name}/forcing/1_raw_data"),
                "start-date": config.forcing_raw_time.split(',')[0] + "-01-01T13:00:00",
                "end-date": config.forcing_raw_time.split(',')[1] + "-12-31T12:00:00",
                "lat-lims": "",
                "lon-lims": "",
                "shape-file": f"{config.root_path}/domain_{config.domain_name}/shapefiles/catchment/{config.catchment_shp_name}",
                "model": "",
                "ensemble": "",
                "prefix": f"domain_{config.domain_name}_",
                "email": "",
                "cache": config.datatool_cache,
                "account": config.datatool_account,
                "_flags": [
                    "submit-job",
                    "parsable"
                ]
            }],
            "gis": [
                {
                    "dataset": "landsat",
                    "dataset-dir": f"{config.gistool_dataset_root}/Landsat",
                    "variable": "land-cover",
                    "start-date": "2020",
                    "end-date": "2020",
                    "output-dir": str(Path(config.root_path) / f"domain_{config.domain_name}/parameters/landclass"),
                    "lat-lims": "",
                    "lon-lims": "",
                    "shape-file": f"{config.root_path}/domain_{config.domain_name}/shapefiles/catchment/{config.catchment_shp_name}",
                    "print-geotiff": "true",
                    "stat": [
                        "frac",
                        "majority",
                        "coords"
                    ],
                    "quantile": "",
                    "lib-path": config.gistool_lib_path,
                    "cache": config.gistool_cache,
                    "prefix": f"domain_{config.domain_name}_",
                    "email": "",
                    "account": config.gistool_account,
                    "fid": config.river_basin_shp_rm_hruid,
                    "_flags": [
                        "include-na",
                        "submit-job",
                        "parsable"
                    ]
                },
                {
                    "dataset": "soil_class",
                    "dataset-dir": f"{config.gistool_dataset_root}/soil_classes",
                    "variable": "soil_classes",
                    "start-date": "",
                    "end-date": "",
                    "output-dir": str(Path(config.root_path) / f"domain_{config.domain_name}/parameters/soilclass"),
                    "lat-lims": "",
                    "lon-lims": "",
                    "cache": config.gistool_cache,
                    "shape-file": f"{config.root_path}/domain_{config.domain_name}/shapefiles/catchment/{config.catchment_shp_name}",
                    "print-geotiff": "true",
                    "stat": [
                        "majority"
                    ],
                    "quantile": "",
                    "prefix": f"domain_{config.domain_name}_",
                    "email": "",
                    "lib-path": config.gistool_lib_path,
                    "account": config.gistool_account,
                    "fid": config.river_basin_shp_rm_hruid,
                    "_flags": [
                        "include-na",
                        "submit-job",
                        "parsable"
                    ]
                },
                {
                    "dataset": "merit-hydro",
                    "dataset-dir": f"{config.gistool_dataset_root}/MERIT-Hydro",
                    "variable": "elv,hnd",
                    "start-date": "",
                    "end-date": "",
                    "output-dir":  str(Path(config.root_path) / f"domain_{config.domain_name}/parameters/dem"),
                    "lat-lims": "",
                    "lon-lims": "",
                    "lib-path": config.gistool_lib_path,
                    "cache": config.gistool_cache,
                    "account": config.gistool_account,
                    "shape-file": f"{config.root_path}/domain_{config.domain_name}/shapefiles/catchment/{config.catchment_shp_name}",
                    "print-geotiff": "true",
                    "stat": [
                        "min",
                        "max",
                        "mean",
                        "median"
                    ],
                    "prefix": f"domain_{config.domain_name}_",
                    "email": "",
                    "_flags": [
                        "include-na",
                        "submit-job",
                        "parsable"
                    ]
                }
            ],
            "remap": [{
                "case-name": "remapped",
                "cache": config.easymore_cache,
                "shapefile": f"{config.root_path}/domain_{config.domain_name}/shapefiles/catchment/{config.catchment_shp_name}",
                "shapefile-id": config.river_basin_shp_rm_hruid,
                "source-nc": str(Path(config.root_path) / f"domain_{config.domain_name}/forcing/1_raw_data/**/*.nc*"),
                "variable-lon": "lon",
                "variable-lat": "lat",
                "variable": [
                    "RDRS_v2.1_P_P0_SFC",
                    "RDRS_v2.1_P_HU_09944",
                    "RDRS_v2.1_P_TT_09944",
                    "RDRS_v2.1_P_UVC_09944",
                    "RDRS_v2.1_A_PR0_SFC",
                    "RDRS_v2.1_P_FB_SFC",
                    "RDRS_v2.1_P_FI_SFC"
                ],
                "remapped-var-id": config.river_basin_shp_rm_hruid,
                "remapped-dim-id": config.river_basin_shp_rm_hruid,
                "output-dir": f"{config.root_path}/domain_{config.domain_name}/forcing/3_basin_averaged_data" + '/',
                "job-conf": str(Path(config.root_path) / "installs/MAF/02_model_agnostic_component/easymore-job.slurm"),
                "_flags": [
                    "submit-job"
                ]
            }]
        },
        "order": {
            "met": 1,
            "gis": -1,
            "remap": 2
        }
    }

    # Save the JSON file
    json_path = Path(config.root_path) / f"domain_{config.domain_name}/maf_config.json"
    with open(json_path, 'w') as f:
        json.dump(maf_config, f, indent=2)

    return json_path

def cleanup_and_checks(config, logger):
    """Perform cleanup and checks on the MAF output."""
    logger.info("Performing cleanup and checks on MAF output")
    
    # Define paths
    path_general = Path(config.root_path) / f"domain_{config.domain_name}"
    path_soil_type = path_general / f'parameters/soilclass/domain_{config.domain_name}_stats_soil_classes.csv'
    path_landcover_type = path_general / f'parameters/landclass/domain_{config.domain_name}_stats_NA_NALCMS_landcover_2020_30m.csv'
    path_elevation_mean = path_general / f'parameters/dem/domain_{config.domain_name}_stats_elv.csv'

    # Read files
    soil_type = pd.read_csv(path_soil_type)
    landcover_type = pd.read_csv(path_landcover_type)
    elevation_mean = pd.read_csv(path_elevation_mean)

    # Sort by COMID
    soil_type = soil_type.sort_values(by='COMID').reset_index(drop=True)
    landcover_type = landcover_type.sort_values(by='COMID').reset_index(drop=True)
    elevation_mean = elevation_mean.sort_values(by='COMID').reset_index(drop=True)

    # Check if COMIDs are the same across all files
    if not (len(soil_type) == len(landcover_type) == len(elevation_mean) and
            (soil_type['COMID'] == landcover_type['COMID']).all() and
            (landcover_type['COMID'] == elevation_mean['COMID']).all()):
        raise ValueError("COMIDs are not consistent across soil, landcover, and elevation files")

    # Process soil type
    majority_value = soil_type['majority'].replace(0, np.nan).mode().iloc[0]
    soil_type['majority'] = soil_type['majority'].replace(0, majority_value).fillna(majority_value)
    if config.unify_soil:
        soil_type['majority'] = majority_value

    # Process landcover
    for col in landcover_type.columns:
        if col.startswith('frac_'):
            landcover_type[col] = landcover_type[col].apply(lambda x: 0 if x < config.minimume_land_fraction else x)
    
    for index, row in landcover_type.iterrows():
        frac_columns = [col for col in landcover_type.columns if col.startswith('frac_')]
        row_sum = row[frac_columns].sum()
        if row_sum > 0:
            for col in frac_columns:
                landcover_type.at[index, col] /= row_sum

    missing_columns = [f"frac_{i}" for i in range(1, config.num_land_cover+1) if f"frac_{i}" not in landcover_type.columns]
    for col in missing_columns:
        landcover_type[col] = 0

    frac_columns = [col for col in landcover_type.columns if col.startswith('frac_')]
    frac_columns.sort(key=lambda x: int(x.split('_')[1]))
    sorted_columns = [col for col in landcover_type.columns if col not in frac_columns] + frac_columns
    landcover_type = landcover_type.reindex(columns=sorted_columns)

    for col in frac_columns:
        if landcover_type.loc[0, col] < 0.00001:
            landcover_type.loc[0, col] = 0.00001

    # Process elevation
    elevation_mean['mean'].fillna(0, inplace=True)

    # Save modified files
    soil_type.to_csv(path_general / 'parameters/soilclass/modified_domain_stats_soil_classes.csv', index=False)
    landcover_type.to_csv(path_general / 'parameters/landclass/modified_domain_stats_NA_NALCMS_landcover_2020_30m.csv', index=False)
    elevation_mean.to_csv(path_general / 'parameters/dem/modified_domain_stats_elv.csv', index=False)

    logger.info("Cleanup and checks completed")