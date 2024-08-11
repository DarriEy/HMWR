from pathlib import Path
from typing import Dict, Any
import sys
import subprocess
import json
import numpy as np # type: ignore
import pandas as pd # type: ignore
import re

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.logging_utils import get_logger # type: ignore
from utils.config import preConfig # type: ignore
from utils.geospatial_processing import (create_pour_point_shapefile, delineate_grus, subset_merit_hydrofabrics, subset_TDX_hydrofabrics, subset_NWS_hydrofabrics,find_bounding_box_coordinates,generate_hrus,prepare_hru_parameters) # type: ignore
from utils.file_utils import make_folder_structure # type: ignore
from utils.summaflow import write_summa_attribute, write_summa_forcing, write_summa_paramtrial, write_summa_initial_conditions, write_summa_filemanager, copy_summa_static_files # type: ignore
from utils.observation_preprocessing import preprocess_modis_data # type: ignore

class preProcessor:
    def __init__(self, config: 'Config'): # type: ignore
        self.config = config
        self.logger = get_logger('PreProcessor', self.config.root_path, self.config.domain_name, 'preprocessing')


    def prepare_maf_json(self):
        """Prepare the JSON file for the Model Agnostic Framework."""
        maf_config = {
            "exec": {
                "met": str(Path(self.config.root_path) / "installs/datatool/extract-dataset.sh"),
                "gis": str(Path(self.config.root_path) / "installs/gistool/extract-gis.sh"),
                "remap": "easymore cli"
            },
            "args": {
                "met": [{
                    "dataset": self.config.forcing_dataset,
                    "dataset-dir": f"{self.config.datatool_dataset_root}/rdrsv2.1/",
                    "variable": [
                        "RDRS_v2.1_P_P0_SFC",
                        "RDRS_v2.1_P_HU_09944",
                        "RDRS_v2.1_P_TT_09944",
                        "RDRS_v2.1_P_UVC_09944",
                        "RDRS_v2.1_A_PR0_SFC",
                        "RDRS_v2.1_P_FB_SFC",
                        "RDRS_v2.1_P_FI_SFC"
                    ],
                    "output-dir": str(Path(self.config.root_path) / f"domain_{self.config.domain_name}/forcing/1_raw_data"),
                    "start-date": self.config.forcing_raw_time.split(',')[0] + "-01-01T13:00:00",
                    "end-date": self.config.forcing_raw_time.split(',')[1] + "-12-31T12:00:00",
                    "lat-lims": "",
                    "lon-lims": "",
                    "shape-file": f"{self.config.root_path}/domain_{self.config.domain_name}/shapefiles/catchment/{self.config.catchment_shp_name}",
                    "model": "",
                    "ensemble": "",
                    "prefix": f"domain_{self.config.domain_name}_",
                    "email": "",
                    "cache": self.config.datatool_cache,
                    "account": self.config.datatool_account,
                    "_flags": [
                        "submit-job",
                        "parsable"
                    ]
                }],
                "gis": [
                    {
                        "dataset": "landsat",
                        "dataset-dir": f"{self.config.gistool_dataset_root}/Landsat",
                        "variable": "land-cover",
                        "start-date": "2020",
                        "end-date": "2020",
                        "output-dir": str(Path(self.config.root_path) / f"domain_{self.config.domain_name}/parameters/landclass"),
                        "lat-lims": "",
                        "lon-lims": "",
                        "shape-file": f"{self.config.root_path}/domain_{self.config.domain_name}/shapefiles/catchment/{self.config.catchment_shp_name}",
                        "print-geotiff": "true",
                        "stat": [
                            "frac",
                            "majority",
                            "coords"
                        ],
                        "quantile": "",
                        "lib-path": self.config.gistool_lib_path,
                        "cache": self.config.gistool_cache,
                        "prefix": f"domain_{self.config.domain_name}_",
                        "email": "",
                        "account": self.config.gistool_account,
                        "fid": self.config.river_basin_shp_rm_hruid,
                        "_flags": [
                            "include-na",
                            "submit-job",
                            "parsable"
                        ]
                    },
                    {
                        "dataset": "soil_class",
                        "dataset-dir": f"{self.config.gistool_dataset_root}/soil_classes",
                        "variable": "soil_classes",
                        "start-date": "",
                        "end-date": "",
                        "output-dir": str(Path(self.config.root_path) / f"domain_{self.config.domain_name}/parameters/soilclass"),
                        "lat-lims": "",
                        "lon-lims": "",
                        "cache": self.config.gistool_cache,
                        "shape-file": f"{self.config.root_path}/domain_{self.config.domain_name}/shapefiles/catchment/{self.config.catchment_shp_name}",
                        "print-geotiff": "true",
                        "stat": [
                            "majority"
                        ],
                        "quantile": "",
                        "prefix": f"domain_{self.config.domain_name}_",
                        "email": "",
                        "lib-path": self.config.gistool_lib_path,
                        "account": self.config.gistool_account,
                        "fid": self.config.river_basin_shp_rm_hruid,
                        "_flags": [
                            "include-na",
                            "submit-job",
                            "parsable"
                        ]
                    },
                    {
                        "dataset": "merit-hydro",
                        "dataset-dir": f"{self.config.gistool_dataset_root}/MERIT-Hydro",
                        "variable": "elv,hnd",
                        "start-date": "",
                        "end-date": "",
                        "output-dir":  str(Path(self.config.root_path) / f"domain_{self.config.domain_name}/parameters/dem"),
                        "lat-lims": "",
                        "lon-lims": "",
                        "lib-path": self.config.gistool_lib_path,
                        "cache": self.config.gistool_cache,
                        "account": self.config.gistool_account,
                        "shape-file": f"{self.config.root_path}/domain_{self.config.domain_name}/shapefiles/catchment/{self.config.catchment_shp_name}",
                        "print-geotiff": "true",
                        "stat": [
                            "min",
                            "max",
                            "mean",
                            "median"
                        ],
                        "prefix": f"domain_{self.config.domain_name}_",
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
                    "cache": self.config.easymore_cache,
                    "shapefile": f"{self.config.root_path}/domain_{self.config.domain_name}/shapefiles/catchment/{self.config.catchment_shp_name}",
                    "shapefile-id": self.config.river_basin_shp_rm_hruid,
                    "source-nc": str(Path(self.config.root_path) / f"domain_{self.config.domain_name}/forcing/1_raw_data/**/*.nc*"),
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
                    "remapped-var-id": self.config.river_basin_shp_rm_hruid,
                    "remapped-dim-id": self.config.river_basin_shp_rm_hruid,
                    "output-dir": f"{self.config.root_path}/domain_{self.config.domain_name}/forcing/3_basin_averaged_data" + '/',
                    "job-conf": str(Path(self.config.root_path) / "installs/MAF/02_model_agnostic_component/easymore-job.slurm"),
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
        json_path = Path(self.config.root_path) / f"domain_{self.config.domain_name}/maf_config.json"
        with open(json_path, 'w') as f:
            json.dump(maf_config, f, indent=2)

        return json_path

    def run_maf(self):
        """Run the Model Agnostic Framework."""
        #json_path = self.prepare_maf_json()
        #maf_script = Path(self.config.root_path) / "installs/MAF/02_model_agnostic_component/model-agnostic.sh"
        
        # Run the MAF script
        #try:
            #subprocess.run([str(maf_script), str(json_path)], check=True)
            #self.logger.info("Model Agnostic Framework completed successfully.")
        #except subprocess.CalledProcessError as e:
            #self.logger.error(f"Error running Model Agnostic Framework: {e}")
            #raise

        self.cleanup_and_checks()
        #self.write_summa_files()

    def cleanup_and_checks(self):
        """Perform cleanup and checks on the MAF output."""
        self.logger.info("Performing cleanup and checks on MAF output")
        
        # Define paths
        path_general = Path(self.config.root_path) / f"domain_{self.config.domain_name}"
        path_soil_type = path_general / 'parameters/soilclass/domain_stats_soil_classes.csv'
        path_landcover_type = path_general / 'parameters/landclass/domain_stats_NA_NALCMS_landcover_2020_30m.csv'
        path_elevation_mean = path_general / 'parameters/dem/domain_stats_elv.csv'

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
        if self.config.unify_soil:
            soil_type['majority'] = majority_value

        # Process landcover
        for col in landcover_type.columns:
            if col.startswith('frac_'):
                landcover_type[col] = landcover_type[col].apply(lambda x: 0 if x < self.config.minimume_land_fraction else x)
        
        for index, row in landcover_type.iterrows():
            frac_columns = [col for col in landcover_type.columns if col.startswith('frac_')]
            row_sum = row[frac_columns].sum()
            if row_sum > 0:
                for col in frac_columns:
                    landcover_type.at[index, col] /= row_sum

        missing_columns = [f"frac_{i}" for i in range(1, self.config.num_land_cover+1) if f"frac_{i}" not in landcover_type.columns]
        for col in missing_columns:
            landcover_type[col] = 0

        frac_columns = [col for col in landcover_type.columns if re.match(r'^frac_\d+$', col)]
        frac_columns.sort(key=lambda x: int(re.search(r'\d+$', x).group()))
        sorted_columns = [col for col in landcover_type.columns if col not in frac_columns] + frac_columns
        landcover_type = landcover_type.reindex(columns=sorted_columns)

        for col in frac_columns:
            if landcover_type.loc[0, col] < 0.00001:
                landcover_type.loc[0, col] = 0.00001

        # Process elevation
        elevation_mean['mean'].fillna(0, inplace=True)

        # Save modified files
        soil_type.to_csv(path_general / 'modified_domain_stats_soil_classes.csv', index=False)
        landcover_type.to_csv(path_general / 'modified_domain_stats_NA_NALCMS_landcover_2020_30m.csv', index=False)
        elevation_mean.to_csv(path_general / 'modified_domain_stats_elv.csv', index=False)

        self.logger.info("Cleanup and checks completed")
        print('cleanup and checks completed')

    def write_summa_files(self):
        """Write SUMMA input files."""
        self.logger.info("Writing SUMMA input files")

        output_path = Path(self.config.root_path) / f"domain_{self.config.domain_name}" / "summa_setup"
        output_path.mkdir(parents=True, exist_ok=True)

        # Write attribute file
        attr = write_summa_attribute(output_path, self.config.basins_path, self.config.rivers_path, 
                                     Path(self.config.root_path) / f"domain_{self.config.domain_name}" / "gistool-outputs", 
                                     self.config.frac_threshold, self.config.hru_discr, self.config.write_mizuroute_domain)

        # Write forcing file
        forcing = write_summa_forcing(output_path, Path(self.config.root_path) / f"domain_{self.config.domain_name}" / "easymore-outputs", attr)

        # Write parameter trial file
        write_summa_paramtrial(attr, output_path)

        # Write initial conditions file
        write_summa_initial_conditions(attr, self.config.soil_mLayerDepth, output_path)

        # Write file manager
        write_summa_filemanager(output_path, forcing)

        # Copy static files
        copy_summa_static_files(output_path)

        self.logger.info("SUMMA input files written successfully")

    def prepare_modis_data(self) -> None:
        """Preprocess MODIS data and save it for later use in model evaluation."""
        self.logger.info("Preprocessing MODIS data")
        
     
        try:
            hru_shapefile = Path(self.config.root_path) / f"domain_{self.config.domain_name}" / "shapefiles" / "catchment" / f"{self.config.catchment_shp_name}"
            modis_dir = Path(self.config.root_path) / f"domain_{self.config.domain_name}" / "observations/snow/raw_data/"
            modis_files = list(modis_dir.glob("MODIS_Snow_Cover_*.tif"))
            output_file = Path(self.config.root_path) / f"domain_{self.config.domain_name}" /  f"observations/snow/preprocessed/{self.config.domain_name}_preprocessed_modis_data.csv"

            if not modis_files:
                self.logger.warning(f"No MODIS files found in {modis_dir}")
                return

            self.logger.info(f"Found {len(modis_files)} MODIS files")
            for file in modis_files:
                self.logger.info(f"  {file.name}")

            preprocess_modis_data(
                hru_shapefile,
                modis_files,
                output_file,
                self.config.modis_ndsi_threshold
            )
            
            self.logger.info(f"Preprocessed MODIS data saved to {output_file}")
        except Exception as e:
            self.logger.error(f"Error in prepare_modis_data: {str(e)}", exc_info=True)

    def prepare_forcing_and_parameters(self):
        """Prepare forcing and parameter data using MAF."""
        self.run_maf()

    def prepare_observation_data(self) -> None:
        """Prepare observation data for model evaluation."""
        self.logger.info("Preparing observation data")

        self.prepare_modis_data()

    def prepare_spatial_data(self) -> None:
        """Prepare spatial data (e.g., catchment boundaries, river network)."""
        self.logger.info("Preparing spatial data")

        self.config.pour_point_shapefile_path = create_pour_point_shapefile(self.config, self.logger)
        
        if self.config.domain_subsetting == 'delineate':
            self.config.gru_shapefile_path, self.config.subbasin_shapefile_path, self.config.river_network_shapefile_path = delineate_grus(self.config, self.logger)
        elif self.config.domain_subsetting == 'Merit':
            self.config.basins_path, self.config.rivers_path = subset_merit_hydrofabrics(self.config, self.logger)
        elif self.config.domain_subsetting == 'TDX':
            self.config.basins_path, self.config.rivers_path = subset_TDX_hydrofabrics(self.config, self.logger)
        elif self.config.domain_subsetting == 'NWS':
            self.config.basins_path, self.config.rivers_path = subset_NWS_hydrofabrics(self.config, self.logger)
        else:
            self.logger.warning(f"Unsupported domain_subsetting: {self.config.domain_subsetting}")

        #self.config.bb_coords = find_bounding_box_coordinates(self.config, self.logger)

        #generate_hrus(self.config, self.logger)
        #prepare_hru_parameters(self.config, self.logger)

    def run_preprocessing(self) -> Dict[str, Any]:
        """Run all preprocessing steps and return any necessary information."""
        self.logger.info("Starting preprocessing workflow")

        #self.config.domain_folder = make_folder_structure(self.config, self.logger)
        #self.prepare_spatial_data()
        #self.prepare_observation_data()
        
        self.prepare_forcing_and_parameters()
        # Add other preprocessing steps here as needed
        
        self.logger.info("Preprocessing completed")
        
        return {
            "domain_folder": str(Path(self.config.root_path) / f"domain_{self.config.domain_name}"),
            # Add other relevant information here
        }

def main():
    control_files = ["control_Copper.txt"] #["control_Makaha_Str_nr_Makaha__Oahu__HI.txt","control_Kamananui_Str_at_Maunawai__Oahu__HI.txt","control_Kaukonahua_Stream_blw_Wahiawa_Reservoir__Oahu__HI.txt","control_Honouliuli_Str_at_H-1_Freeway_nr_Waipahu__Oahu__HI.txt","control_Waikele_Str_at_Waipahu__Oahu__HI.txt","control_Opaeula_Str_nr_Wahiawa__Oahu__HI.txt","control_SF_Kaukonahua_Str_at_E_pump__nr_Wahiawa__Oahu__HI.txt","control_NF_Kaukonahua_Str_abv_RB__nr_Wahiawa__Oahu__HI.txt","control_N._Halawa_Str_nr_Quar._Stn._at_Halawa__Oahu__HI.txt","control_Kaluanui_Stream_nr_Punaluu__Oahu__HI.txt","control_N._Halawa_Str_nr_Honolulu__Oahu__HI.txt","control_Punaluu_Str_abv_Punaluu_Ditch_Intake__Oahu__HI.txt","control_Kahana_Str_at_alt_30_ft_nr_Kahana__Oahu__HI.txt","control_Waikane_Str_at_alt_75_ft_at_Waikane__Oahu__HI.txt","control_Waihee_Str_nr_Kahaluu__Oahu__HI.txt","control_Moanalua_Stream_nr_Kaneohe__Oahu__HI.txt","control_Waiahole_Stream_above_Kamehameha_Hwy__Oahu__HI.txt","control_Kalihi_Str_nr_Honolulu__Oahu__HI.txt","control_Kahaluu_Str_nr_Ahuimanu__Oahu__HI.txt","control_Makiki_Stream_at_King_St._bridge__Oahu__HI.txt","control_Heeia_Stream_at_Haiku_Valley_nr_Kaneohe__Oahu__HI.txt","control_Manoa-Palolo_Drainage_Canal_at_Moiliili__Oahu__HI.txt","control_Manoa_Stream_at_Woodlawn_Drive__Oahu__HI.txt","control_Waihi_Stream_at_Honolulu__Oahu__HI.txt","control_Waiakeakua_Str_at_Honolulu__Oahu__HI.txt","control_Kaneohe_Str_blw_Kamehameha_Hwy__Oahu__HI.txt","control_Kawa_Str_at_Kaneohe__Oahu__HI.txt","control_Pukele_Stream_near_Honolulu__Oahu__HI.txt","control_Makawao_Str_nr_Kailua__Oahu__HI.txt","control_Waimanalo_Str_at_Waimanalo__Oahu__HI.txt","control_Honouliuli_Stream_Tributary_near_Waipahu__Oahu__HI.txt","control_Kaunakakai_Gulch_at_altitude_75_feet__Molokai__HI.txt","control_Kawela_Gulch_near_Moku__Molokai__HI.txt","control_Halawa_Stream_near_Halawa__Molokai__HI.txt","control_Honokohau_Stream_near_Honokohau__Maui__HI.txt","control_Kahakuloa_Stream_near_Honokohau__Maui__HI.txt","control_Waihee_Rv_abv_Waihee_Dtch_intk_nr_Waihee__Maui__HI.txt","control_Wailuku_River_at_Kepaniwai_Park__Maui__HI.txt","control_Honopou_Stream_near_Huelo__Maui__HI.txt","control_Waikamoi_Str_abv_Kula_PL_intake_nr_Olinda__Maui_HI.txt","control_West_Wailuaiki_Stream_near_Keanae__Maui__HI.txt","control_Hanawi_Stream_near_Nahiku__Maui__HI.txt","control_Oheo_Gulch_at_dam_near_Kipahulu__Maui__HI.txt","control_Waimea_River_near_Waimea__Kauai__HI.txt","control_Kawaikoi_Stream_nr_Waimea__Kauai__HI.txt","control_Waialae_Str_at_alt_3_820_ft_nr_Waimea__Kauai__HI.txt","control_Wainiha_River_nr_Hanalei__Kauai__HI.txt","control_Hanalei_River_nr_Hanalei__Kauai__HI.txt","control_Halaulani_Str_at_alt_400_ft_nr_Kilauea__Kauai__HI.txt","control_EB_of_NF_Wailua_River_nr_Lihue__Kauai__HI.txt","control_Left_Branch_Opaekaa_Str_nr_Kapaa__Kauai__HI.txt","control_SF_Wailua_River_nr_Lihue__Kauai__HI.txt","control_Waiaha_Stream_at_Holualoa__HI.txt","control_Kawainui_Stream_nr_Kamuela__HI.txt","control_Alakahi_Stream_near_Kamuela__HI.txt","control_Paauau_Gulch_at_Pahala__HI.txt","control_Honolii_Stream_nr_Papaikou__HI.txt","control_Wailuku_River_at_Piihonua__HI.txt"]
    for site in control_files:
        control_file = Path(f"../0_control_files/{site}")
        config = preConfig.from_control_file(control_file)
        
        preprocessor = preProcessor(config)
        preprocessing_results = preprocessor.run_preprocessing()
        
        print(f"Preprocessing completed. Results: {preprocessing_results}")

if __name__ == "__main__":
    main()