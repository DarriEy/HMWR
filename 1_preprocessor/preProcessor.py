from pathlib import Path
from typing import Dict, Any
import sys
import subprocess
import json
import numpy as np # type: ignore
import pandas as pd # type: ignore
import re
import cdo # type: ignore

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.logging_utils import get_logger # type: ignore
from utils.config import preConfig # type: ignore
from utils.geospatial_processing import (create_pour_point_shapefile, delineate_grus, subset_merit_hydrofabrics, subset_TDX_hydrofabrics, subset_NWS_hydrofabrics,find_bounding_box_coordinates,generate_hrus,prepare_hru_parameters) # type: ignore
from utils.file_utils import make_folder_structure # type: ignore
from utils.summaflow import write_summa_attribute, write_summa_forcing, write_summa_paramtrial, write_summa_initial_conditions, write_summa_filemanager, copy_summa_static_files # type: ignore
from utils.observation_preprocessing import preprocess_modis_data # type: ignore
from utils.hypeflow import write_hype_geo_files, write_hype_info_filedir_files, write_hype_par_file, write_hype_forcing # type: ignore
from utils.input_data_processing import prepare_maf_json, cleanup_and_checks  # type: ignore 

class preProcessor:
    def __init__(self, config: 'Config'): # type: ignore
        self.config = config
        self.logger = get_logger('PreProcessor', self.config.root_path, self.config.domain_name, 'preprocessing')

    def run_maf(self):
        ''' Run the Model Agnostic Framework. '''
        json_path = prepare_maf_json(self.config)
        maf_script = Path(self.config.root_path) / "installs/MAF/02_model_agnostic_component/model-agnostic.sh"
        
        #Run the MAF script
        try:
            subprocess.run([str(maf_script), str(json_path)], check=True)
            self.logger.info("Model Agnostic Framework completed successfully.")
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error running Model Agnostic Framework: {e}")
            raise

        cleanup_and_checks(self.config, self.logger)

    def write_summa_files(self):
        """Write SUMMA input files."""
        self.logger.info("Writing SUMMA input files")

        output_path = f'{self.config.root_path}/domain_{self.config.domain_name}/settings/summa_setup/'
        Path(output_path).mkdir(parents=True, exist_ok=True)

        catchment_path = f'{self.config.root_path}/domain_{self.config.domain_name}/shapefiles/catchment/{self.config.catchment_shp_name}'
        river_network_path = f'{self.config.root_path}/domain_{self.config.domain_name}/shapefiles/river_network/{self.config.river_network_shp_name}'
        parameters_path = Path(self.config.root_path) / f"domain_{self.config.domain_name}" / "parameters"

        # Write attribute file
        attr = write_summa_attribute(output_path, catchment_path, river_network_path, parameters_path,
                                     self.config.frac_threshold, self.config.hru_discr, self.config.write_mizuroute_domain)

        # Write forcing file
        forcing = write_summa_forcing(output_path, f'{self.config.root_path}/domain_{self.config.domain_name}/forcing/3_basin_averaged_data', attr)

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

    def prepare_hype_forcing(self):
        """Prepare forcing data for HYPE."""
        self.logger.info("Preparing HYPE forcing data")
        
        easymore_output = Path(self.config.root_path) / f"domain_{self.config.domain_name}/forcing/3_basin_averaged_data"
        output_path = Path(self.config.root_path) / f"domain_{self.config.domain_name}/settings/hype_setup"
        
        write_hype_forcing(str(easymore_output), str(output_path))
        
        self.logger.info("HYPE forcing data prepared successfully")

    def write_hype_geo_files(self):
        """Write GeoClass and GeoData files for HYPE."""
        self.logger.info("Writing HYPE GeoClass and GeoData files")
        
        gistool_output = Path(self.config.root_path) / f"domain_{self.config.domain_name}/parameters"
        output_path = Path(self.config.root_path) / f"domain_{self.config.domain_name}/settings/hype_setup"
        
        write_hype_geo_files(
            str(gistool_output),
            str(self.config.basins_path),
            str(self.config.rivers_path),
            self.config.frac_threshold,
            str(output_path)
        )
        
        self.logger.info("HYPE GeoClass and GeoData files written successfully")

    def write_hype_par_file(self):
        """Write parameter file for HYPE."""
        self.logger.info("Writing HYPE parameter file")
        
        output_path = Path(self.config.root_path) / f"domain_{self.config.domain_name}/settings/hype_setup"
        
        write_hype_par_file(str(output_path))
        
        self.logger.info("HYPE parameter file written successfully")

    def write_hype_info_filedir_files(self):
        """Write info.txt and Filedir.txt files for HYPE."""
        self.logger.info("Writing HYPE info and filedir files")
        
        output_path = Path(self.config.root_path) / f"domain_{self.config.domain_name}/settings/hype_setup"
        
        write_hype_info_filedir_files(str(output_path), self.config.spinup_days)
        
        self.logger.info("HYPE info and filedir files written successfully")

    def prepare_hype_data(self):
        """Prepare all data for HYPE model."""
        self.logger.info("Preparing HYPE data")
        
        self.prepare_hype_forcing()
        self.write_hype_geo_files()
        self.write_hype_par_file()
        self.write_hype_info_filedir_files()

        self.logger.info("HYPE data preparation completed")


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

        self.config.bb_coords = find_bounding_box_coordinates(self.config, self.logger)

        generate_hrus(self.config, self.logger)
        prepare_hru_parameters(self.config, self.logger)

    def run_preprocessing(self) -> Dict[str, Any]:
        """Run all preprocessing steps and return any necessary information."""
        self.logger.info("Starting preprocessing workflow")

        self.config.domain_folder = make_folder_structure(self.config, self.logger)
        self.prepare_spatial_data()
        self.prepare_observation_data()
        self.prepare_forcing_and_parameters()
        
        if self.config.model == 'SUMMA':
            self.write_summa_files()
        elif self.config.model == 'MESH':
            self.prepare_mesh_data()
        elif self.config.model == 'HYPE':
            self.prepare_hype_data()
        else:
            self.logger.error(f"Unsupported model type: {self.config.model_type}")
            raise ValueError(f"Unsupported model type: {self.config.model_type}")
        
        # Add other preprocessing steps here as needed
        
        self.logger.info("Preprocessing completed")
        
        return {"domain_folder": str(Path(self.config.root_path) / f"domain_{self.config.domain_name}")}

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