import pandas as pd # type: ignore
import geopandas as gpd # type: ignore
import rasterio # type: ignore
from rasterio.mask import mask # type: ignore
from shapely.geometry import mapping # type: ignore
from tqdm import tqdm # type: ignore
from pathlib import Path
from typing import List
import re
from datetime import datetime

def preprocess_modis_data(hru_shapefile: Path, modis_files: List[Path], output_file: Path, ndsi_threshold: float):
    """
    Preprocess MODIS data and save it for later use in model evaluation.

    Args:
        hru_shapefile (Path): Path to the HRU shapefile.
        modis_files (List[Path]): List of paths to MODIS GeoTIFF files.
        output_file (Path): Path to save the preprocessed data.
        ndsi_threshold (float): NDSI threshold for snow cover determination.
    """
    hru_gdf = gpd.read_file(hru_shapefile)
    all_data = []

    for modis_file in tqdm(modis_files, desc="Processing MODIS files"):
        try:
            with rasterio.open(modis_file) as src:
                # Extract year from filename
                year_match = re.search(r'MODIS_Snow_Cover_(\d{4})\.tif', modis_file.name)
                if not year_match:
                    print(f"Could not extract year from filename: {modis_file.name}")
                    continue
                year = year_match.group(1)

                for band in range(1, src.count + 1):
                    # Extract date from band name
                    band_name = src.descriptions[band - 1]  # band indexing starts at 1, but descriptions indexing starts at 0
                    date_match = re.search(r'(\d{4}-\d{2}-\d{2})', band_name)
                    if not date_match:
                        print(f"Could not extract date from band name: {band_name}")
                        continue
                    date_str = date_match.group(1)
                    print(date_str)
                    try:
                        date = datetime.strptime(date_str, '%Y-%m-%d')
                    except ValueError:
                        print(f"Invalid date format in {modis_file.name}, band {band}: {date_str}")
                        continue

                    band_data = src.read(band)

                    for idx, hru in hru_gdf.iterrows():
                        geom = [mapping(hru.geometry)]
                        out_image, out_transform = mask(src, geom, crop=True)
                        out_image = out_image[0]  # Get the 2D array
                        
                        # Calculate zonal statistics
                        valid_pixels = out_image[out_image != src.nodata]
                        if len(valid_pixels) > 0:
                            mean_ndsi = valid_pixels.mean()
                            obs_sca = 1 if mean_ndsi > ndsi_threshold else 0
                            
                            all_data.append({
                                'date': date,
                                'hru_id': hru['HRU_ID'],
                                'obs_sca': obs_sca
                            })
        except rasterio.errors.RasterioIOError:
            print(f"Error reading file: {modis_file}")
            continue

    if not all_data:
        print("No data processed. Check your MODIS files and HRU shapefile.")
        return

    df = pd.DataFrame(all_data)
    df.to_csv(output_file, index=False)
    print(f"Preprocessed data saved to {output_file}")
    print(f"Data shape: {df.shape}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")