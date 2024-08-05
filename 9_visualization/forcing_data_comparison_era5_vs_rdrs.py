import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection


# Paths for ERA5 and RDRS data
era5_path = Path('/Users/darrieythorsson/compHydro/data/CWARHM_data/domain_Goodpaster_Merit/forcing/4_SUMMA_input')
rdrs_path = Path('/Users/darrieythorsson/compHydro/data/CWARHM_data/domain_Goodpaster_Merit/forcing/4b_SUMMA_input_rdrs')

# Output folder for plots
output_folder = Path('comparison_plots')
output_folder.mkdir(exist_ok=True)

# Variables to compare
variables = ['airtemp', 'airpres', 'pptrate', 'SWRadAtm', 'LWRadAtm', 'spechum', 'windspd']


def plot_comparison(era5_data, rdrs_data, variable, output_folder):
    fig = plt.figure(figsize=(20, 15))
    
    # Time series plot
    ax1 = fig.add_subplot(2, 2, 1)
    era5_mean = era5_data[variable].mean(dim='hru')
    rdrs_mean = rdrs_data[variable].mean(dim='hru')
    
    ax1.plot(era5_data.time, era5_mean, label='ERA5')
    ax1.plot(rdrs_data.time, rdrs_mean, label='RDRS')
    ax1.set_title(f'{variable} Time Series')
    ax1.set_xlabel('Time')
    ax1.set_ylabel(era5_data[variable].units)
    ax1.legend()
    
    # Scatter plot
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.scatter(era5_mean, rdrs_mean, alpha=0.5)
    min_val = min(era5_mean.min(), rdrs_mean.min())
    max_val = max(era5_mean.max(), rdrs_mean.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--')
    ax2.set_title(f'{variable} Scatter Plot')
    ax2.set_xlabel('ERA5')
    ax2.set_ylabel('RDRS')
    
    # Function to create HRU polygons
    def create_hru_polygons(lons, lats):
        polygons = []
        for lon, lat in zip(lons, lats):
            # Create a small square around each point
            # Adjust the size as needed
            size = 0.01  # in degrees
            poly = Polygon([
                (lon-size, lat-size),
                (lon+size, lat-size),
                (lon+size, lat+size),
                (lon-size, lat+size)
            ])
            polygons.append(poly)
        return polygons

    # ERA5 Map plot
    ax3 = fig.add_subplot(2, 2, 3, projection=ccrs.PlateCarree())
    era5_map = era5_data[variable].mean(dim='time')
    polygons = create_hru_polygons(era5_data.longitude, era5_data.latitude)
    p = PatchCollection(polygons, cmap='viridis', transform=ccrs.PlateCarree())
    p.set_array(era5_map)
    ax3.add_collection(p)
    ax3.set_title(f'ERA5 {variable} Map')
    ax3.add_feature(cfeature.COASTLINE)
    ax3.add_feature(cfeature.BORDERS)
    ax3.set_extent([era5_data.longitude.min()-0.1, era5_data.longitude.max()+0.1, 
                    era5_data.latitude.min()-0.1, era5_data.latitude.max()+0.1])
    plt.colorbar(p, ax=ax3, orientation='horizontal', pad=0.05)
    
    # RDRS Map plot
    ax4 = fig.add_subplot(2, 2, 4, projection=ccrs.PlateCarree())
    rdrs_map = rdrs_data[variable].mean(dim='time')
    polygons = create_hru_polygons(rdrs_data.longitude, rdrs_data.latitude)
    p = PatchCollection(polygons, cmap='viridis', transform=ccrs.PlateCarree())
    p.set_array(rdrs_map)
    ax4.add_collection(p)
    ax4.set_title(f'RDRS {variable} Map')
    ax4.add_feature(cfeature.COASTLINE)
    ax4.add_feature(cfeature.BORDERS)
    ax4.set_extent([rdrs_data.longitude.min()-0.1, rdrs_data.longitude.max()+0.1, 
                    rdrs_data.latitude.min()-0.1, rdrs_data.latitude.max()+0.1])
    plt.colorbar(p, ax=ax4, orientation='horizontal', pad=0.05)
    
    plt.tight_layout()
    plt.savefig(output_folder / f'{variable}_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def compare_datasets():
    year, month = 2010, 5
    era5_pattern = f'*{year}-{month:02d}-01-00-00-00.nc'
    rdrs_pattern = f'*{year}-{month:02d}-01-00-00-00.nc'
    
    era5_files = list(era5_path.glob(era5_pattern))
    rdrs_files = list(rdrs_path.glob(rdrs_pattern))
    
    if not era5_files or not rdrs_files:
        print(f"Files not found for {year}-{month:02d}")
        return
    
    era5_file = era5_files[0]
    rdrs_file = rdrs_files[0]
    
    print(f"Processing {era5_file.name} and {rdrs_file.name}")
    
    era5_data = xr.open_dataset(era5_file)
    rdrs_data = xr.open_dataset(rdrs_file)
    
    print("ERA5 data structure:")
    print(era5_data)
    print("\nRDRS data structure:")
    print(rdrs_data)

    # Ensure time periods match
    start_time = max(era5_data.time.min(), rdrs_data.time.min())
    end_time = min(era5_data.time.max(), rdrs_data.time.max())
    
    era5_data = era5_data.sel(time=slice(start_time, end_time))
    rdrs_data = rdrs_data.sel(time=slice(start_time, end_time))
    
    # Check for different precipitation rate variable names
    ppt_vars = ['pptrate', 'RDRS_v2.1_A_PR0_SFC', 'RDRS_v2.1_P_PR0_SFC']
    for var in ppt_vars:
        if var in rdrs_data:
            rdrs_data['pptrate'] = rdrs_data[var]
            break
    
    for variable in variables:
        if variable in era5_data and variable in rdrs_data:
            plot_comparison(era5_data, rdrs_data, variable, output_folder)
        else:
            print(f"Variable {variable} not found in both datasets")
    
    era5_data.close()
    rdrs_data.close()

if __name__ == "__main__":
    compare_datasets()
    print("Comparison plots created.")