import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path
import geopandas as gpd

# Define paths and files here
era5_file = '/Users/darrieythorsson/compHydro/data/CWARHM_data/domain_Iceland/forcing/4_SUMMA_input/Iceland_remapped_2005-05-01-00-00-00.nc'
compare_file = '/Users/darrieythorsson/compHydro/data/CWARHM_data/domain_Iceland/forcing/4_SUMMA_input/Iceland_carra_remapped_2005-05-01-01-00-00.nc'
output_dir = Path('/Users/darrieythorsson/compHydro/data/CWARHM_data/domain_Iceland/plots/SUMMA_input_comparison/')
hru_shapefile = '/Users/darrieythorsson/compHydro/data/CWARHM_data/domain_Iceland/shapefiles/catchment/Iceland_TDX_Basins.shp'
compare_dataset = 'carra'  # or 'rdrs'

# Create output directory
output_dir.mkdir(exist_ok=True)

# Variables to compare
variables = ['airtemp', 'pptrate', 'SWRadAtm', 'LWRadAtm', 'airpres', 'spechum', 'windspd']

def compute_stats(data):
    return {
        'mean': data.mean().item(),
        'std': data.std().item(),
        'min': data.min().item(),
        'max': data.max().item(),
        'median': data.median().item()
    }

def create_comparison_plots(era5_file, compare_file, variable, hru_shapefile):
    era5_data = xr.open_dataset(era5_file)
    compare_data = xr.open_dataset(compare_file)
    
    print(f"ERA5 {variable} shape: {era5_data[variable].shape}")
    print(f"ERA5 longitude shape: {era5_data.longitude.shape}")
    print(f"ERA5 latitude shape: {era5_data.latitude.shape}")
    print(f"CARRA {variable} shape: {compare_data[variable].shape}")
    print(f"CARRA longitude shape: {compare_data.longitude.shape}")
    print(f"CARRA latitude shape: {compare_data.latitude.shape}")
    
    # Find common HRUs
    era5_hrus = set(era5_data.hru.values)
    carra_hrus = set(compare_data.hru.values)
    common_hrus = list(era5_hrus.intersection(carra_hrus))
    
    print(f"Number of common HRUs: {len(common_hrus)}")
    print(f"Missing HRUs in CARRA: {era5_hrus - carra_hrus}")
    
    # Select only common HRUs
    era5_data = era5_data.sel(hru=common_hrus)
    compare_data = compare_data.sel(hru=common_hrus)
    
    # Load HRU shapefile
    hru_gdf = gpd.read_file(hru_shapefile)
    hru_gdf = hru_gdf[hru_gdf['fid'].isin(common_hrus)]
    
    print(f"Number of HRUs in shapefile: {len(hru_gdf)}")
    
    # Align time periods and resample to a common frequency (e.g., daily)
    start_time = max(era5_data.time.min(), compare_data.time.min())
    end_time = min(era5_data.time.max(), compare_data.time.max())
    
    era5_data = era5_data.sel(time=slice(start_time, end_time)).resample(time='D').mean()
    compare_data = compare_data.sel(time=slice(start_time, end_time)).resample(time='D').mean()
    
    # Create figure
    fig = plt.figure(figsize=(20, 20))
    gs = fig.add_gridspec(3, 3)
    
    # Time series and scatter plots
    ax1 = fig.add_subplot(gs[0, :2])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])
    ax4 = fig.add_subplot(gs[1, 2])
    
    # Geographical maps
    ax5 = fig.add_subplot(gs[0, 2], projection=ccrs.PlateCarree())
    ax6 = fig.add_subplot(gs[2, :], projection=ccrs.PlateCarree())
    
    fig.suptitle(f'Comparison of {variable}', fontsize=16)
    
    # Time series plot
    era5_avg = era5_data[variable].mean(dim='hru')
    compare_avg = compare_data[variable].mean(dim='hru')
    
    ax1.plot(era5_avg.time, era5_avg, label='ERA5')
    ax1.plot(compare_avg.time, compare_avg, label='CARRA')
    ax1.set_title('Time Series of Spatial Averages (Daily)')
    ax1.set_xlabel('Time')
    ax1.set_ylabel(variable)
    ax1.legend()
    
    # Scatter plot
    df = pd.DataFrame({
        'ERA5': era5_avg.values,
        'CARRA': compare_avg.values
    }, index=era5_avg.time.values)
    
    ax2.scatter(df['ERA5'], df['CARRA'], alpha=0.5)
    ax2.set_xlabel('ERA5')
    ax2.set_ylabel('CARRA')
    ax2.set_title('Scatter Plot of Spatial Averages (Daily)')
    
    # Add 1:1 line
    min_val = min(df['ERA5'].min(), df['CARRA'].min())
    max_val = max(df['ERA5'].max(), df['CARRA'].max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', label='1:1 Line')
    
    # Compute and plot regression line
    slope, intercept, r_value, _, _ = stats.linregress(df['ERA5'], df['CARRA'])
    x = np.array([min_val, max_val])
    ax2.plot(x, slope * x + intercept, 'g-', label=f'R² = {r_value**2:.3f}')
    ax2.legend()
    
    # Compute statistics
    era5_stats = compute_stats(era5_avg)
    compare_stats = compute_stats(compare_avg)
    
    # Add statistics
    stats_text = (f"ERA5 stats:\n"
                  f"Mean: {era5_stats['mean']:.2f}\n"
                  f"Std: {era5_stats['std']:.2f}\n"
                  f"Min: {era5_stats['min']:.2f}\n"
                  f"Max: {era5_stats['max']:.2f}\n"
                  f"Median: {era5_stats['median']:.2f}\n\n"
                  f"CARRA stats:\n"
                  f"Mean: {compare_stats['mean']:.2f}\n"
                  f"Std: {compare_stats['std']:.2f}\n"
                  f"Min: {compare_stats['min']:.2f}\n"
                  f"Max: {compare_stats['max']:.2f}\n"
                  f"Median: {compare_stats['median']:.2f}")
    ax3.text(0.05, 0.95, stats_text, transform=ax3.transAxes, 
             verticalalignment='top', fontsize=10)
    ax3.axis('off')
    
    # Additional statistics
    additional_stats = (f"Comparison Statistics:\n"
                        f"Correlation: {df['ERA5'].corr(df['CARRA']):.4f}\n"
                        f"MAE: {np.mean(np.abs(df['ERA5'] - df['CARRA'])):.4f}\n"
                        f"RMSE: {np.sqrt(np.mean((df['ERA5'] - df['CARRA'])**2)):.4f}\n"
                        f"Slope: {slope:.4f}\n"
                        f"Intercept: {intercept:.4f}\n"
                        f"R-squared: {r_value**2:.4f}")
    ax4.text(0.05, 0.95, additional_stats, transform=ax4.transAxes, 
             verticalalignment='top', fontsize=10)
    ax4.axis('off')
    
    # Map plot (using ERA5 data for example)
    era5_mean = era5_data[variable].mean(dim='time')
    
    # Create a dictionary mapping HRU IDs to mean values
    hru_mean_dict = dict(zip(era5_data.hru.values, era5_mean.values))
    
    # Assign mean values to the GeoDataFrame
    hru_gdf['era5_mean'] = hru_gdf['fid'].map(hru_mean_dict)
    
    hru_gdf.plot(column='era5_mean', ax=ax5, legend=True, 
                 legend_kwds={'label': f'{variable} ({era5_data[variable].units})', 'orientation': 'horizontal'},
                 cmap='viridis')
    ax5.set_title('ERA5 Mean')
    ax5.add_feature(cfeature.COASTLINE)
    ax5.add_feature(cfeature.BORDERS)
    
    # Plot missing HRUs
    missing_hrus = list(era5_hrus - carra_hrus)
    if missing_hrus:
        missing_gdf = gpd.read_file(hru_shapefile)
        missing_gdf = missing_gdf[missing_gdf['fid'].isin(missing_hrus)]
        missing_gdf.plot(ax=ax6, color='red', label='Missing HRUs')
    
    hru_gdf.plot(ax=ax6, color='blue', alpha=0.5, label='Common HRUs')
    ax6.set_title('HRU Distribution')
    ax6.add_feature(cfeature.COASTLINE)
    ax6.add_feature(cfeature.BORDERS)
    ax6.legend()
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_dir / f'{variable}_comparison.png')
    plt.close()


variable = 'airtemp'

create_comparison_plots(era5_file, compare_file, variable, hru_shapefile)

print(f"Comparison plots have been saved in {output_dir}")