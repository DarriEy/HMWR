import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os
import pandas as pd
from scipy import stats
import matplotlib.animation as animation
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec

def load_dataset(file_path):
    return xr.open_dataset(file_path)

def plot_comparison(carra_ds, era5_ds, variable, output_dir):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    
    # Time series plot
    carra_data = carra_ds[variable].mean(dim='point')
    era5_data = era5_ds[variable].mean(dim=['latitude', 'longitude'])
    
    ax1.plot(carra_data.time, carra_data, label='CARRA')
    ax1.plot(era5_data.time, era5_data, label='ERA5')
    
    ax1.set_title(f'Time Series Comparison: {variable}')
    ax1.set_xlabel('Time')
    ax1.set_ylabel(carra_ds[variable].units)
    ax1.legend()
    
    # Scatter plot
    df = pd.DataFrame({
        'CARRA': carra_data.to_series(),
        'ERA5': era5_data.to_series()
    })
    df = df.dropna()
    
    ax2.scatter(df['CARRA'], df['ERA5'], alpha=0.5)
    ax2.set_title(f'Scatter Plot: {variable}')
    ax2.set_xlabel('CARRA')
    ax2.set_ylabel('ERA5')
    
    min_val = min(df['CARRA'].min(), df['ERA5'].min())
    max_val = max(df['CARRA'].max(), df['ERA5'].max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', label='1:1 Line')
    ax2.legend()
    
    # Map plots
    carra_mean = carra_ds[variable].mean(dim='time')
    era5_mean = era5_ds[variable].mean(dim='time')
    
    vmin = min(carra_mean.min(), era5_mean.min())
    vmax = max(carra_mean.max(), era5_mean.max())
    
    # Set common x and y limits
    lon_min, lon_max = min(carra_ds.longitude.min(), era5_ds.longitude.min()), max(carra_ds.longitude.max(), era5_ds.longitude.max())
    lat_min, lat_max = min(carra_ds.latitude.min(), era5_ds.latitude.min()), max(carra_ds.latitude.max(), era5_ds.latitude.max())
    
    carra_plot = ax3.scatter(carra_ds.longitude, carra_ds.latitude, c=carra_mean, cmap='viridis', 
                             vmin=vmin, vmax=vmax, s=10)
    plt.colorbar(carra_plot, ax=ax3, label=f'{variable} ({carra_ds[variable].units})')
    ax3.set_title(f'CARRA Map: {variable}')
    ax3.set_xlabel('Longitude')
    ax3.set_ylabel('Latitude')
    ax3.set_xlim(lon_min, lon_max)
    ax3.set_ylim(lat_min, lat_max)
    
    era5_plot = ax4.pcolormesh(era5_ds.longitude, era5_ds.latitude, era5_mean, cmap='viridis', 
                               vmin=vmin, vmax=vmax)
    plt.colorbar(era5_plot, ax=ax4, label=f'{variable} ({era5_ds[variable].units})')
    ax4.set_title(f'ERA5 Map: {variable}')
    ax4.set_xlabel('Longitude')
    ax4.set_ylabel('Latitude')
    ax4.set_xlim(lon_min, lon_max)
    ax4.set_ylim(lat_min, lat_max)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{variable}_comparison.png'))
    plt.close()

    # Create GIF
    create_map_gif(carra_ds, era5_ds, variable, output_dir)

    # Calculate and save diagnostic statistics
    stats_text = f"Diagnostic Statistics for {variable}\n"
    stats_text += "==================================\n\n"
    
    stats_text += "CARRA Statistics:\n"
    stats_text += f"Mean: {carra_data.mean().values:.4f}\n"
    stats_text += f"Median: {carra_data.median().values:.4f}\n"
    stats_text += f"Standard Deviation: {carra_data.std().values:.4f}\n"
    stats_text += f"Min: {carra_data.min().values:.4f}\n"
    stats_text += f"Max: {carra_data.max().values:.4f}\n\n"
    
    stats_text += "ERA5 Statistics:\n"
    stats_text += f"Mean: {era5_data.mean().values:.4f}\n"
    stats_text += f"Median: {era5_data.median().values:.4f}\n"
    stats_text += f"Standard Deviation: {era5_data.std().values:.4f}\n"
    stats_text += f"Min: {era5_data.min().values:.4f}\n"
    stats_text += f"Max: {era5_data.max().values:.4f}\n\n"
    
    stats_text += "Comparison Statistics:\n"
    stats_text += f"Correlation Coefficient: {df['CARRA'].corr(df['ERA5']):.4f}\n"
    stats_text += f"Mean Absolute Error: {np.mean(np.abs(df['CARRA'] - df['ERA5'])):.4f}\n"
    stats_text += f"Root Mean Square Error: {np.sqrt(np.mean((df['CARRA'] - df['ERA5'])**2)):.4f}\n"
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(df['CARRA'], df['ERA5'])
    stats_text += f"Linear Regression - Slope: {slope:.4f}, Intercept: {intercept:.4f}, R-squared: {r_value**2:.4f}\n"

    # Print statistics to console
    print(stats_text)

    # Save statistics to file
    stats_file = os.path.join(output_dir, f'{variable}_statistics.txt')
    with open(stats_file, 'w') as f:
        f.write(stats_text)

def create_map_gif(carra_ds, era5_ds, variable, output_dir):
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 0.05], width_ratios=[1, 1])
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    cax = fig.add_subplot(gs[1, :])

    carra_data = carra_ds[variable]
    era5_data = era5_ds[variable].sel(time=carra_ds.time)  # Align ERA5 time with CARRA

    vmin = min(carra_data.min(), era5_data.min())
    vmax = max(carra_data.max(), era5_data.max())

    lon_min, lon_max = min(carra_ds.longitude.min(), era5_ds.longitude.min()), max(carra_ds.longitude.max(), era5_ds.longitude.max())
    lat_min, lat_max = min(carra_ds.latitude.min(), era5_ds.latitude.min()), max(carra_ds.latitude.max(), era5_ds.latitude.max())

    # Create a ScalarMappable for the colorbar
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])

    # Add the colorbar
    cbar = fig.colorbar(sm, cax=cax, orientation='horizontal')
    cbar.set_label(f'{variable} ({carra_ds[variable].units})')

    def update(frame):
        ax1.clear()
        ax2.clear()

        carra_plot = ax1.scatter(carra_ds.longitude, carra_ds.latitude, c=carra_data[frame], 
                                 cmap='viridis', vmin=vmin, vmax=vmax, s=10)
        ax1.set_title(f'CARRA: {variable}')
        ax1.set_xlim(lon_min, lon_max)
        ax1.set_ylim(lat_min, lat_max)
        ax1.set_xlabel('Longitude')
        ax1.set_ylabel('Latitude')

        era5_plot = ax2.pcolormesh(era5_ds.longitude, era5_ds.latitude, era5_data[frame], 
                                   cmap='viridis', vmin=vmin, vmax=vmax)
        ax2.set_title(f'ERA5: {variable}')
        ax2.set_xlim(lon_min, lon_max)
        ax2.set_ylim(lat_min, lat_max)
        ax2.set_xlabel('Longitude')
        ax2.set_ylabel('Latitude')

        plt.suptitle(f'{variable} Comparison: CARRA vs ERA5\nTimestamp: {carra_ds.time[frame].values}', fontsize=16, y=0.98)

        return carra_plot, era5_plot

    anim = animation.FuncAnimation(fig, update, frames=len(carra_ds.time), interval=200, blit=False)
    
    plt.subplots_adjust(top=0.9, bottom=0.1, left=0.05, right=0.95, hspace=0.2, wspace=0.1)
    anim.save(os.path.join(output_dir, f'{variable}_comparison.gif'), writer='pillow', fps=5)
    plt.close()

def visualize_data(carra_file, era5_file, output_dir):
    carra_ds = load_dataset(carra_file)
    era5_ds = load_dataset(era5_file)
    
    start_time = max(carra_ds.time.min(), era5_ds.time.min())
    end_time = min(carra_ds.time.max(), era5_ds.time.max())
    
    carra_ds = carra_ds.sel(time=slice(start_time, end_time))
    era5_ds = era5_ds.sel(time=slice(start_time, end_time))
    
    variables = ['airtemp', 'airpres', 'spechum', 'windspd', 'SWRadAtm', 'LWRadAtm', 'pptrate']
    
    for var in variables:
        if var in carra_ds and var in era5_ds:
            plot_comparison(carra_ds, era5_ds, var, output_dir)
        else:
            print(f"Variable {var} not found in both datasets. Skipping.")
    
    print(f"Visualizations, statistics, and GIFs saved in {output_dir}")

# Usage
carra_file = '/Users/darrieythorsson/compHydro/data/CWARHM_data/domain_Iceland/forcing/2_merged_data/CARRA_processed_carra_iceland_2005_01.nc'
era5_file = '/Users/darrieythorsson/compHydro/data/CWARHM_data/domain_Iceland/forcing/2_merged_data/ERA5_merged_200501.nc'
output_dir = '/Users/darrieythorsson/compHydro/data/CWARHM_data/domain_Iceland/plots'

os.makedirs(output_dir, exist_ok=True)

visualize_data(carra_file, era5_file, output_dir)