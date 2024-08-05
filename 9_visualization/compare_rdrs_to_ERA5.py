import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from pathlib import Path
from scipy import stats
import pandas as pd
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER


def print_dataset_info(ds, name):
    print(f"\n{name} Dataset Information:")
    print(f"Dimensions: {ds.dims}")
    print("Variables:")
    for var in ds.data_vars:
        print(f"  - {var}: {ds[var].dims}, {ds[var].dtype}")

def get_variable_mapping():
    return {
        'LWRadAtm': 'LWRadAtm',
        'SWRadAtm': 'SWRadAtm',
        'pptrate': 'pptrate',
        'airpres': 'airpres',
        'airtemp': 'airtemp',
        'spechum': 'spechum',
        'windspd': 'windspd'
    }

def align_datasets(era5_ds, rdrs_ds, era5_var, rdrs_var):
    # Convert to pandas DataFrames
    era5_df = era5_ds[era5_var].mean(dim=['latitude', 'longitude']).to_dataframe()
    rdrs_df = rdrs_ds[rdrs_var].mean(dim=['rlat', 'rlon']).to_dataframe()

    # Print DataFrame information for debugging
    print(f"ERA5 DataFrame info:\n{era5_df.info()}\n")
    print(f"RDRS DataFrame info:\n{rdrs_df.info()}\n")

    # Ensure both datasets have the same index
    common_index = era5_df.index.intersection(rdrs_df.index)
    era5_aligned = era5_df.loc[common_index]
    rdrs_aligned = rdrs_df.loc[common_index]

    # Print aligned DataFrame information
    print(f"Aligned ERA5 DataFrame info:\n{era5_aligned.info()}\n")
    print(f"Aligned RDRS DataFrame info:\n{rdrs_aligned.info()}\n")

    return era5_aligned, rdrs_aligned

def compare_datasets(era5_file, rdrs_file, era5_var, rdrs_var, output_folder):
    era5 = xr.open_dataset(era5_file)
    rdrs = xr.open_dataset(rdrs_file)

    print_dataset_info(era5, "ERA5")
    print_dataset_info(rdrs, "RDRS")

    print(f"ERA5 time range: {era5.time.values[0]} to {era5.time.values[-1]}")
    print(f"RDRS time range: {rdrs.time.values[0]} to {rdrs.time.values[-1]}")

    if era5_var not in era5 or rdrs_var not in rdrs:
        print(f"Error: Variable {era5_var} not found in ERA5 or {rdrs_var} not found in RDRS")
        return

    era5_aligned, rdrs_aligned = align_datasets(era5, rdrs, era5_var, rdrs_var)

    if era5_aligned.empty or rdrs_aligned.empty:
        print(f"No overlapping data found for {era5_var} (ERA5) vs {rdrs_var} (RDRS)")
        return

    era5_mean = era5[era5_var].mean(dim='time')
    rdrs_mean = rdrs[rdrs_var].mean(dim='time')
    era5_daily = era5_aligned.resample('D').mean()
    rdrs_daily = rdrs_aligned.resample('D').mean()

    fig = plt.figure(figsize=(20, 16))
    
    def add_map(ax, data, title, cmap='viridis'):
        if 'longitude' in data.coords and 'latitude' in data.coords:
            lon, lat = data.longitude, data.latitude
        elif 'rlon' in data.coords and 'rlat' in data.coords:
            lon, lat = data.rlon, data.rlat
        else:
            raise ValueError("Unable to determine longitude and latitude coordinates")
        
        im = ax.pcolormesh(lon, lat, data, transform=ccrs.PlateCarree(), cmap=cmap)
        ax.set_title(title)
        ax.coastlines()
        gl = ax.gridlines(draw_labels=True, linestyle='--', alpha=0.5)
        gl.top_labels = gl.right_labels = False
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        gl.xlabel_style = {'size': 8}
        gl.ylabel_style = {'size': 8}
        return im

    ax1 = fig.add_subplot(221, projection=ccrs.PlateCarree())
    im1 = add_map(ax1, era5_mean, f'ERA5 {era5_var}')
    plt.colorbar(im1, ax=ax1, orientation='horizontal', pad=0.05)

    ax2 = fig.add_subplot(222, projection=ccrs.PlateCarree())
    im2 = add_map(ax2, rdrs_mean, f'RDRS {rdrs_var}')
    plt.colorbar(im2, ax=ax2, orientation='horizontal', pad=0.05)

    ax3 = fig.add_subplot(223)
    ax3.plot(era5_aligned.index, era5_aligned.iloc[:, 0], label='ERA5')
    ax3.plot(rdrs_aligned.index, rdrs_aligned.iloc[:, 0], label='RDRS')
    ax3.set_title(f'Time Series Comparison: {era5_var} (ERA5) vs {rdrs_var} (RDRS)')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Value')
    ax3.legend()

    ax4 = fig.add_subplot(224)
    ax4.scatter(era5_daily.iloc[:, 0], rdrs_daily.iloc[:, 0], alpha=0.5)
    ax4.set_xlabel(f'ERA5 {era5_var} (Daily Avg)')
    ax4.set_ylabel(f'RDRS {rdrs_var} (Daily Avg)')
    ax4.set_title('Scatter Plot (Daily Averages)')

    min_val = min(era5_daily.iloc[:, 0].min(), rdrs_daily.iloc[:, 0].min())
    max_val = max(era5_daily.iloc[:, 0].max(), rdrs_daily.iloc[:, 0].max())
    ax4.plot([min_val, max_val], [min_val, max_val], 'r--', label='1:1 line')

    slope, intercept, r_value, _, _ = stats.linregress(era5_daily.iloc[:, 0], rdrs_daily.iloc[:, 0])
    line = slope * np.array([min_val, max_val]) + intercept
    ax4.plot([min_val, max_val], line, 'g-', label=f'Regression line (R² = {r_value**2:.2f})')
    ax4.legend()

    plt.tight_layout()
    output_file = output_folder / f'{era5_var}_{rdrs_var}_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nStatistics for {era5_var} (ERA5) vs {rdrs_var} (RDRS):")
    print(f"ERA5 mean: {era5_aligned.iloc[:, 0].mean()}")
    print(f"RDRS mean: {rdrs_aligned.iloc[:, 0].mean()}")
    print(f"ERA5 std: {era5_aligned.iloc[:, 0].std()}")
    print(f"RDRS std: {rdrs_aligned.iloc[:, 0].std()}")
    print(f"Correlation coefficient: {era5_aligned.iloc[:, 0].corr(rdrs_aligned.iloc[:, 0])}")
    print(f"ERA5 units: {era5[era5_var].units if 'units' in era5[era5_var].attrs else 'Not specified'}")
    print(f"RDRS units: {rdrs[rdrs_var].units if 'units' in rdrs[rdrs_var].attrs else 'Not specified'}")

    era5.close()
    rdrs.close()

def main():
    # Set up paths
    era5_folder = Path("/Users/darrieythorsson/compHydro/data/CWARHM_data/domain_Goodpaster_Merit/forcing/2_merged_data")
    rdrs_folder = Path("/Users/darrieythorsson/compHydro/data/CWARHM_data/domain_Goodpaster_Merit/forcing/2b_rdrs_merged_data")
    output_folder = Path("/Users/darrieythorsson/compHydro/data/CWARHM_data/domain_Goodpaster_Merit/plots")

    output_folder.mkdir(parents=True, exist_ok=True)

    variable_mapping = get_variable_mapping()

    for year in range(2010, 2011):  # Adjust the range as needed
        for month in range(3, 4):
            era5_file = era5_folder / f"ERA5_merged_{year}{month:02d}.nc"
            rdrs_file = rdrs_folder / f"RDRS_monthly_{year}{month:02d}.nc"

            if not era5_file.exists() or not rdrs_file.exists():
                print(f"Files not found for {year}-{month:02d}. Skipping.")
                continue

            print(f"\nComparing datasets for {year}-{month:02d}")
            
            for era5_var, rdrs_var in variable_mapping.items():
                try:
                    compare_datasets(era5_file, rdrs_file, era5_var, rdrs_var, output_folder)
                except Exception as e:
                    print(f"Error processing {era5_var} (ERA5) vs {rdrs_var} (RDRS): {str(e)}")
                    import traceback
                    traceback.print_exc()
                    continue

    print("\nComparison complete.")

if __name__ == "__main__":
    main()