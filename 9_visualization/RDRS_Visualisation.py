import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Load the processed file
file_path = '/Users/darrieythorsson/compHydro/data/CWARHM_data/domain_Goodpaster_Merit/forcing/1b_RDRS_raw_data/RDRS_merged_1980122112.nc'  # Replace with your actual file path
ds = xr.open_dataset(file_path)

# Create a figure with subplots for each variable
variables = ['airtemp', 'airpres', 'pptrate', 'windspd', 'LWRadAtm', 'SWRadAtm', 'spechum']
fig, axs = plt.subplots(3, 3, figsize=(20, 20), subplot_kw={'projection': ccrs.PlateCarree()})
axs = axs.flatten()

for i, var in enumerate(variables):
    if i < len(axs):
        ax = axs[i]
        
        # Plot the data
        im = ds[var].isel(time=0).plot(ax=ax, transform=ccrs.PlateCarree(), add_colorbar=False)
        
        # Add colorbar
        plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05)
        
        # Add coastlines and borders
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS)
        
        # Set title
        ax.set_title(var)
        
        # Add gridlines
        ax.gridlines(draw_labels=True)

# Remove any unused subplots
for j in range(i+1, len(axs)):
    fig.delaxes(axs[j])

plt.tight_layout()
#plt.savefig('rdrs_visualization.png', dpi=300, bbox_inches='tight')
#plt.close()
plt.show()

# Print some basic statistics
for var in variables:
    print(f"\nStatistics for {var}:")
    print(ds[var].isel(time=0).describe())

# Close the dataset
ds.close()

print("\nVisualization saved as 'rdrs_visualization.png'")