import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import datetime
from pathlib import Path
import seaborn as sns
import geopandas as gpd
from scipy import stats
import calendar
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.control_file_utils import read_from_control, make_default_path

# Easy access to control file folder
controlFolder = Path('../0_control_files')
controlFile = 'control_active.txt'

def get_plot_folder():
    plot_folder = read_from_control(controlFolder/controlFile, 'plot_folder')
    if plot_folder == 'default':
        root_path = Path(read_from_control(controlFolder/controlFile, 'root_path'))
        domain_name = read_from_control(controlFolder/controlFile, 'domain_name')
        return root_path / f"domain_{domain_name}" / "plots/output_diagnostics"
    else:
        return Path(plot_folder)

def load_mizuroute_data(file_path):
    ds = xr.open_dataset(file_path)
    df = ds['IRFroutedRunoff'].to_dataframe().reset_index()
    
    # Create a mapping between segment index and COMID
    seg_to_comid = dict(zip(range(len(ds.reachID)), ds.reachID.values))
    
    df['COMID'] = df['seg'].map(seg_to_comid)
    df = df.pivot(index='time', columns='COMID', values='IRFroutedRunoff')
    return df, seg_to_comid

def load_summa_data(file_path):
    ds = xr.open_dataset(file_path)
    
    variables = ['pptrate', 'scalarSWE', 'scalarTotalET', 'scalarInfiltration', 'scalarAquiferStorage', 'scalarTotalSoilWat', 'scalarSurfaceRunoff']
    df = ds[variables].to_dataframe().reset_index()
    
    # Determine the correct column name for HRU identifier
    hru_id_column = 'hru'  # This is a common name, but we'll check if it exists
    if 'hru' not in df.columns:
        possible_hru_columns = [col for col in df.columns if 'hru' in col.lower() or 'id' in col.lower()]
        if possible_hru_columns:
            hru_id_column = possible_hru_columns[0]
        else:
            raise ValueError("Could not find a column that represents HRU identifier. Please specify the correct column name.")
        
    df = df.pivot(index='time', columns=hru_id_column, values=variables)
    return df

def load_observation_data(file_path):
    df = pd.read_csv(file_path, parse_dates=['datetime'], index_col='datetime')
    return df['discharge_cms']

# Time series plotting functions
def plot_streamflow_comparison(obs, sim1, sim2, title, plot_folder):
    # Find overlapping time period
    start_date = max(obs.index.min(), sim1.index.min(), sim2.index.min())
    end_date = min(obs.index.max(), sim1.index.max(), sim2.index.max())
    
    # Filter data for overlapping period
    obs_filtered = obs.loc[start_date:end_date]
    sim1_filtered = sim1.loc[start_date:end_date]
    sim2_filtered = sim2.loc[start_date:end_date]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    obs_filtered.plot(ax=ax, label='Observed', color='blue')
    sim1_filtered.plot(ax=ax, label='Simulated 1', color='red')
    sim2_filtered.plot(ax=ax, label='Simulated 2', color='green')
    ax.set_title(title)
    ax.set_xlabel('Date')
    ax.set_ylabel('Streamflow (m³/s)')
    ax.legend()
    plt.tight_layout()
    
    plot_path = plot_folder / f"{title.replace(' ', '_')}.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved plot to {plot_path}")

def plot_summa_variables(data1, data2, variables, catchment_areas, title, plot_folder):
    total_area = catchment_areas.sum()
    
    fig, axes = plt.subplots(len(variables), 1, figsize=(12, 4*len(variables)), sharex=True)
    if len(variables) == 1:
        axes = [axes]
    
    for i, var in enumerate(variables):
        # Calculate area-weighted average for both simulations
        weighted_var1 = (data1[var] * catchment_areas).sum(axis=1) / total_area
        weighted_var2 = (data2[var] * catchment_areas).sum(axis=1) / total_area
        
        weighted_var1.plot(ax=axes[i], label='Simulation 1')
        weighted_var2.plot(ax=axes[i], label='Simulation 2')
        axes[i].set_title(f"Domain-averaged {var}")
        axes[i].set_ylabel('Value')
        axes[i].legend()
    
    axes[-1].set_xlabel('Date')
    fig.suptitle(title)
    plt.tight_layout()
    
    plot_path = plot_folder / f"{title.replace(' ', '_')}.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved plot to {plot_path}")

def plot_monthly_water_balance(summa_data, mizu_data, start_date, end_date, plot_folder):
    # Resample data to monthly
    monthly_data = summa_data.resample('ME').mean()
    monthly_mizu = mizu_data.resample('ME').mean()

    # Calculate monthly totals (converting from mm/hr to mm/month)
    days_in_month = pd.Series([calendar.monthrange(d.year, d.month)[1] for d in monthly_data.index], index=monthly_data.index)
    hours_in_month = days_in_month * 24

    # Aggregate across all HRUs/grid cells
    precip = (monthly_data['pptrate'].mean(axis=1) * hours_in_month).values
    et = (monthly_data['scalarTotalET'].mean(axis=1) * hours_in_month).values  # Negative for outgoing
    runoff = -(monthly_mizu.mean(axis=1) * 3600 * hours_in_month).values  # Negative for outgoing

    # Calculate storage change
    storage_change = precip + et + runoff  # Note: et and runoff are already negative

    # Create stacked bar plot
    fig, ax = plt.subplots(figsize=(15, 8))
    
    width = 20  # Adjust this value to change bar width

    ax.bar(monthly_data.index, precip, width=width, label='Precipitation', color='blue')
    ax.bar(monthly_data.index, et, width=width, label='Evapotranspiration', color='red')
    ax.bar(monthly_data.index, runoff, bottom=et, width=width, label='Runoff', color='green')
    ax.bar(monthly_data.index, storage_change, bottom=et+runoff, width=width, label='Storage Change', color='purple')

    # Set y-axis limits
    y_min = min(0, np.min([et, runoff, et+runoff, et+runoff+storage_change]))
    y_max = max(0, np.max([precip, et+runoff+storage_change]))
    ax.set_ylim(y_min * 1.1, y_max * 1.1)  # Add 10% padding

    ax.set_xlabel('Date')
    ax.set_ylabel('Water Balance Components (mm/month)')
    ax.set_title('Monthly Water Balance')
    ax.legend()

    # Adjust x-axis to show all months
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)

    # Add horizontal line at y=0
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)

    plt.tight_layout()
    plot_path = plot_folder / "Monthly_Water_Balance.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved plot to {plot_path}")

def plot_flow_duration_curve(obs, sim1, sim2, title, plot_folder):
    # Find overlapping time period
    start_date = max(obs.index.min(), sim1.index.min(), sim2.index.min())
    end_date = min(obs.index.max(), sim1.index.max(), sim2.index.max())
    
    # Filter data for overlapping period
    obs_filtered = obs.loc[start_date:end_date]
    sim1_filtered = sim1.loc[start_date:end_date]
    sim2_filtered = sim2.loc[start_date:end_date]

    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Sort values in descending order
    obs_sorted = obs_filtered.sort_values(ascending=False)
    sim1_sorted = sim1_filtered.sort_values(ascending=False)
    sim2_sorted = sim2_filtered.sort_values(ascending=False)
    
    # Calculate exceedance probabilities
    obs_exc = np.arange(1, len(obs_sorted) + 1) / (len(obs_sorted) + 1)
    sim1_exc = np.arange(1, len(sim1_sorted) + 1) / (len(sim1_sorted) + 1)
    sim2_exc = np.arange(1, len(sim2_sorted) + 1) / (len(sim2_sorted) + 1)
    
    ax.plot(obs_exc, obs_sorted, label='Observed', color='blue')
    ax.plot(sim1_exc, sim1_sorted, label='Simulated 1', color='red')
    ax.plot(sim2_exc, sim2_sorted, label='Simulated 2', color='green')
    
    ax.set_xscale('log')
    ax.set_xlabel('Exceedance Probability')
    ax.set_ylabel('Streamflow (m³/s)')
    ax.set_title(title)
    ax.legend()
    plt.grid(True)
    plt.tight_layout()
    
    plot_path = plot_folder / f"{title.replace(' ', '_')}.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved plot to {plot_path}")

# Statistical analysis functions
def calculate_flow_statistics(data):
    return {
        'mean': data.mean(),
        'median': data.median(),
        'std': data.std(),
        'Q5': data.quantile(0.05),
        'Q95': data.quantile(0.95)
    }


def calculate_water_balance(summa_data, mizu_data, catchment_areas):
    total_area = catchment_areas.sum()
    seconds_per_year = 365.25 * 24 * 3600
    simulation_hours = len(summa_data)
    
    # Weight the precipitation and ET by HRU area
    weighted_precip = (summa_data['pptrate'] * catchment_areas).sum(axis=1) / total_area
    weighted_et = (summa_data['scalarTotalET'] * catchment_areas).sum(axis=1) / total_area
    
    # Convert rates to totals
    total_precip = (weighted_precip * simulation_hours).sum()  # Now in mm
    total_et = (weighted_et * simulation_hours).sum()  # Now in mm
    
    # Use the outlet streamflow for runoff (last column of mizu_data)
    outlet_runoff = mizu_data.iloc[:, -1]
    total_runoff = (outlet_runoff * seconds_per_year / simulation_hours).sum() / (total_area) * 1000  # Convert m³/s to mm over the catchment area
    
    # Calculate storage change
    storage_change = (
        (summa_data['scalarAquiferStorage'].iloc[-1] - summa_data['scalarAquiferStorage'].iloc[0]).mean() +
        (summa_data['scalarTotalSoilWat'].iloc[-1] - summa_data['scalarTotalSoilWat'].iloc[0]).mean() +
        (summa_data['scalarSWE'].iloc[-1] - summa_data['scalarSWE'].iloc[0]).mean()
    )  # Already in mm

    return {
        'total_precip': total_precip / 1000,  # Convert mm to m
        'total_et': abs(total_et) / 1000,  # Convert mm to m, ensure it's positive
        'total_runoff': total_runoff / 1000,  # Convert mm to m
        'storage_change': storage_change / 1000,  # Convert mm to m
        'mean_outlet_runoff': outlet_runoff.mean(),
        'max_outlet_runoff': outlet_runoff.max(),
        'min_outlet_runoff': outlet_runoff.min(),
        'total_area': total_area,
        'simulation_hours': simulation_hours
    }

def create_spatial_map(data1, data2, variable, title, shapefile_path, plot_folder):
    # Read the shapefile
    gdf = gpd.read_file(shapefile_path)
    
    # For mizuRoute data, we don't need to select a specific variable
    if variable == 'IRFroutedRunoff':
        data_agg1 = data1.mean()
        data_agg2 = data2.mean()
    else:
        data_agg1 = data1[variable].mean()
        data_agg2 = data2[variable].mean()
    
    # Ensure COMID is integer in the shapefile
    gdf['COMID'] = gdf['COMID'].astype(int)
    
    # Create DataFrames with COMID as a column, not index
    data_df1 = pd.DataFrame({'COMID': data_agg1.index, f'{variable}_1': data_agg1.values})
    data_df2 = pd.DataFrame({'COMID': data_agg2.index, f'{variable}_2': data_agg2.values})
    
    # Merge the aggregated data with the spatial data
    gdf = gdf.merge(data_df1, on='COMID', how='left')
    gdf = gdf.merge(data_df2, on='COMID', how='left')
    
    # Check if the merge was successful
    if gdf[f'{variable}_1'].isnull().all() and gdf[f'{variable}_2'].isnull().all():
        print(f"Warning: No data for {variable} after merging. Check COMID compatibility.")
        return
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Inside create_spatial_map function, before creating the plot
    vmin = min(gdf[f'{variable}_1'].min(), gdf[f'{variable}_2'].min())
    vmax = max(gdf[f'{variable}_1'].max(), gdf[f'{variable}_2'].max())

    # Then, when plotting:
    gdf.plot(column=f'{variable}_1', cmap='viridis', legend=False, ax=ax1, vmin=vmin, vmax=vmax)
    gdf.plot(column=f'{variable}_2', cmap='viridis', legend=False, ax=ax2, vmin=vmin, vmax=vmax)

    # Plot for Simulation 1
    ax1.set_title(f"{title} - Simulation 1")
    ax1.axis('off')
    
    # Plot for Simulation 2
    ax2.set_title(f"{title} - Simulation 2")
    ax2.axis('off')
    
    plt.tight_layout()
    
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=[ax1, ax2], orientation='horizontal', fraction=0.05, pad=0.1)
    cbar.set_label(variable)

    # Save the plot in the specified folder
    plot_path = plot_folder / f"{title.replace(' ', '_')}.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved plot to {plot_path}")

# Main function
def main():
    # Read necessary information from control file
    experiment_id = read_from_control(controlFolder/controlFile, 'experiment_id')
    domain_name = read_from_control(controlFolder/controlFile, 'domain_name')
    root_path = Path(read_from_control(controlFolder/controlFile, 'root_path'))
    
    mizuroute_file = read_from_control(controlFolder/controlFile, 'mizuroute_output_file')
    summa_file = read_from_control(controlFolder/controlFile, 'summa_output_file')
    summa_file_2 = read_from_control(controlFolder/controlFile, 'summa_output_file_2')
    mizuroute_file_2 = read_from_control(controlFolder/controlFile, 'mizuroute_output_file_2')

    obs_file = read_from_control(controlFolder/controlFile, 'obs_file_path')
    segment_id = int(read_from_control(controlFolder/controlFile, 'sim_reach_ID'))
    
    catchment_path = make_default_path(controlFolder, controlFile, 'shapefiles/catchment')
    catchment_name = read_from_control(controlFolder/controlFile, 'catchment_shp_name')
    catchment_file = str(catchment_path/catchment_name)
    
    river_network_path = make_default_path(controlFolder, controlFile, 'shapefiles/river_network')
    river_network_name = read_from_control(controlFolder/controlFile, 'river_network_shp_name')
    river_network_file = str(river_network_path/river_network_name)

   # Get plot folder
    plot_folder = get_plot_folder()
    plot_folder.mkdir(parents=True, exist_ok=True)

    # Read file paths for the second simulation
    try:
        # Load data for both simulations
        mizu_data_1, seg_to_comid_1 = load_mizuroute_data(mizuroute_file)
        summa_data_1 = load_summa_data(summa_file)
        mizu_data_2, seg_to_comid_2 = load_mizuroute_data(mizuroute_file_2)
        summa_data_2 = load_summa_data(summa_file_2)
        obs_data = load_observation_data(obs_file)

        # Load catchment areas
        catchment_shp = gpd.read_file(catchment_file)
        catchment_areas = catchment_shp.set_index('HRU_ID')['HRU_area']  # Now in m²

        # Generate plots
        plot_streamflow_comparison(obs_data, mizu_data_1[segment_id], mizu_data_2[segment_id], f"Streamflow Comparison - Segment {segment_id}", plot_folder)
        plot_summa_variables(summa_data_1, summa_data_2, ['pptrate', 'scalarSWE', 'scalarTotalET', 'scalarInfiltration', 'scalarAquiferStorage', 'scalarTotalSoilWat', 'scalarSurfaceRunoff'], catchment_areas, "SUMMA Variables - Domain Average", plot_folder)
        plot_flow_duration_curve(obs_data, mizu_data_1[segment_id], mizu_data_2[segment_id], f"Flow Duration Curve - Segment {segment_id}", plot_folder)

        # Plot monthly water balance
        start_date = summa_data_1.index[0].strftime('%Y-%m-%d')
        end_date = summa_data_1.index[-1].strftime('%Y-%m-%d')
        plot_monthly_water_balance(summa_data_1, mizu_data_1, start_date, end_date, plot_folder)



        # Calculate and print statistics for both simulations
        flow_stats_1 = calculate_flow_statistics(mizu_data_1[segment_id])
        flow_stats_2 = calculate_flow_statistics(mizu_data_2[segment_id])
        water_balance_1 = calculate_water_balance(summa_data_1, mizu_data_1, catchment_areas)
        water_balance_2 = calculate_water_balance(summa_data_2, mizu_data_2, catchment_areas)
        
        print("Flow Statistics (Simulation 1):", flow_stats_1)
        print("Flow Statistics (Simulation 2):", flow_stats_2)
        for i, (water_balance, mizu_data, summa_data) in enumerate([(water_balance_1, mizu_data_1, summa_data_1), (water_balance_2, mizu_data_2, summa_data_2)], 1):
            print(f"\nWater Balance (Simulation {i}) (in meters):")
            for component, value in water_balance.items():
                if component in ['total_precip', 'total_et', 'total_runoff', 'storage_change']:
                    print(f"  {component}: {value:.4f} m")
                elif component in ['mean_outlet_runoff', 'max_outlet_runoff', 'min_outlet_runoff']:
                    print(f"  {component}: {value:.4f} m³/s")
                else:
                    print(f"  {component}: {value}")
            
            print(f"  Balance check: {water_balance['total_precip'] - water_balance['total_et'] - water_balance['total_runoff'] - water_balance['storage_change']:.4f} m")
            
            # Print some statistics about the mizuRoute output
            print(f"  mizuRoute output statistics:")
            print(f"    Mean streamflow: {mizu_data.mean().mean():.4f} m³/s")
            print(f"    Max streamflow: {mizu_data.max().max():.4f} m³/s")
            print(f"    Min streamflow: {mizu_data.min().min():.4f} m³/s")
            
            # Print some statistics about the SUMMA output
            print(f"  SUMMA output statistics:")
            print(f"    Mean precipitation rate: {summa_data['pptrate'].mean().mean():.6f} mm/hr")
            print(f"    Total precipitation: {(summa_data['pptrate'] * water_balance['simulation_hours']).mean().sum():.2f} mm")
            print(f"    Mean ET rate: {abs(summa_data['scalarTotalET'].mean().mean()):.6f} mm/hr")
            print(f"    Total ET: {abs(summa_data['scalarTotalET'] * water_balance['simulation_hours']).mean().sum():.2f} mm")
            print(f"    Final - Initial SWE: {(summa_data['scalarSWE'].iloc[-1] - summa_data['scalarSWE'].iloc[0]).mean():.4f} mm")
            print(f"    Final - Initial soil water: {(summa_data['scalarTotalSoilWat'].iloc[-1] - summa_data['scalarTotalSoilWat'].iloc[0]).mean():.4f} mm")
            print(f"    Final - Initial aquifer storage: {(summa_data['scalarAquiferStorage'].iloc[-1] - summa_data['scalarAquiferStorage'].iloc[0]).mean():.4f} mm")

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()