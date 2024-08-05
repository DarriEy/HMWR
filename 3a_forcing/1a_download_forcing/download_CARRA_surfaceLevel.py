import cdsapi
from pathlib import Path
import eccodes
import numpy as np
from netCDF4 import Dataset
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import os

# Control file handling
controlFolder = Path('../../0_control_files')
controlFile = 'control_Iceland.txt'

def read_from_control(file, setting):
    with open(file) as contents:
        for line in contents:
            if setting in line and not line.startswith('#'):
                substring = line.split('|', 1)[1].split('#', 1)[0].strip()
                return substring
    return None

def make_default_path(suffix):
    rootPath = Path(read_from_control(controlFolder/controlFile, 'root_path'))
    domainName = read_from_control(controlFolder/controlFile, 'domain_name')
    domainFolder = 'domain_' + domainName
    return rootPath / domainFolder / suffix


def download_carra_data(year, month, output_file):
    if output_file.exists():
        print(f"File already exists for {year}-{month}, skipping download.")
        return True

    c = cdsapi.Client()
    
    try:
        c.retrieve(
            'reanalysis-carra-single-levels',
            {
                'domain': 'west_domain',
                'level_type': 'surface_or_atmosphere',
                'variable': [
                    '10m_u_component_of_wind', '10m_v_component_of_wind', '2m_specific_humidity',
                    '2m_temperature', 'surface_net_solar_radiation', 'thermal_surface_radiation_downwards',
                    'surface_pressure', 'total_precipitation',
                ],
                'product_type': 'forecast',
                'time': [
                    '00:00', '03:00', '06:00',
                    '09:00', '12:00', '15:00',
                    '18:00', '21:00',
                ],
                'year': year,
                'month': month,
                'day': [f"{i:02d}" for i in range(1, 32)],
                'format': 'grib',
                'leadtime_hour': '1',
            },
            output_file)
        print(f"Download complete for {year}-{month}")
        return True
    except Exception as e:
        print(f"Download failed for {year}-{month}: {str(e)}")
        return False



def subset_iceland_data(input_file, output_file):
    if os.path.exists(output_file):
        print(f"Output file {output_file} already exists. Skipping processing.")
        return True

    try:
        timestamps = set()
        variables = set()
        
        with open(input_file, 'rb') as f_in:
            while True:
                msg = eccodes.codes_grib_new_from_file(f_in)
                if msg is None:
                    break

                date = eccodes.codes_get(msg, 'validityDate')
                time = eccodes.codes_get(msg, 'validityTime')
                timestamps.add(f"{date}{time:04d}")
                
                variables.add(eccodes.codes_get(msg, 'shortName'))

                eccodes.codes_release(msg)

        timestamps = sorted(list(timestamps))
        variables = sorted(list(variables))

        print(f"Number of timestamps: {len(timestamps)}")
        print(f"Variables: {', '.join(variables)}")

        with Dataset(output_file, 'w', format='NETCDF4') as nc:
            nc.createDimension('time', len(timestamps))
            nc.createDimension('point', None)

            times = nc.createVariable('time', 'f8', ('time',))
            lats = nc.createVariable('latitude', 'f4', ('point',))
            lons = nc.createVariable('longitude', 'f4', ('point',))

            times.units = 'hours since 1900-01-01 00:00:00'
            lats.units = 'degrees_north'
            lons.units = 'degrees_east'

            var_dict = {var: nc.createVariable(var, 'f4', ('time', 'point'), zlib=True, complevel=4) for var in variables}

            times[:] = [(datetime.strptime(ts, '%Y%m%d%H%M') - datetime(1900, 1, 1)).total_seconds() / 3600 for ts in timestamps]

            first_msg = True
            with open(input_file, 'rb') as f_in:
                while True:
                    msg = eccodes.codes_grib_new_from_file(f_in)
                    if msg is None:
                        break

                    lats_full = eccodes.codes_get_array(msg, 'latitudes')
                    lons_full = eccodes.codes_get_array(msg, 'longitudes')
                    values = eccodes.codes_get_values(msg)

                    if first_msg:
                        print(f"Full data shape: {lats_full.shape}")
                        print(f"Lat range: {lats_full.min()} to {lats_full.max()}")
                        print(f"Lon range: {lons_full.min()} to {lons_full.max()}")

                        # Convert longitudes to -180 to 180 range
                        lons_full = (lons_full + 180) % 360 - 180
                        
                        mask = (lats_full >= 62.5) & (lats_full <= 67.5) & \
                               (lons_full >= -25.5) & (lons_full <= -12.5)
                        
                        print(f"Mask sum: {mask.sum()}")
                        
                        if mask.sum() == 0:
                            print("No points found within specified range. Adjusting range...")
                            mask = (lats_full >= 60) & (lats_full <= 70) & \
                                   (lons_full >= -30) & (lons_full <= -10)
                            print(f"New mask sum: {mask.sum()}")

                        lats[:] = lats_full[mask]
                        lons[:] = lons_full[mask]
                        
                        print(f"Subsetted lat range: {lats[:].min()} to {lats[:].max()}")
                        print(f"Subsetted lon range: {lons[:].min()} to {lons[:].max()}")
                        
                        first_msg = False
                    else:
                        # Convert longitudes to -180 to 180 range for consistency
                        lons_full = (lons_full + 180) % 360 - 180

                    masked_values = values[mask]

                    varname = eccodes.codes_get(msg, 'shortName')
                    date = eccodes.codes_get(msg, 'validityDate')
                    time = eccodes.codes_get(msg, 'validityTime')
                    timestamp = f"{date}{time:04d}"

                    time_index = timestamps.index(timestamp)

                    var_dict[varname][time_index, :] = masked_values

                    eccodes.codes_release(msg)

            print(f"Final dimensions - time: {nc.dimensions['time'].size}, point: {nc.dimensions['point'].size}")
            for var in variables:
                print(f"Variable {var} shape: {nc.variables[var].shape}")

        print(f"Successfully subsetted data and saved as NetCDF")
        return True
    except Exception as e:
        print(f"Error subsetting data: {str(e)}")
        return False

def main():
    # Find which years to download
    years = read_from_control(controlFolder/controlFile,'forcing_raw_time')

    # Split the string into 2 integers
    years = years.split(',')
    years = [int(year) for year in years]
    years = range(years[0], years[1]+1)
    months = [f"{i:02d}" for i in range(1, 13)]
    carra_folder = read_from_control(controlFolder/controlFile, 'forcing_carra_folder')
    if carra_folder == 'default':
        carra_folder = make_default_path('forcing/1b_CARRA_raw_data')
    else:
        carra_folder = Path(carra_folder)
    carra_folder.mkdir(parents=True, exist_ok=True)

    for year in years:
        for month in months:
            raw_file = carra_folder / f'carra_raw_{year}_{month}.grib'
            subset_file = carra_folder / f'carra_iceland_{year}_{month}.nc'

            # Check if subsetted file already exists
            if subset_file.exists():
                print(f"Subsetted file already exists for {year}-{month}, skipping.")
                continue

            # Check if raw file exists
            if raw_file.exists():
                print(f"Raw file exists for {year}-{month}, skipping download.")
            else:
                # Download raw file if it doesn't exist
                if not download_carra_data(year, month, raw_file):
                    print(f"Failed to download data for {year}-{month}, skipping.")
                    continue

            # Proceed with subsetting
            if subset_iceland_data(raw_file, subset_file):
                print(f"Processed data for {year}-{month}")
                # Remove the original file after successful subsetting
                raw_file.unlink()
                print(f"Removed raw file for {year}-{month}")
            else:
                print(f"Failed to subset data for {year}-{month}")

if __name__ == "__main__":
    main()