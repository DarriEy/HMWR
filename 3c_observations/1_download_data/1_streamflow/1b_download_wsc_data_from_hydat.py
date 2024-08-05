import os
import requests
import lzma
import shutil
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from pathlib import Path

# Easy access to control file folder
controlFolder = Path('../../../0_control_files')

# Store the name of the 'active' file in a variable
controlFile = 'control_active.txt'

# Function to extract a given setting from the control file
def read_from_control( file, setting ):
    
    # Open 'control_active.txt' and ...
    with open(file) as contents:
        for line in contents:
            
            # ... find the line with the requested setting
            if setting in line and not line.startswith('#'):
                break
    
    # Extract the setting's value
    substring = line.split('|',1)[1]      # Remove the setting's name (split into 2 based on '|', keep only 2nd part)
    substring = substring.split('#',1)[0] # Remove comments, does nothing if no '#' is found
    substring = substring.strip()         # Remove leading and trailing whitespace, tabs, newlines
       
    # Return this value    
    return substring
    
# Function to specify a default path
def make_default_path(suffix):
    
    # Get the root path
    rootPath = Path( read_from_control(controlFolder/controlFile,'root_path') )
    
    # Get the domain folder
    domainName = read_from_control(controlFolder/controlFile,'domain_name')
    domainFolder = 'domain_' + domainName
    
    # Specify the forcing path
    defaultPath = rootPath / domainFolder / suffix
    
    return defaultPath

# Base URL of the data source
base_url = "https://collaboration.cmc.ec.gc.ca/cmc/hydrometrics/www/UnitValueData/Discharge/corrected/05/"

# Local base directory for the Yukon domain
base_dir = read_from_control(controlFolder/controlFile,'root_path') + '/domain_' + read_from_control(controlFolder/controlFile,'domain_name') + "/observations/Streamflow/WSC2022_Stations/"

# Function to download and extract file
def download_and_extract(url, station_code):
    local_dir = os.path.join(base_dir, f"CAN_{station_code}")
    os.makedirs(local_dir, exist_ok=True)
    
    local_file = os.path.join(local_dir, os.path.basename(url))
    extracted_file = local_file[:-3]  # Remove the .xz extension
    
    print(f"Downloading file to: {local_file}")
    
    # Download file
    response = requests.get(url)
    with open(local_file, 'wb') as f:
        f.write(response.content)
    
    print(f"Extracting file to: {extracted_file}")
    
    # Extract file
    with lzma.open(local_file, 'rb') as f_in:
        with open(extracted_file, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    
    # Remove the compressed file
    os.remove(local_file)
    print(f"Removed compressed file: {local_file}")
    print(f"Extracted file saved at: {extracted_file}")

# Get the list of files from the web page
response = requests.get(base_url)
soup = BeautifulSoup(response.text, 'html.parser')

for link in soup.find_all('a'):
    href = link.get('href')
    if href.startswith("Discharge.Working@05") and href.endswith(".csv.xz"):
        station_code = href.split('@')[1][:7]
        file_url = urljoin(base_url, href)
        print(f"\nProcessing: {file_url}")
        download_and_extract(file_url, station_code)

print("\nDownload and extraction complete.")