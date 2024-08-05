import os
import sys
import csv
import argparse
import re
import subprocess
from pathlib import Path
from collections import OrderedDict

def read_control_file(file_path):
    settings = OrderedDict()
    structure = []
    skip_rest = False
    max_key_length = 0

    with open(file_path, 'r') as f:
        for line in f:
            line = line.rstrip()
            if line.startswith("# Default folder structure"):
                skip_rest = True
                break
            if line.startswith('# '):
                structure.append(('header', line))
            elif line and not line.startswith('#'):
                parts = line.split('|')
                if len(parts) >= 2:
                    key = parts[0].strip()
                    value = parts[1].split('#')[0].strip()
                    comment = line.split('#', 1)[1].strip() if '#' in line else ''
                    settings[key] = value
                    structure.append(('setting', key, value, comment))
                    max_key_length = max(max_key_length, len(key))
            elif line == '':
                structure.append(('blank',))
            else:
                structure.append(('comment', line))

    return settings, structure, max_key_length

def get_latest_settings(control_folder):
    latest_settings = OrderedDict()
    latest_structure = None
    latest_file = None
    max_key_length = 0

    for file in sorted(os.listdir(control_folder)):
        if file.startswith('control_') and file.endswith('.txt'):
            file_path = os.path.join(control_folder, file)
            file_settings, file_structure, file_max_key_length = read_control_file(file_path)
            latest_settings.update(file_settings)
            latest_structure = file_structure
            latest_file = file_path
            max_key_length = max(max_key_length, file_max_key_length)

    return latest_settings, latest_structure, latest_file, max_key_length

def create_control_file(domain_name, settings, default_settings, structure, max_key_length):
    file_name = f'control_{domain_name}.txt'
    with open(file_name, 'w') as f:
        for item in structure:
            if item[0] == 'header':
                f.write(f"{item[1]}\n")
            elif item[0] == 'setting':
                key, _, comment = item[1], item[2], item[3]
                value = settings.get(key, default_settings.get(key, 'N/A'))
                f.write(f"{key.ljust(max_key_length)} | {value.ljust(20)}")
                if comment:
                    f.write(f" # {comment}")
                f.write("\n")
            elif item[0] == 'blank':
                f.write("\n")
            elif item[0] == 'comment':
                f.write(f"{item[1]}\n")
    print(f"Control file created: {file_name}")

def sanitize_key(key):
    # Remove leading/trailing whitespace and convert to lowercase
    key = key.strip().lower()
    # Replace spaces and hyphens with underscores
    key = re.sub(r'[ -]', '_', key)
    # Remove any other non-alphanumeric characters (except underscores)
    key = re.sub(r'[^\w]+', '', key)
    return key

def process_csv(csv_file, script_path, default_settings):
    created_files = []
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cmd = [sys.executable, script_path]
            domain_name = None
            for key, value in row.items():
                key = key.strip()
                value = value.strip()
                if value:
                    sanitized_key = sanitize_key(key)
                    matching_key = next((k for k in default_settings if sanitize_key(k) == sanitized_key), key)
                    cmd.extend([f'-{sanitized_key}', value])
                    if sanitized_key == 'domain_name':
                        domain_name = value
            print(f"Executing: {' '.join(cmd)}")
            try:
                subprocess.run(cmd, check=True)
                if domain_name:
                    created_files.append(f"control_{domain_name}.txt")
            except subprocess.CalledProcessError as e:
                print(f"Error processing row: {row}")
                print(f"Command failed: {e}")
    return created_files

def main():
    parser = argparse.ArgumentParser(description='Create control files for hydrological modeling.')
    parser.add_argument('--csv', help='Path to CSV file with domain settings')
    parser.add_argument('--control_folder', default='../0_control_files', help='Path to folder containing existing control files')
    parser.add_argument('-domain_name', help='Domain name')  # Explicitly add domain_name argument
    
    args, unknown = parser.parse_known_args()

    control_folder = Path(args.control_folder)
    default_settings, structure, latest_file, max_key_length = get_latest_settings(control_folder)

    # Create dynamic arguments based on default_settings
    for key in default_settings.keys():
        if key and key.lower() != 'domain_name':  # Skip domain_name as it's already added
            sanitized_key = sanitize_key(key)
            if not any(action.dest == key for action in parser._actions):  # Check if the argument already exists
                parser.add_argument(f'-{sanitized_key}', dest=key, help=f'Value for {key}')

    args = parser.parse_args()

    if args.csv:
        created_files = process_csv(args.csv, sys.argv[0], default_settings)
        print("\nCreated control files:")
        formatted_files = [f'"{file}"' for file in created_files]
        print(",".join(formatted_files))
    else:
        settings = {key: value for key, value in vars(args).items() if value is not None and (key in default_settings or key == 'domain_name')}

        if not settings:
            print("Error: No settings provided. Use -[setting_name] for manual input, or --csv for CSV input.")
            sys.exit(1)

        domain_name = settings.get('domain_name')
        if not domain_name:
            print("Error: domain_name is required.")
            sys.exit(1)

        create_control_file(domain_name, settings, default_settings, structure, max_key_length)

if __name__ == "__main__":
    main()