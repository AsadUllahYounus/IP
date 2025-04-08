import zipfile
import os

# Name of the zip file
zip_file_name = "your_file.zip"  # Replace with your actual zip file name

# Get the current directory
current_dir = os.getcwd()

# Full path to the zip file
zip_path = os.path.join(current_dir, zip_file_name)

# Extract to current directory
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(current_dir)

print(f"Extracted '{zip_file_name}' to '{current_dir}'")
