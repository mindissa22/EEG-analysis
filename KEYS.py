import os

# Path to the directory containing the .mat files
directory_path = '/Users/minolidissanayake/Desktop/Keele/Modules/SEM 3/Dataset/'

# List all files in the directory
files = os.listdir(directory_path)

# Print all .mat files
mat_files = [file for file in files if file.endswith('.mat')]
print("MAT files in the directory:", mat_files)

import scipy.io

# Path to the .mat file
file_name = 'v36p.mat'
file_path = f'/Users/minolidissanayake/Desktop/Keele/Modules/SEM 3/Dataset/{file_name}'

# Load the .mat file
mat_contents = scipy.io.loadmat(file_path)

# Print the keys of the loaded dictionary
print(f"Keys in the file {file_name}:", mat_contents.keys())

import os
import scipy.io

# Path to the directory containing the .mat files
directory_path = '/Users/minolidissanayake/Desktop/Keele/Modules/SEM 3/Dataset/'

# List all files in the directory
files = os.listdir(directory_path)

# Filter out .mat files
mat_files = [file for file in files if file.endswith('.mat')]

# Check the keys in each .mat file
for mat_file in mat_files:
    file_path = os.path.join(directory_path, mat_file)
    mat_contents = scipy.io.loadmat(file_path)
    print(f"Keys in {mat_file}:", mat_contents.keys())


import scipy.io
import numpy as np

# Path to a sample .mat file
sample_file = 'v36p.mat'
file_path = f'/Users/minolidissanayake/Desktop/Keele/Modules/SEM 3/Dataset/{sample_file}'

# Load the .mat file
mat_contents = scipy.io.loadmat(file_path)

# Extract the data from the key
key_name = 'v36p'  # Update this with the actual key for the file you're inspecting
data = mat_contents[key_name]

# Print information about the data
print(f"Type of data in '{sample_file}':", type(data))
print(f"Shape of data in '{sample_file}':", data.shape)
print(f"Data sample:\n", data)



