# Data loading
from pathlib import Path
import scipy.io as sio

import numpy as np
import pandas as pd

data_dir = Path.home() / Path("/Users/minolidissanayake/Desktop/Keele/Modules/SEM 3/Dataset" )
# Channel labels and electrode positions
channel_labels = {
    0: "Fp1",
    1: "Fp2",
    2: "F3",
    3: "F4",
    4: "C3",
    5: "C4",
    6: "P3",
    7: "P4",
    8: "O1",
    9: "O2",
    10: "F7",
    11: "F8",
    12: "T7",
    13: "T8",
    14: "P7",
    15: "P8",
    16: "Fz",
    17: "Cz",
    18: "Pz",
}

electrode_positions = {
    "Fp1": (-18, 0.511, 0.95, 0.309, -0.0349, 18, -2, 1),
    "Fp2": (18, 0.511, 0.95, -0.309, -0.0349, -18, -2, 1),
    "F7": (-54, 0.511, 0.587, 0.809, -0.0349, 54, -2, 1),
    "F3": (-39, 0.333, 0.673, 0.545, 0.5, 39, 30, 1),
    "Fz": (0, 0.256, 0.719, 0, 0.695, 0, 44, 1),
    "F4": (39, 0.333, 0.673, -0.545, 0.5, -39, 30, 1),
    "F8": (54, 0.511, 0.587, -0.809, -0.0349, -54, -2, 1),
    "T7": (-90, 0.511, 6.12e-17, 0.999, -0.0349, 90, -2, 1),
    "C3": (-90, 0.256, 4.4e-17, 0.719, 0.695, 90, 44, 1),
    "Cz": (90, 0, 3.75e-33, -6.12e-17, 1, -90, 90, 1),
    "C4": (90, 0.256, 4.4e-17, -0.719, 0.695, -90, 44, 1),
    "T8": (90, 0.511, 6.12e-17, -0.999, -0.0349, -90, -2, 1),
    "P7": (-126, 0.511, -0.587, 0.809, -0.0349, 126, -2, 1),
    "P3": (-141, 0.333, -0.673, 0.545, 0.5, 141, 30, 1),
    "Pz": (180, 0.256, -0.719, -8.81e-17, 0.695, -180, 44, 1),
    "P4": (141, 0.333, -0.673, -0.545, 0.5, -141, 30, 1),
    "P8": (126, 0.511, -0.587, -0.809, -0.0349, -126, -2, 1),
    "O1": (-162, 0.511, -0.95, 0.309, -0.0349, 162, -2, 1),
    "O2": (162, 0.511, -0.95, -0.309, -0.0349, -162, -2, 1),
}

# Sampling Frequency Hz
Sampling_Frequency = 128

# Set the chunk size
chunk_size = 512

def split_into_chunks(df, chunk_size, initial_chunk_number=0):
    # Calculate the number of full chunks
    n_chunks = len(df) // chunk_size
    chunks = []

    # Split into chunks and keep track of the chunk number
    for i in range(n_chunks):
        chunk = df.iloc[i * chunk_size : (i + 1) * chunk_size].copy()  # Get the chunk
        chunk["chunk_number"] = (
            initial_chunk_number + i
        )  # Add the chunk number as a new column
        chunks.append(chunk)

    # Concatenate the chunks back together
    return pd.concat(chunks, ignore_index=True), n_chunks
def load_data(data_dirs):
    data_list = []
    chunked_data_list = []

    chunk_index = 0
    for directory in data_dirs:
        # print(f"Loading data from {directory}")

        for filepath in directory.glob("*.mat"):
            mat = sio.loadmat(filepath)
            key = list(mat.keys())[-1]  # Get the last key (the id of the patient)
            eeg_data = mat[key]

            # Convert the EEG data to a DataFrame
            # Assuming the EEG data is a 2D array (time x channels)
            df = pd.DataFrame(eeg_data)
            df = df.rename(columns=channel_labels)
            # Add a column to identify the source
            df["subject_id"] = key

            # print(f"Loaded data for patient {key}; chunks start at {chunk_index}")
            chucked_df, chunks = split_into_chunks(df, chunk_size, chunk_index)
            chunk_index += chunks

            # Append the DataFrame to the list
            data_list.append(df)
            chunked_data_list.append(chucked_df)

        # Concatenate all DataFrames in the list into a single DataFrame
        full_eeg_df = pd.concat(data_list, ignore_index=True)
        chunked_eeg_df = pd.concat(chunked_data_list, ignore_index=True)

    return full_eeg_df, chunked_eeg_df
adhd_dir1 = data_dir / Path("ADHD_part1")
adhd_dir2 = data_dir / Path("ADHD_part2")
adhd_df, adhd_chunks_df = load_data([adhd_dir1, adhd_dir2])

control_dir1 = data_dir / Path("Control_part1")
control_dir2 = data_dir / Path("Control_part2")
control_df, control_chunks_df = load_data([control_dir1, control_dir2])

adhd_subjects = adhd_df["subject_id"].unique()
control_subjects = control_df["subject_id"].unique()
intersection = [item for item in adhd_subjects if item in control_subjects]


num_patients = len(adhd_subjects) + len(control_subjects)
num_data_points = adhd_df.shape[0] + control_df.shape[0]
num_chunks = 0
num_chunks2 = 0

print(f"Number of ADHD subjects: {len(adhd_subjects)}")
print(f"Number of Control subjects: {len(control_subjects)}")
print(f"Number of subjects in both groups: {len(intersection)}")

# print(adhd_df.info())
# print(control_df.info())

# print(adhd_df.describe())
# print(control_df.describe())

chunck_size = 512

num_chunks = 0
for patient in adhd_subjects:
    data_points = adhd_df[adhd_df["subject_id"] == patient].shape[0]
    num_chunks += data_points // chunck_size

print(f"ADHD - total number of data points: {adhd_df.shape[0]:,}")
print(f"ADHD - total number of chunks of size {chunck_size}: {num_chunks}")

num_chunks = 0
for patient in control_subjects:
    data_points = control_df[control_df["subject_id"] == patient].shape[0]
    num_chunks += data_points // chunck_size

print(f"Control - total number of data points: {control_df.shape[0]:,}")
print(f"Control - total number of chunks of size {chunck_size}: {num_chunks}")

# adhd_df[adhd_df['subject_id']==adhd_subjects[0]].drop(columns=['subject_id']).plot(subplots=True, figsize=(10, 10), title='ADHD for subject ' + adhd_subjects[0])

# Let's put all the data together (we will use the "chunked" data)
adhd_chunks_df["label"] = "ADHD"
control_chunks_df["label"] = "Control"

# we need to "renumber" the chunks for the control data so that they are continuous with the ADHD data
control_chunks_df["chunk_number"] += adhd_chunks_df["chunk_number"].max() + 1

all_data_df = pd.concat([adhd_chunks_df, control_chunks_df], ignore_index=True)

print(all_data_df["chunk_number"].nunique())
