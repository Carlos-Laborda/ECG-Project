import mne
import numpy as np
import os
import pandas as pd
import h5py
from datetime import datetime
from config import CATEGORY_MAPPING, FOLDERPATH, OUTPUT_DIR_PATH

# Set the folder path containing the raw ECG files and the metadata
filepaths = [
    os.path.join(FOLDERPATH, f)
    for f in os.listdir(FOLDERPATH)
    if f.endswith("_ECG.edf")
]
files = [f for f in os.listdir(FOLDERPATH) if f.endswith("_ECG.edf")]
participants = [f[:5] for f in files]  # Extract participant IDs from filenames

# Output directory for processed segments
output_dir_path = OUTPUT_DIR_PATH

# Read the metadata file
timestamps = pd.read_csv(
    os.path.join(FOLDERPATH, "TimeStamps_Merged.txt"), sep="\t", decimal="."
)
timestamps["LabelStart"] = pd.to_datetime(
    timestamps["LabelStart"], format="%Y-%m-%d %H:%M:%S", utc=True
)
timestamps["LabelEnd"] = pd.to_datetime(
    timestamps["LabelEnd"], format="%Y-%m-%d %H:%M:%S", utc=True
)
timestamps["Subject_ID"] = timestamps["Subject_ID"].astype(str)

# Sampling frequency of the ECG data
fs = 1000

# Create HDF5 file
hdf5_filename = os.path.join(output_dir_path, "ecg_data.h5")
with h5py.File(hdf5_filename, "w") as f:
    for p, participant_id in enumerate(
        participants[:10]
    ):  # only 2 for testing purposes
        print(f"Processing Participant {participant_id} ({p + 1}/{len(participants)})")

        # Load the participant's EDF file
        ecg_edf = mne.io.read_raw_edf(filepaths[p], preload=True, verbose=False)
        ecg_signal = ecg_edf.get_data()[0]
        n_samples = len(ecg_signal)
        start_time = ecg_edf.annotations.orig_time

        if start_time is None:
            print(f"Skipping {participant_id}, no start time in EDF.")
            continue

        # Filter metadata for this participant
        timestamps_subj = timestamps[timestamps["Subject_ID"] == participant_id]

        # Create participant group in HDF5
        participant_group = f.create_group(f"participant_{participant_id}")

        # Initialize a dictionary to store segments grouped by label
        grouped_segments = {category: [] for category in CATEGORY_MAPPING.keys()}

        # Loop through each label/interval
        for _, row in timestamps_subj.iterrows():
            category = row["Category"]

            # Determine the label for this category
            label = None
            for key, categories in CATEGORY_MAPPING.items():
                if category in categories:
                    label = key
                    break

            if label is None:
                print(f"Category {category} does not match any label. Skipping.")
                continue

            # Interval times
            label_start = row["LabelStart"]
            label_end = row["LabelEnd"]

            # Convert start/end times to sample indices
            idx_start = int((label_start - start_time).total_seconds() * fs)
            idx_end = int((label_end - start_time).total_seconds() * fs)

            # Clip indices to valid range
            idx_start = max(0, idx_start)
            idx_end = min(n_samples, idx_end)

            if idx_end <= idx_start:
                print(f"Invalid interval for {participant_id}: {row}")
                continue

            # Extract the segment and store it under the appropriate label
            ecg_segment = ecg_signal[idx_start:idx_end]
            grouped_segments[label].append(ecg_segment)

        # Save grouped segments for this participant
        for label, segments in grouped_segments.items():
            if not segments:
                print(f"No data for label {label} for participant {participant_id}.")
                continue

            # Concatenate all segments for this label
            concatenated_segments = np.concatenate(segments)

            # Create dataset in HDF5
            dataset_name = f"{label}"
            participant_group.create_dataset(
                dataset_name,
                data=concatenated_segments,
                compression="gzip",
                compression_opts=4,
            )

print(f"Data saved to {hdf5_filename}")
