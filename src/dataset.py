import mne
import numpy as np
import os
import pandas as pd


# Set the folder path containing the raw ECG files and the metadata
folderpath = "../data/raw/Raw ECG project"
filepaths = [
    os.path.join(folderpath, f)
    for f in os.listdir(folderpath)
    if f.endswith("_ECG.edf")
]
files = [f for f in os.listdir(folderpath) if f.endswith("_ECG.edf")]
participants = [f[:5] for f in files]  # Extract participant IDs from filenames

# Output directory for processed segments
output_dir_path = "../data/interim/"

# Read the metadata file
timestamps = pd.read_csv(
    os.path.join(folderpath, "TimeStamps_Merged.txt"), sep="\t", decimal="."
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

for p, participant_id in enumerate(participants[:2]):  # only 2 for testing purposes
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

    # Initialize a list to store extracted segments
    segments = []

    # Loop through each label/interval
    for _, row in timestamps_subj.iterrows():
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

        # Extract the segment
        ecg_segment = ecg_signal[idx_start:idx_end]
        segments.append(
            {
                "Subject_ID": row["Subject_ID"],
                "Category": row["Category"],
                "Code": row["Code"],
                "ECG_Segment": ecg_segment,
                "LabelStart": row["LabelStart"],
                "LabelEnd": row["LabelEnd"],
            }
        )

    # Save Segments
    for segment in segments:
        category = segment["Category"]
        participant_dir = os.path.join(output_dir_path, participant_id)
        os.makedirs(participant_dir, exist_ok=True)

        # Save the segment as a NumPy array
        segment_filename = os.path.join(participant_dir, f"{category}.npy")
        np.save(segment_filename, segment["ECG_Segment"])
        print(f"Saved: {segment_filename}")
