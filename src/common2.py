import os
from datetime import datetime

import h5py
import mne
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch

from config import CATEGORY_MAPPING, FOLDERPATH, OUTPUT_DIR_PATH
from utils import load_ecg_data

# ------------------------------------------------------
# Match ECG data with labels and save segments individually
# ------------------------------------------------------
def process_ecg_data(hdf5_path):
    """
    Process raw ECG data and write it to an HDF5 file at hdf5_path.
    Each segment is stored individually under its participant and category.
    """
    filepaths = [
        os.path.join(FOLDERPATH, f)
        for f in os.listdir(FOLDERPATH)
        if f.endswith("_ECG.edf")
    ]
    files = [f for f in os.listdir(FOLDERPATH) if f.endswith("_ECG.edf")]
    participants = [f[:5] for f in files]  # Extract participant IDs from filenames

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

    fs = 1000  # sampling frequency

    # Create or overwrite HDF5 file
    with h5py.File(hdf5_path, "w") as f:
        for p, participant_id in enumerate(participants[:2]):  # 2 participants for now
            print(f"Processing Participant {participant_id} ({p + 1}/{len(participants)})")

            ecg_edf = mne.io.read_raw_edf(filepaths[p], preload=True, verbose=False)
            ecg_signal = ecg_edf.get_data()[0]
            n_samples = len(ecg_signal)
            start_time = ecg_edf.annotations.orig_time

            if start_time is None:
                print(f"Skipping {participant_id}, no start time in EDF.")
                continue

            # Filter metadata for this participant
            timestamps_subj = timestamps[timestamps["Subject_ID"] == participant_id]

            # Create participant group
            participant_group = f.create_group(f"participant_{participant_id}")

            # Loop through intervals and save each segment individually
            for idx, row in timestamps_subj.iterrows():
                category = row["Category"]
                label = None
                for key, cat_list in CATEGORY_MAPPING.items():
                    if category in cat_list:
                        label = key
                        break

                if label is None:
                    print(f"Category {category} not found in mapping. Skipping.")
                    continue

                label_start = row["LabelStart"]
                label_end = row["LabelEnd"]

                idx_start = int((label_start - start_time).total_seconds() * fs)
                idx_end = int((label_end - start_time).total_seconds() * fs)

                idx_start = max(0, idx_start)
                idx_end = min(n_samples, idx_end)

                if idx_end <= idx_start:
                    print(f"Invalid interval for {participant_id}: {row}")
                    continue

                segment = ecg_signal[idx_start:idx_end].astype(np.float32)

                # Create or get category group under participant
                if label not in participant_group:
                    category_group = participant_group.create_group(label)
                else:
                    category_group = participant_group[label]
                
                # Save this segment under a unique dataset name
                seg_name = f"segment_{len(category_group.keys())}"
                category_group.create_dataset(
                    seg_name,
                    data=segment,
                    compression="gzip",
                    compression_opts=4,
                    dtype=np.float32,
                )

    print(f"ECG data successfully written to {hdf5_path}")

# ------------------------------------------------------
# Cleaning functions
# ------------------------------------------------------
def highpass_filter(signal, fs, cutoff=0.5, order=5):
    nyq = 0.5 * fs
    high = cutoff / nyq
    b, a = butter(order, high, btype="high")
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal

def notch_filter(signal, fs, notch_freq=50.0, quality_factor=30):
    nyq = 0.5 * fs
    w0 = notch_freq / nyq
    b, a = iirnotch(w0, quality_factor)
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal

def clean_ecg_signal(signal, fs):
    hp_filtered = highpass_filter(signal, fs, cutoff=0.5, order=5)
    cleaned_signal = notch_filter(hp_filtered, fs, notch_freq=50.0, quality_factor=30)
    return cleaned_signal

# ------------------------------------------------------
# Clean each individual segment and save the cleaned versions
# ------------------------------------------------------
def process_save_cleaned_data(segmented_data_path, output_hdf5_path, fs=1000):
    """
    Loads ECG data from segmented_data_path, cleans each individual segment,
    and saves the cleaned segments to output_hdf5_path preserving the structure.
    """
    with h5py.File(segmented_data_path, "r") as f_in, h5py.File(output_hdf5_path, "w") as f_out:
        for participant in f_in.keys():
            print(f"Cleaning data for {participant}...")
            participant_in = f_in[participant]
            participant_out = f_out.create_group(participant)
            for category in participant_in.keys():
                category_in = participant_in[category]
                category_out = participant_out.create_group(category)
                for segment_name in category_in.keys():
                    signal = category_in[segment_name][...]
                    try:
                        cleaned_signal = clean_ecg_signal(signal, fs)
                    except Exception as e:
                        print(f"Error cleaning signal for {participant}/{category}/{segment_name}: {e}")
                        continue
                    # Save cleaned segment with same segment name
                    category_out.create_dataset(
                        segment_name,
                        data=cleaned_signal.astype(np.float32),
                        compression="gzip",
                        compression_opts=4,
                        dtype='float32'
                    )
    print(f"Cleaned ECG data saved to {output_hdf5_path}")

# ------------------------------------------------------
# 4. Sliding window function
# ------------------------------------------------------
def sliding_window(signal: np.ndarray, window_size: int, step_size: int):
    n_samples = len(signal)
    num_steps = (n_samples - window_size) // step_size + 1
    if num_steps < 1:
        return []
    windows = []
    for i in range(num_steps):
        start = i * step_size
        end = start + window_size
        windows.append(signal[start:end])
    return windows

# ------------------------------------------------------
# 5. Segment cleaned data into windows for each individual segment
# ------------------------------------------------------
def segment_data_into_windows(cleaned_data_path, hdf5_path, fs=1000, window_size=10, step_size=5):
    """
    Reads cleaned ECG segments from cleaned_data_path, applies a sliding window to
    each individual segment (so that windows do not span discontinuities), and writes
    the windowed data to hdf5_path preserving the participant/category/segment structure.
    """
    window_size_samples = window_size * fs
    step_size_samples = step_size * fs
    
    with h5py.File(cleaned_data_path, "r") as f_in, h5py.File(hdf5_path, "w") as f_out:
        for participant in f_in.keys():
            print(f"Windowing data for {participant}...")
            participant_in = f_in[participant]
            participant_out = f_out.create_group(participant)
            for category in participant_in.keys():
                category_in = participant_in[category]
                category_out = participant_out.create_group(category)
                for segment_name in category_in.keys():
                    signal = category_in[segment_name][...]
                    windows_list = sliding_window(signal, window_size_samples, step_size_samples)
                    if len(windows_list) == 0:
                        print(f"-> No windows for {participant}/{category}/{segment_name} (segment too short). Skipping.")
                        continue
                    windows_array = np.array(windows_list, dtype=np.float32)
                    # Save windows under the same segment name
                    category_out.create_dataset(
                        segment_name,
                        data=windows_array,
                        compression="gzip",
                        compression_opts=4
                    )
                    print(f"-> {participant}/{category}/{segment_name}: {windows_array.shape[0]} windows stored.")
    print(f"Segmented data saved to {hdf5_path}")
