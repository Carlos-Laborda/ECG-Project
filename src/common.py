import os
import mne
import h5py
import numpy as np
import pandas as pd
import scipy.signal
import scipy.stats
from datetime import datetime

from config import CATEGORY_MAPPING, FOLDERPATH, OUTPUT_DIR_PATH


# ------------------------------------------------------
# 1) process_ecg_data with argument for HDF5 output path
# ------------------------------------------------------
def process_ecg_data(hdf5_path):
    """
    Process raw ECG data and write it to an HDF5 file at hdf5_path.
    Expects global config for FOLDERPATH, CATEGORY_MAPPING, etc.
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
        for p, participant_id in enumerate(participants[:10]):  # first 10 participants
            print(
                f"Processing Participant {participant_id} ({p + 1}/{len(participants)})"
            )

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

            # Prepare container for each category
            grouped_segments = {category: [] for category in CATEGORY_MAPPING.keys()}

            # Loop through intervals
            for _, row in timestamps_subj.iterrows():
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

                segment = ecg_signal[idx_start:idx_end]
                grouped_segments[label].append(segment)

            # Save grouped segments to HDF5
            for label, segments in grouped_segments.items():
                if not segments:
                    print(f"No data for {label} (Participant {participant_id}).")
                    continue

                concatenated_segments = np.concatenate(segments)
                participant_group.create_dataset(
                    label,
                    data=concatenated_segments,
                    compression="gzip",
                    compression_opts=4,
                )

    print(f"ECG data successfully written to {hdf5_path}")


# ------------------------------------------------------
# 2) Helper Functions for Feature Extraction
# ------------------------------------------------------
def extract_ecg_features(ecg_signal, fs):
    """
    Extracts features from ECG signal.
    """
    features = {}
    # Time-domain
    features["mean"] = np.mean(ecg_signal)
    features["std"] = np.std(ecg_signal)
    features["min"] = np.min(ecg_signal)
    features["max"] = np.max(ecg_signal)
    features["rms"] = np.sqrt(np.mean(ecg_signal**2))
    features["iqr"] = np.subtract(*np.percentile(ecg_signal, [75, 25]))

    # Frequency-domain
    freq, psd = scipy.signal.welch(ecg_signal, fs=fs)
    features["psd_mean"] = np.mean(psd)
    features["psd_max"] = np.max(psd)
    features["dominant_freq"] = freq[np.argmax(psd)]

    # Nonlinear
    hist_vals, _ = np.histogram(ecg_signal, bins=10)
    features["shannon_entropy"] = -np.sum(np.log2(hist_vals + 1e-10))
    features["sample_entropy"] = scipy.stats.entropy(hist_vals)

    return features


def sliding_window(signal, window_size, step_size):
    """
    Generate sliding windows from a 1D signal.
    """
    num_steps = int((len(signal) - window_size) / step_size) + 1
    return [
        signal[i : i + window_size] for i in range(0, num_steps * step_size, step_size)
    ]


# ------------------------------------------------------
# 3) preprocess_features
# ------------------------------------------------------
def preprocess_features(data, fs=1000, window_size=10, step_size=1):
    """
    Preprocess ECG data to extract features.

    Args:
        data (dict): Dictionary of ECG data. (participant, category) -> np.array
        fs (int): Sampling frequency (default=1000).
        window_size (int): Window size in seconds (default=10).
        step_size (int): Step size in seconds (default=1).

    Returns:
        pd.DataFrame: DataFrame containing the extracted features.
    """
    window_size_samples = window_size * fs
    step_size_samples = step_size * fs

    all_features = []

    for (participant_id, category), signal in data.items():
        # Generate sliding windows
        windows = sliding_window(signal, window_size_samples, step_size_samples)
        for i, window in enumerate(windows):
            feats = extract_ecg_features(window, fs)
            feats["participant_id"] = participant_id
            feats["category"] = category
            feats["window_index"] = i
            all_features.append(feats)

    features_df = pd.DataFrame(all_features)
    return features_df
