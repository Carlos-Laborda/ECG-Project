import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.signal
import scipy.stats
from scipy.signal import welch
import mne
import os
import pandas as pd
from datetime import datetime
from config import CATEGORY_MAPPING, FOLDERPATH


def process_ecg_data():
    """
    Process ECG data and return it as a dictionary.

    Returns:
        dict: Processed ECG data. Keys are (participant, category) tuples, and values are numpy arrays.
    """
    data = {}

    # Set the folder path containing the raw ECG files and the metadata
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

    # Sampling frequency of the ECG data
    fs = 1000

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

        # Store grouped segments for this participant
        for label, segments in grouped_segments.items():
            if not segments:
                print(f"No data for label {label} for participant {participant_id}.")
                continue

            # Concatenate all segments for this label
            concatenated_segments = np.concatenate(segments)

            # Store the data in the dictionary
            data[(participant_id, label)] = concatenated_segments

    return data


def extract_ecg_features(ecg_signal, fs):
    """
    Extracts features from ECG signal.
    """
    features = {}

    # Time-domain features
    features["mean"] = np.mean(ecg_signal)
    features["std"] = np.std(ecg_signal)
    features["min"] = np.min(ecg_signal)
    features["max"] = np.max(ecg_signal)
    features["rms"] = np.sqrt(np.mean(ecg_signal**2))
    features["iqr"] = np.subtract(*np.percentile(ecg_signal, [75, 25]))

    # Frequency-domain features
    freq, psd = scipy.signal.welch(ecg_signal, fs=fs)
    features["psd_mean"] = np.mean(psd)
    features["psd_max"] = np.max(psd)
    features["dominant_freq"] = freq[np.argmax(psd)]

    # Nonlinear features
    features["shannon_entropy"] = -np.sum(
        np.log2(np.histogram(ecg_signal, bins=10)[0] + 1e-10)
    )
    features["sample_entropy"] = scipy.stats.entropy(
        np.histogram(ecg_signal, bins=10)[0]
    )

    return features


def sliding_window(signal, window_size, step_size):
    """
    Generate sliding windows from a 1D signal.
    """
    num_steps = int((len(signal) - window_size) / step_size) + 1
    return [
        signal[i : i + window_size] for i in range(0, num_steps * step_size, step_size)
    ]


def preprocess_features(data, fs=1000, window_size=10, step_size=1):
    """
    Preprocess ECG data to extract features.

    Args:
        data (dict): Dictionary of ECG data. Keys are (participant, category) tuples, and values are numpy arrays.
        fs (int): Sampling frequency of the ECG signal (default: 1000 Hz).
        window_size (int): Window size in seconds (default: 10 seconds).
        step_size (int): Step size in seconds (default: 1 second).

    Returns:
        pd.DataFrame: DataFrame containing the extracted features.
    """
    window_size_samples = window_size * fs
    step_size_samples = step_size * fs

    # Initialize a list to store extracted features
    all_features = []

    for (participant_id, category), signal in data.items():
        print(f"Processing Participant: {participant_id}, Category: {category}")

        # Apply sliding window
        windows = sliding_window(signal, window_size_samples, step_size_samples)

        # Extract features for each window
        for i, window in enumerate(windows):
            features = extract_ecg_features(window, fs=fs)
            features["participant_id"] = participant_id
            features["category"] = category
            features["window_index"] = i
            all_features.append(features)

    # Convert to a DataFrame
    features_df = pd.DataFrame(all_features)
    return features_df


def plot_feature_distributions(features_df, feature):
    """
    Overlay the distributions of a feature for all categories.
    """
    plt.figure(figsize=(12, 6))
    title = f"Distribution of {feature} Across All Categories"
    sns.kdeplot(
        data=features_df,
        x=feature,
        hue="category",
        fill=True,
        alpha=0.4,
    )
    plt.xlabel(feature)
    plt.ylabel("Density")
    plt.title(title)
    plt.grid()
    plt.show()
