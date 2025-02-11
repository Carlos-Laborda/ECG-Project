import os
from datetime import datetime

import h5py
import mne
import numpy as np
import pandas as pd
import scipy.signal
import scipy.stats
import keras
from scipy.signal import butter, filtfilt, iirnotch
from keras import Input, layers, models, optimizers, losses, metrics

from config import CATEGORY_MAPPING, FOLDERPATH, OUTPUT_DIR_PATH
from utils import load_ecg_data

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
        for p, participant_id in enumerate(participants[:20]):  # first 10 participants
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


# Clean the data
def highpass_filter(signal, fs, cutoff=0.5, order=5):
    """
    Apply a Butterworth high-pass filter to remove baseline wander.

    Parameters:
        signal (np.ndarray): The raw ECG signal.
        fs (int): Sampling frequency in Hz.
        cutoff (float): Cutoff frequency (default 0.5 Hz).
        order (int): Filter order (default 5).

    Returns:
        np.ndarray: The high-pass filtered signal.
    """
    nyq = 0.5 * fs
    high = cutoff / nyq
    b, a = butter(order, high, btype="high")
    # Use filtfilt for zero-phase filtering
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal

def notch_filter(signal, fs, notch_freq=50.0, quality_factor=30):
    """
    Apply a notch filter to remove powerline interference.

    Parameters:
        signal (np.ndarray): The ECG signal.
        fs (int): Sampling frequency in Hz.
        notch_freq (float): Notch frequency (default 50 Hz).
        quality_factor (float): Quality factor (default 30).

    Returns:
        np.ndarray: The notch filtered signal.
    """
    nyq = 0.5 * fs
    w0 = notch_freq / nyq
    b, a = iirnotch(w0, quality_factor)
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal


def clean_ecg_signal(signal, fs):
    """
    Clean an ECG signal by first applying a high-pass filter (0.5Hz)
    and then a 50 Hz notch filter.

    Parameters:
        signal (np.ndarray): The raw ECG signal.
        fs (int): Sampling frequency.

    Returns:
        np.ndarray: The cleaned ECG signal.
    """
    # Apply high-pass filter to remove baseline wander
    hp_filtered = highpass_filter(signal, fs, cutoff=0.5, order=5)

    # Apply notch filter to remove powerline interference
    cleaned_signal = notch_filter(hp_filtered, fs, notch_freq=50.0, quality_factor=30)
    return cleaned_signal


def process_save_cleaned_data(segmented_data_path, output_hdf5_path, fs=1000):
    """
    Loads ECG data from segmented_data_path, cleans it, and saves the cleaned signals to output_hdf5_path.
    """
    raw_data = load_ecg_data(segmented_data_path)

    cleaned_data = {}
    for key, signal in raw_data.items():
        participant, category = key
        try:
            cleaned_signal = clean_ecg_signal(signal, fs)
        except Exception as e:
            print(f"Error cleaning signal for {key}: {e}")
            continue
        cleaned_data[key] = cleaned_signal

    print(f"Saving cleaned data to {output_hdf5_path}...")
    with h5py.File(output_hdf5_path, "w") as f_out:
        for (participant, category), signal in cleaned_data.items():
            group_name = f"participant_{participant}"
            if group_name not in f_out:
                grp = f_out.create_group(group_name)
            else:
                grp = f_out[group_name]
            grp.create_dataset(
                category, data=signal, compression="gzip", compression_opts=4
            )
    print(f"Cleaned ECG data saved to {output_hdf5_path}")

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


def sliding_window(signal: np.ndarray, window_size: int, step_size: int):
    """
    Generate sliding windows from a 1D signal.

    Parameters:
      signal (np.ndarray): Input 1D array
      window_size (int): Number of samples in each window
      step_size (int): Number of samples between the start of consecutive windows

    Returns:
      windows (list of np.ndarray): Each entry is a 1D array of length `window_size`.
    """
    n_samples = len(signal)
    num_steps = (n_samples - window_size) // step_size + 1
    if num_steps < 1:
        return []

    windows = []
    for i in range(num_steps):
        start = i * step_size
        end = start + window_size
        window = signal[start:end]
        windows.append(window)
    return windows


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


# ------------------------------------------------------
# Segmenting data into windows
# ------------------------------------------------------
def segment_data_into_windows(data, hdf5_path, fs=1000, window_size=10, step_size=1):
    window_size_samples = window_size * fs
    step_size_samples = step_size * fs
    
    with h5py.File(hdf5_path, "w") as f_out:
        for (participant_id, category), signal in data.items():
            grp = f_out.require_group(f"participant_{participant_id}")
            windows_list = sliding_window(
                signal, window_size_samples, step_size_samples
            )
            if len(windows_list) == 0:
                print(
                    f"-> No windows for {participant_id}/{category} (too short?). Skipping."
                )
                continue
            windows_array = np.array(windows_list, dtype=np.float32)
            grp.create_dataset(
                category, data=windows_array, compression="gzip", compression_opts=4
            )
            print(
                f"-> {participant_id}/{category}: {windows_array.shape[0]} windows stored."
            )
    print(f"Segmented data saved to {hdf5_path}")

# ------------------------------------------------------
# 4) Model Building
# ------------------------------------------------------
def baseline_1DCNN(input_shape=(10000, 1)):
    """
    Build a minimal 1D CNN for binary ECG classification.

    Args:
        input_shape (tuple): Shape of the input signal, e.g. (window_length, channels=1).

    Returns:
        keras.Model: A compiled Keras model with JAX as backend.
    """
    model = models.Sequential([
        layers.Conv1D(8, kernel_size=3, activation="relu", input_shape=input_shape),
        layers.GlobalAveragePooling1D(),  
        layers.Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss=losses.BinaryCrossentropy(),
        metrics=[metrics.BinaryAccuracy()],
    )

    return model

def baseline_1DCNN_improved(input_shape=(10000, 1)):
    """
    Build a minimal 1D CNN for binary ECG classification.

    Args:
        input_shape (tuple): Shape of the input signal, e.g. (window_length, channels=1).

    Returns:
        keras.Model: A compiled Keras model with JAX as backend.
    """
    model = models.Sequential([
        # Input layer
        layers.Input(shape=input_shape),
        
        # First convolutional block
        layers.Conv1D(32, kernel_size=5, activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.2),
        
        # Second convolutional block
        layers.Conv1D(64, kernel_size=3, activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.2),
        
        # Third convolutional block
        layers.Conv1D(128, kernel_size=3, activation="relu"),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling1D(),
        layers.Dropout(0.3),
        
        # Output layer
        layers.Dense(64, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer=optimizers.Adam(),
        loss=losses.BinaryCrossentropy(),
        metrics=[metrics.BinaryAccuracy()],
    )

    return model

def neural_network(input_shape=(10000, 1)):
    """Build and compile the neural network to predict the species of a penguin."""

    model = models.Sequential([
        # Reshape input to be 1D
        layers.Reshape((input_shape,), input_shape=(input_shape,)),
        
        # Dense layers
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(32, activation="relu"),
        
        # Output layer for binary classification
        layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.01),  # Changed to Adam
        loss="binary_crossentropy",
        metrics=["binary_accuracy"]
    )

    return model
