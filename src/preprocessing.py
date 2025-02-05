#!/usr/bin/env python
import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import butter, filtfilt, iirnotch
from utils import load_ecg_data

# Config
hdf5_path = "../data/interim/ecg_data.h5"  # Raw data file path
output_hdf5_path = "../data/interim/ecg_data_cleaned.h5"  # Where to save cleaned data
fs = 1000  # Sampling frequency in Hz


# Filter Functions
def bandpass_filter(signal, fs, lowcut=0.5, highcut=40.0, order=5):
    """
    Apply a Butterworth bandpass filter to remove baseline wander and high-frequency noise.

    Parameters:
        signal (np.ndarray): The raw ECG signal.
        fs (int): Sampling frequency in Hz.
        lowcut (float): Lower cutoff frequency (default 0.5 Hz).
        highcut (float): Upper cutoff frequency (default 40 Hz).
        order (int): Filter order (default 5).

    Returns:
        np.ndarray: The bandpass filtered signal.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    # Use filtfilt for zero-phase filtering
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal


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


# Load raw data
print("Loading raw ECG data from HDF5...")
raw_data = load_ecg_data(hdf5_path)
print(f"Loaded data for {len(raw_data)} (participant, category) pairs.")

# Clean the data
cleaned_data = {}
for key, signal in raw_data.items():
    participant, category = key
    print(f"Cleaning signal for Participant {participant}, Category {category}...")
    try:
        cleaned_signal = clean_ecg_signal(signal, fs)
    except Exception as e:
        print(f"Error cleaning signal for {key}: {e}")
        continue
    cleaned_data[key] = cleaned_signal

# Plot Example Signal Before and After Cleaning
example_key = list(cleaned_data.keys())[1]
raw_signal_example = raw_data[example_key]
cleaned_signal_example = cleaned_data[example_key]

plt.figure(figsize=(12, 6))
plt.plot(raw_signal_example[:5000], label="Raw ECG")
plt.plot(cleaned_signal_example[:5000], label="Cleaned ECG")
plt.title(f"ECG Signal Cleaning Example for {example_key}")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.legend()
plt.show()

# Save Cleaned Data into a New HDF5 File
print(f"Saving cleaned data to {output_hdf5_path}...")
with h5py.File(output_hdf5_path, "w") as f_out:
    for (participant, category), signal in cleaned_data.items():
        group_name = f"participant_{participant}"
        if group_name not in f_out:
            grp = f_out.create_group(group_name)
        else:
            grp = f_out[group_name]
        # Save each category's cleaned signal
        grp.create_dataset(
            category, data=signal, compression="gzip", compression_opts=4
        )
print(f"Cleaned ECG data saved to {output_hdf5_path}")
