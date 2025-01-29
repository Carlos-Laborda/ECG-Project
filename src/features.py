import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.signal
import scipy.stats
from scipy.signal import welch
from utils import load_ecg_data


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


# Path to the HDF5 file
hdf5_path = "../data/interim/ecg_data.h5"

# Sampling frequency
fs = 1000  # Hz
window_size = 10 * fs  # 10 seconds
step_size = 1 * fs  # 1 second

# Load the grouped data from HDF5
data = load_ecg_data(hdf5_path)

# Initialize a list to store extracted features
all_features = []

for (participant_id, category), signal in data.items():
    print(f"Processing Participant: {participant_id}, Category: {category}")

    # Apply sliding window
    windows = sliding_window(signal, window_size, step_size)

    # Extract features for each window
    for i, window in enumerate(windows):
        features = extract_ecg_features(window, fs=fs)
        features["participant_id"] = participant_id
        features["category"] = category
        features["window_index"] = i
        all_features.append(features)

# Convert to a DataFrame
features_df = pd.DataFrame(all_features)


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


# Example plot
plot_feature_distributions(features_df, feature="mean")

# Save to Parquet for efficiency purposes
output_path = "../data/processed/features.parquet"
features_df.to_parquet(output_path, index=False)
print(f"Features saved to {output_path}")
