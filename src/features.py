import numpy as np
import pandas as pd
from scipy.signal import welch
from utils import load_ecg_data
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.signal
import scipy.stats


# Extract Features Function
def extract_ecg_features(ecg_signal, fs):
    """
    Extracts features from ECG signal.

    Args:
        ecg_signal (np.array): ECG signal
        fs (int): Sampling frequency

    Returns:
        dict: Dictionary of extracted features
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

    Args:
        signal (np.ndarray): The input ECG signal as a 1D array.
        window_size (int): The size of each window (in samples).
        step_size (float): The step size between consecutive windows (in samples).

    Returns:
        list: A list of 1D arrays, each representing a window.
    """
    # Calculate the number of steps as an integer
    step_ratio = step_size / window_size
    num_steps = int((len(signal) - window_size) / step_size) + 1

    # Generate indices using integer arithmetic
    indices = [i * step_size for i in range(num_steps)]
    indices = [int(round(idx)) for idx in indices]

    return [signal[i : i + window_size] for i in indices]


# Path to the data
base_path = "../data/interim"

# Sampling frequency
fs = 1000  # Hz
window_size = 10 * fs  # 10 seconds
step_size = 1 * fs  # 1 second

# Load the grouped data
data = load_ecg_data(base_path)

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


# Plot the distributions of the features for each category
def plot_feature_distributions(features_df, feature):
    """
    Overlay the distributions of a feature for all categories.

    Args:
        features_df (pd.DataFrame): The DataFrame containing the features.
        feature (str): The feature to plot (column name in the DataFrame).
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


plot_feature_distributions(features_df, feature="mean")


# Save to CSV
output_path = "../data/processed/features.csv"
features_df.to_csv(output_path, index=False)
print(f"Features saved to {output_path}")
