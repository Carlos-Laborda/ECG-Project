import numpy as np
import pandas as pd
from scipy.signal import welch
from utils import load_ecg_data


################################################################################
# Extract Features Function
################################################################################
def extract_features(window, fs=1000):
    """
    Extract features from an ECG signal window.

    Args:
        window (np.ndarray): A 1D array containing the ECG signal for the window.
        fs (int): Sampling frequency of the ECG signal (default: 1000 Hz).

    Returns:
        dict: A dictionary of extracted features.
    """
    # Time-domain features
    mean = np.mean(window)
    std_dev = np.std(window)
    min_val = np.min(window)
    max_val = np.max(window)
    rms = np.sqrt(np.mean(window**2))

    # Frequency-domain features
    freqs, psd = welch(window, fs)
    lf_power = np.sum(psd[(freqs >= 0.04) & (freqs <= 0.15)])
    hf_power = np.sum(psd[(freqs > 0.15) & (freqs <= 0.4)])
    lf_hf_ratio = lf_power / hf_power if hf_power > 0 else np.nan

    # Combine features into a dictionary
    features = {
        "mean": mean,
        "std_dev": std_dev,
        "min_val": min_val,
        "max_val": max_val,
        "rms": rms,
        "lf_power": lf_power,
        "hf_power": hf_power,
        "lf_hf_ratio": lf_hf_ratio,
    }
    return features


def sliding_window(signal, window_size, step_size):
    """
    Generate sliding windows from a 1D signal.

    Args:
        signal (np.ndarray): The input ECG signal as a 1D array.
        window_size (int): The size of each window (in samples).
        step_size (int): The step size between consecutive windows (in samples).

    Returns:
        list: A list of 1D arrays, each representing a window.
    """
    num_windows = (len(signal) - window_size) // step_size + 1
    return [
        signal[i : i + window_size]
        for i in range(0, num_windows * step_size, step_size)
    ]


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
        features = extract_features(window, fs=fs)
        features["participant_id"] = participant_id
        features["category"] = category
        features["window_index"] = i
        all_features.append(features)

# Convert to a DataFrame
features_df = pd.DataFrame(all_features)

# Save to CSV
output_path = "../data/processed/features.csv"
features_df.to_csv(output_path, index=False)
print(f"Features saved to {output_path}")
