import numpy as np
from scipy.signal import welch


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
