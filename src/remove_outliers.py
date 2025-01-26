import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from utils import load_ecg_data

# ---------------------------------------------------------------------------------------------------------------------
# Adjust plot settings
# ---------------------------------------------------------------------------------------------------------------------
mpl.style.use("fivethirtyeight")
mpl.rcParams["figure.figsize"] = (20, 5)
mpl.rcParams["figure.dpi"] = 100

# ---------------------------------------------------------------------------------------------------------------------
# Set the path where npy files are stored and load the data
# ---------------------------------------------------------------------------------------------------------------------
base_path = "../data/interim"

# Load data using the utility function
data = load_ecg_data(base_path)


# ---------------------------------------------------------------------------------------------------------------------
# Plotting Outliers Function
# ---------------------------------------------------------------------------------------------------------------------
def plot_ecg_outliers(
    ecg_signal, outlier_mask, title="ECG Outliers", sample_range=None
):
    """
    Plot outliers in an ECG signal based on a binary outlier mask.

    Args:
        ecg_signal (np.ndarray): The ECG signal as a 1D array.
        outlier_mask (np.ndarray): A binary mask of the same length as ecg_signal, where True marks an outlier.
        title (str): Title of the plot.
        sample_range (tuple, optional): Range of samples to plot (start, end). Defaults to full signal.
    """
    if sample_range:
        start, end = sample_range
        ecg_signal = ecg_signal[start:end]
        outlier_mask = outlier_mask[start:end]

    fig, ax = plt.subplots(figsize=(20, 5))
    ax.set_title(title)
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("ECG Signal")

    # Plot non-outliers in default color
    ax.plot(
        np.where(~outlier_mask)[0], ecg_signal[~outlier_mask], "+", label="Non-outlier"
    )
    # Plot outliers in red
    ax.plot(np.where(outlier_mask)[0], ecg_signal[outlier_mask], "r+", label="Outlier")

    plt.legend(loc="upper center", ncol=2, fancybox=True, shadow=True)
    plt.show()


# ---------------------------------------------------------------------------------------------------------------------
# Z-Score Outliers
# ---------------------------------------------------------------------------------------------------------------------
def z_score_outlier_detection(signal, threshold=3.0):
    mean = np.mean(signal)
    std = np.std(signal)
    z_scores = (signal - mean) / std
    return np.abs(z_scores) > threshold


# Iterate only over the first key-value pair in the dictionary
first_key_value = next(iter(data.items()))
(participant, category), signal = first_key_value
print(f"Processing Participant: {participant}, Category: {category}")

# Detect outliers using Z-Score
outliers = z_score_outlier_detection(signal, threshold=6.0)

# Plot the outliers
plot_ecg_outliers(
    signal, outliers, title=f"Z-Score Outliers: {participant}, {category}"
)


# ---------------------------------------------------------------------------------------------------------------------
# Median Absolute Deviation (MAD)
# ---------------------------------------------------------------------------------------------------------------------
def mad_outlier_detection(signal, threshold=3.5):
    median = np.median(signal)
    mad = np.median(np.abs(signal - median))
    modified_z_scores = 0.6745 * (signal - median) / mad
    return np.abs(modified_z_scores) > threshold


# Iterate only over the first key-value pair in the dictionary
first_key_value = next(iter(data.items()))
(participant, category), signal = first_key_value
print(f"Processing Participant: {participant}, Category: {category}")

# Detect outliers using MAD
outliers = mad_outlier_detection(signal, threshold=10)

# Plot the outliers
plot_ecg_outliers(signal, outliers, title=f"MAD Outliers: {participant}, {category}")

# ---------------------------------------------------------------------------------------------------------------------
# Anomaly Detection with Sliding Window
# ---------------------------------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------------------------------------
# Median Absolute Deviation (MAD)
# ---------------------------------------------------------------------------------------------------------------------
