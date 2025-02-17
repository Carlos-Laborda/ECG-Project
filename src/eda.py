import numpy as np
import h5py
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

from utils import prepare_cnn_data

def exploratory_analysis_ecg(hdf5_path, label_map=None):
    """
    Loads windowed ECG data with prepare_cnn_data and performs
    extended EDA on various per-window stats, comparing baseline vs. mental stress.
    """
    if label_map is None:
        label_map = {"baseline": 0, "mental_stress": 1}

    # 1) Load data
    X, y, groups = prepare_cnn_data(hdf5_path, label_map=label_map)
    # X shape: (N, window_length, 1)
    # Flatten out the last channel => (N, window_length)
    X_2d = X[..., 0]

    print("---- Data Shapes ----")
    print(f"X shape: {X.shape}   (N, window_length, 1)")
    print(f"y shape: {y.shape}   (N,)")
    print(f"groups shape: {groups.shape}")

    # 2) Label distribution
    print("\n---- Label Distribution ----")
    label_counts = Counter(y)
    for label_val, count_val in label_counts.items():
        label_name = [k for k, v in label_map.items() if v == label_val]
        label_str = label_name[0] if label_name else f"label_{label_val}"
        print(f"{label_str}: {count_val} samples")

    # 3) Quick data check
    idx = np.random.choice(X_2d.shape[0])
    print(f"\nRandom sample index: {idx}")
    print("First 30 samples of that window:", X_2d[idx, :30])
    print("Global min:", X_2d.min())
    print("Global max:", X_2d.max())
    print("Global mean:", np.mean(X_2d))

    # 4) Compute some window-level statistics
    # For each window i:
    # - mean_i = mean(X_2d[i])
    # - std_i = std(X_2d[i])
    # - range_i = max(X_2d[i]) - min(X_2d[i])
    # - rms_i = sqrt(mean(X_2d[i]^2))

    window_means = np.mean(X_2d, axis=1)
    window_stds = np.std(X_2d, axis=1)
    window_mins = np.min(X_2d, axis=1)
    window_maxs = np.max(X_2d, axis=1)
    window_ranges = window_maxs - window_mins
    window_rms = np.sqrt(np.mean(X_2d**2, axis=1))

    # Split by label
    baseline_mask = (y == label_map['baseline'])
    stress_mask = (y == label_map['mental_stress'])

    baseline_means = window_means[baseline_mask]
    stress_means   = window_means[stress_mask]

    baseline_stds  = window_stds[baseline_mask]
    stress_stds    = window_stds[stress_mask]

    baseline_rng   = window_ranges[baseline_mask]
    stress_rng     = window_ranges[stress_mask]

    baseline_rms   = window_rms[baseline_mask]
    stress_rms     = window_rms[stress_mask]

    print("\n---- Extended Stats (per-window) ----")
    def print_stats(label, arr):
        print(f"{label}:")
        # Use scientific notation or more decimals to see very small values
        print(f"  Mean of means: {np.mean(arr):.8e}")
        print(f"  Std of means:  {np.std(arr):.8e}")
        print(f"  Min:           {arr.min():.8e}")
        print(f"  Max:           {arr.max():.8e}")
        print(f"  Median:        {np.median(arr):.8e}")

    # Means
    print_stats("Baseline (mean amplitude)", baseline_means)
    print_stats("Mental stress (mean amplitude)", stress_means)

    # Standard dev
    print_stats("Baseline (std dev)", baseline_stds)
    print_stats("Mental stress (std dev)", stress_stds)

    # Range
    print_stats("Baseline (range)", baseline_rng)
    print_stats("Mental stress (range)", stress_rng)

    # RMS
    print_stats("Baseline (RMS)", baseline_rms)
    print_stats("Mental stress (RMS)", stress_rms)

    # 5) Some plots
    # We'll demonstrate range & RMS as examples

    # (a) Boxplot of range
    plt.figure(figsize=(10, 4))
    sns.boxplot(
        x=['baseline'] * len(baseline_rng) + ['mental_stress'] * len(stress_rng),
        y=np.concatenate([baseline_rng, stress_rng])
    )
    plt.title("Distribution of (Max - Min) range in windows")
    plt.ylabel("ECG amplitude range")
    plt.show()

    # (b) Distribution of RMS
    plt.figure(figsize=(10, 4))
    sns.kdeplot(baseline_rms, label='baseline (RMS)', shade=True)
    sns.kdeplot(stress_rms, label='mental_stress (RMS)', shade=True)
    plt.title("KDE Plot of RMS amplitude (Baseline vs. Stress)")
    plt.xlabel("RMS amplitude")
    plt.legend()
    plt.show()

    # 6) Windows per participant
    print("\n---- Windows per participant ----")
    participant_counts = Counter(groups)
    for pid, cnt in participant_counts.items():
        print(f"Participant {pid}: {cnt} windows")

exploratory_analysis_ecg("../data/interim/windowed_data.h5")