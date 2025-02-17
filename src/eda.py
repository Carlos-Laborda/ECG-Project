import numpy as np
import h5py
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

from utils import prepare_cnn_data

def exploratory_analysis(hdf5_path, label_map=None):
    """
    Loads the windowed ECG data with `prepare_cnn_data` and performs
    exploratory data analysis comparing 'baseline' vs. 'mental_stress' windows.
    """
    if label_map is None:
        label_map = {"baseline": 0, "mental_stress": 1}
    
    # ---------------------------
    # 1) Load data via prepare_cnn_data
    # ---------------------------
    X, y, groups = prepare_cnn_data(hdf5_path, label_map=label_map)

    print("---- Data Shapes ----")
    print(f"X shape: {X.shape}   (N, window_length, 1)")
    print(f"y shape: {y.shape}   (N,)")
    print(f"groups shape: {groups.shape}")
    X_2d = X[..., 0]  # shape (N, window_length)

    # ---------------------------
    # 2) Label distribution
    # ---------------------------
    print("\n---- Label Distribution ----")
    label_counts = Counter(y)
    for label_val, count_val in label_counts.items():
        label_name = [k for k, v in label_map.items() if v == label_val]
        label_str = label_name[0] if label_name else f"label_{label_val}"
        print(f"{label_str}: {count_val} samples")
    # e.g. baseline: XXX, mental_stress: YYY

    # ---------------------------
    # 3) Basic descriptive stats per label
    # ---------------------------

    # 2) Quick data check: random sample
    idx = np.random.choice(X_2d.shape[0])
    print(f"\nRandom sample index: {idx}")
    print("First 30 samples of that window:", X_2d[idx, :30])

    print("Global min:", X_2d.min())
    print("Global max:", X_2d.max())
    print("Global mean:", np.mean(X_2d))

    # 3) Basic stats across windows
    sample_means = np.mean(X_2d, axis=1)
    baseline_mask = (y == label_map['baseline'])
    stress_mask = (y == label_map['mental_stress'])

    baseline_means = sample_means[baseline_mask]
    stress_means = sample_means[stress_mask]

    print("\n---- Basic Stats for Mean ECG amplitude per window ----")


    print(f"Baseline: mean={baseline_means.mean():.8f}, std={baseline_means.std():.8f}")
    print(f"Mental stress: mean={stress_means.mean():.8f}, std={stress_means.std():.8f}")

    # Scientific 
    print("(in scientific notation)")
    print("Baseline: mean={}, std={}".format(
        np.format_float_scientific(baseline_means.mean(), precision=6),
        np.format_float_scientific(baseline_means.std(), precision=6)
    ))
    print("Mental stress: mean={}, std={}".format(
        np.format_float_scientific(stress_means.mean(), precision=6),
        np.format_float_scientific(stress_means.std(), precision=6)
    ))

    # ---------------------------
    # 4) Visual comparisons
    # ---------------------------

    # (a) Boxplot or violin plot
    plt.figure(figsize=(8, 5))
    sns.boxplot(
        x=['baseline'] * len(baseline_means) + ['mental_stress'] * len(stress_means),
        y=np.concatenate([baseline_means, stress_means])
    )
    plt.title("Distribution of Mean ECG Amplitude (per window)")
    plt.ylabel("Mean ECG amplitude")
    plt.show()

    # (b) Histograms or kernel density estimates
    plt.figure(figsize=(8, 5))
    sns.kdeplot(baseline_means, label='baseline', shade=True)
    sns.kdeplot(stress_means, label='mental_stress', shade=True)
    plt.title("KDE Plot of Mean ECG amplitude (Baseline vs. Stress)")
    plt.xlabel("Mean ECG amplitude")
    plt.legend()
    plt.show()

    # (c) Possibly correlation with number of participants, etc.
    print("\n---- Windows per participant ----")
    participant_counts = Counter(groups)
    for pid, cnt in participant_counts.items():
        print(f"Participant {pid}: {cnt} windows")

    return

exploratory_analysis("../data/interim/windowed_data.h5")
