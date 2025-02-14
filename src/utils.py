import os
import numpy as np
import h5py


def load_ecg_data(hdf5_path):
    """
    Load ECG data from an HDF5 file.

    Args:
        hdf5_path (str): Path to the HDF5 file.

    Returns:
        dict: A dictionary of loaded data. Keys are (participant, category) tuples, and values are numpy arrays.
    """
    data = {}

    with h5py.File(hdf5_path, "r") as f:
        # Iterate through participants
        for participant_group in f.keys():
            participant_id = participant_group.split("_")[1]

            # Iterate through categories
            for category_dataset in f[participant_group].keys():
                category = category_dataset

                # Get the ECG signal data
                ecg_signal = f[participant_group][category_dataset][:]

                # Store the data with the correct key structure
                data[(participant_id, category)] = ecg_signal

    return data


def prepare_cnn_data(hdf5_path, label_map=None):
    """
    Load windowed ECG data from an HDF5 file and prepare it for a 1D CNN.

    Expects the HDF5 file to have the structure:
      /participant_XXXX/
         -> category (baseline, mental_stress, etc.)
            shape = (num_windows, window_length)

    Args:
        hdf5_path (str): Path to the HDF5 file with windowed data (already cleaned & segmented).
        label_map (dict): Mapping from category (string) to integer label.
                          Example: { "baseline": 0, "mental_stress": 1 }

    Returns:
        X (np.ndarray): shape (N, window_length, 1)
        y (np.ndarray): shape (N,)
        groups (np.ndarray): shape (N,) - each window's participant ID
    """
    if label_map is None:
        label_map = {"baseline": 0, "mental_stress": 1}

    X_list = []
    y_list = []
    groups_list = []

    with h5py.File(hdf5_path, "r") as f:
        participants = list(
            f.keys()
        )  # e.g. ["participant_30101", "participant_30106", ...]
        for participant_key in participants:
            # parse participant ID
            participant_id = participant_key.replace("participant_", "")
            cat_keys = list(f[participant_key].keys())

            for cat in cat_keys:
                # skip category if not in label_map
                if cat not in label_map:
                    continue

                windows_2d = f[participant_key][cat][
                    :
                ]  # shape (num_windows, window_length)
                label_val = label_map[cat]
                n_windows = windows_2d.shape[0]

                # participant groups
                groups_arr = np.array([participant_id] * n_windows, dtype=object)

                X_list.append(windows_2d)
                y_list.append(np.full((n_windows,), label_val, dtype=int))
                groups_list.append(groups_arr)

    if len(X_list) == 0:
        raise ValueError(
            f"No valid data found in {hdf5_path} with label_map {label_map}."
        )

    # Concatenate
    X = np.concatenate(X_list, axis=0)  # shape (N, window_length)
    y = np.concatenate(y_list, axis=0)  # shape (N,)
    groups = np.concatenate(groups_list, axis=0)  # shape (N,)

    # Expand dims for CNN => (N, window_length, 1)
    X = np.expand_dims(X, axis=-1)

    return X, y, groups

window_data_path = "../data/interim/windowed_data.h5"

X, y, groups = prepare_cnn_data(
    hdf5_path=window_data_path,
    label_map={"baseline": 0, "mental_stress": 1},)

len(X), len(y), len(groups)

# shapes
X.shape, y.shape, groups.shape

# visualize the data
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple
import pandas as pd

def plot_ecg_samples(X: np.ndarray, y: np.ndarray, groups: np.ndarray, 
                    num_samples: int = 2) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot sample ECG signals from each class.
    
    Args:
        X: Input data of shape (N, window_length, 1)
        y: Labels of shape (N,)
        groups: Participant IDs
        num_samples: Number of samples to plot per class
    """
    fig, axes = plt.subplots(2, num_samples, figsize=(15, 6))
    classes = ['Baseline', 'Mental Stress']
    
    for class_idx in [0, 1]:
        # Get indices for current class
        class_mask = y == class_idx
        sample_idx = np.where(class_mask)[0][:num_samples]
        
        for i, idx in enumerate(sample_idx):
            signal = X[idx, :, 0]
            participant = groups[idx]
            
            axes[class_idx, i].plot(signal, 'b-', linewidth=0.5)
            axes[class_idx, i].set_title(f'{classes[class_idx]} - Participant {participant}')
            axes[class_idx, i].set_xlabel('Time (samples)')
            axes[class_idx, i].set_ylabel('Amplitude')
            axes[class_idx, i].grid(True)
    
    plt.tight_layout()
    return fig, axes

fig1, _ = plot_ecg_samples(X, y, groups)

def plot_class_distribution(y: np.ndarray, groups: np.ndarray) -> plt.Figure:
    """
    Plot distribution of classes overall and per participant.
    
    Args:
        y: Labels of shape (N,)
        groups: Participant IDs
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Overall class distribution
    sns.countplot(x=y, ax=ax1)
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(['Baseline', 'Mental Stress'])
    ax1.set_title('Overall Class Distribution')
    
    # Class distribution per participant
    participant_stats = pd.DataFrame({
        'Participant': groups,
        'Class': y
    }).groupby('Participant')['Class'].value_counts().unstack()
    
    participant_stats.plot(kind='bar', ax=ax2)
    ax2.set_title('Class Distribution per Participant')
    ax2.set_xlabel('Participant ID')
    ax2.set_ylabel('Count')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    return fig

fig2 = plot_class_distribution(y, groups)


def plot_signal_stats(X: np.ndarray, y: np.ndarray) -> plt.Figure:
    """
    Plot basic signal statistics for each class.
    
    Args:
        X: Input data of shape (N, window_length, 1)
        y: Labels of shape (N,)
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    classes = ['Baseline', 'Mental Stress']
    
    # Calculate statistics
    means = np.mean(X, axis=1)[:, 0]
    stds = np.std(X, axis=1)[:, 0]
    peaks = np.max(X, axis=1)[:, 0]
    
    # Plot distributions
    for class_idx in [0, 1]:
        mask = y == class_idx
        sns.kdeplot(means[mask], ax=axes[0], label=classes[class_idx])
        sns.kdeplot(stds[mask], ax=axes[1], label=classes[class_idx])
        sns.kdeplot(peaks[mask], ax=axes[2], label=classes[class_idx])
    
    axes[0].set_title('Mean Distribution')
    axes[1].set_title('Standard Deviation Distribution')
    axes[2].set_title('Peak Amplitude Distribution')
    
    for ax in axes:
        ax.grid(True)
        ax.legend()
    
    plt.tight_layout()
    return fig

fig3 = plot_signal_stats(X, y)
