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