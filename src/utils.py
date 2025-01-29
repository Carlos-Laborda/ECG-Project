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
