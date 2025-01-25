import os
import numpy as np


def load_ecg_data(base_path, participant_id=None, category=None):
    """
    Load ECG data from .npy files for a specific participant or category.

    Args:
        base_path (str): Base directory where the .npy files are stored.
        participant_id (str, optional): ID of the participant to load data for.
        category (str, optional): Category of the data to load.

    Returns:
        dict: A dictionary of loaded data. Keys are (participant, category) tuples, and values are numpy arrays.
    """
    data = {}

    # Loop over participants
    for participant_dir in os.listdir(base_path):
        participant_path = os.path.join(base_path, participant_dir)
        if not os.path.isdir(participant_path):
            continue

        # If participant_id is specified, skip others
        if participant_id and participant_id != participant_dir:
            continue

        # Loop over .npy files in the participant directory
        for file in os.listdir(participant_path):
            if file.endswith(".npy"):
                category_name = file[:-4]  # Extract category name from filename
                if category and category != category_name:
                    continue

                # Load the .npy file
                file_path = os.path.join(participant_path, file)
                ecg_signal = np.load(file_path)

                # Store in dictionary
                data[(participant_dir, category_name)] = ecg_signal

    return data
