import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

# ---------------------------------------------------------------------------------------------------------------------
# Adjust plot settings
# ---------------------------------------------------------------------------------------------------------------------
mpl.style.use("ggplot")
mpl.rcParams["figure.figsize"] = (20, 5)
mpl.rcParams["figure.dpi"] = 100

# ---------------------------------------------------------------------------------------------------------------------
# Set the path where npy files are stored
# ---------------------------------------------------------------------------------------------------------------------
base_path = "../data/interim"

# ---------------------------------------------------------------------------------------------------------------------
# Plot the first 10000 samples of the ECG data for each category
# ---------------------------------------------------------------------------------------------------------------------
# Loop over all participants
for participant_dir in os.listdir(base_path)[:3]:  # only 1st participant to test
    participant_path = os.path.join(base_path, participant_dir)
    if os.path.isdir(participant_path):
        print(f"Processing participant: {participant_dir}")

        # Loop over all npy files in the participant directory
        for file in os.listdir(participant_path):
            if file.endswith(".npy"):
                file_path = os.path.join(participant_path, file)
                data = np.load(file_path)

                # Plot the first 1000 samples of the data
                plt.figure()
                plt.plot(data[:10000])
                plt.title(f"Participant: {participant_dir}, Category: {file[:-4]}")
                plt.xlabel("Sample Index")
                plt.ylabel("ECG Signal")
                plt.show()

# ---------------------------------------------------------------------------------------------------------------------
# Compare categories of ECG data
# ---------------------------------------------------------------------------------------------------------------------
category_data = {}

# Loop over all participants
for participant_dir in os.listdir(base_path)[:3]:  # only 1st participant to test
    participant_path = os.path.join(base_path, participant_dir)
    if os.path.isdir(participant_path):
        print(f"Processing participant: {participant_dir}")

        # Loop over all npy files in the participant directory
        for file in os.listdir(participant_path):
            if file.endswith(".npy"):
                file_path = os.path.join(participant_path, file)
                data = np.load(file_path)

                # Extract category from filename
                category = file[:-4]

                # Add data to the corresponding category
                if category not in category_data:
                    category_data[category] = []
                category_data[category].append(data)

# Plot all categories on a single plot for comparison
plt.figure()
current_index = 0
for category, data_list in category_data.items():
    # Concatenate all data arrays for the category
    concatenated_data = np.concatenate(data_list)
    concatenated_data = concatenated_data[:10000]
    # Plot the first 10000 samples of the concatenated data
    plt.plot(
        range(current_index, current_index + len(concatenated_data)),
        concatenated_data,
        label=category,
    )
    current_index += len(concatenated_data)

plt.title("Comparison of ECG Data Categories")
plt.xlabel("Sample Index")
plt.ylabel("ECG Signal")
plt.legend()
plt.show()

# ---------------------------------------------------------------------------------------------------------------------
# Compare participants
# ---------------------------------------------------------------------------------------------------------------------
category_participant_data = {}

# Loop over all participants
for participant_dir in os.listdir(base_path):
    participant_path = os.path.join(base_path, participant_dir)
    if os.path.isdir(participant_path):
        print(f"Processing participant: {participant_dir}")

        # Loop over all npy files in the participant directory
        for file in os.listdir(participant_path):
            if file.endswith(".npy"):
                file_path = os.path.join(participant_path, file)
                data = np.load(file_path)

                # Extract category from filename
                category = file[:-4]

                # Append data to the corresponding category and participant in the dictionary
                if category not in category_participant_data:
                    category_participant_data[category] = {}
                if participant_dir not in category_participant_data[category]:
                    category_participant_data[category][participant_dir] = []
                category_participant_data[category][participant_dir].append(data)

# Plot each category, comparing participants
for category, participants_data in category_participant_data.items():
    plt.figure()
    current_index = 0
    for participant, data_list in participants_data.items():
        # Concatenate all data arrays for the participant
        concatenated_data = np.concatenate(data_list)
        concatenated_data = concatenated_data[:10000]
        # Plot the first 10000 samples of the concatenated data
        plt.plot(
            range(current_index, current_index + len(concatenated_data)),
            concatenated_data,
            label=participant,
        )
        current_index += len(concatenated_data)

    plt.title(f"Comparison of ECG Data for Category: {category}")
    plt.xlabel("Sample Index")
    plt.ylabel("ECG Signal")
    plt.legend()
    plt.show()
