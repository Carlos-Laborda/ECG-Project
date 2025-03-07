import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from utils import load_ecg_data

# ---------------------------------------------------------------------------------------------------------------------
# Adjust plot settings
# ---------------------------------------------------------------------------------------------------------------------
mpl.style.use("ggplot")
mpl.rcParams["figure.figsize"] = (20, 5)
mpl.rcParams["figure.dpi"] = 100

# ---------------------------------------------------------------------------------------------------------------------
# Set the path to the HDF5 file and load data
# ---------------------------------------------------------------------------------------------------------------------
hdf5_path = "../data/interim/ecg_data.h5"

# Load data from HDF5
data = load_ecg_data(hdf5_path)

# ---------------------------------------------------------------------------------------------------------------------
# Plot the first 10000 samples of the ECG data for each category
# ---------------------------------------------------------------------------------------------------------------------
# Plot individual signals
# for (participant, category), signal in data.items():
#     plt.figure()
#     plt.plot(signal[:10000])
#     plt.title(f"Participant: {participant}, Category: {category}")
#     plt.xlabel("Sample Index")
#     plt.ylabel("ECG Signal")
#     plt.show()

# ---------------------------------------------------------------------------------------------------------------------
# Compare categories of ECG data
# ---------------------------------------------------------------------------------------------------------------------
# Aggregate data by category
category_data = {}
for (participant, category), signal in data.items():
    if category not in category_data:
        category_data[category] = []
    category_data[category].append(signal)

# Plot all categories on a single plot for comparison
plt.figure()
current_index = 0
for category, signals in category_data.items():
    concatenated_data = np.concatenate(signals)[:10000]
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
# Aggregate data by participants for each category
participant_data = {}

for (participant, category), signal in data.items():
    if category not in participant_data:
        participant_data[category] = {}
    if participant not in participant_data[category]:
        participant_data[category][participant] = []
    participant_data[category][participant].append(signal)

# Plot each category, comparing participants
for category, participants_signals in participant_data.items():
    plt.figure()
    current_index = 0
    for participant, signals in participants_signals.items():
        concatenated_data = np.concatenate(signals)[:10000]
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
