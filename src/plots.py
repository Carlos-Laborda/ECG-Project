import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

# Adjust plot settings
mpl.style.use("seaborn-v0_8-deep")
mpl.rcParams["figure.figsize"] = (20, 5)
mpl.rcParams["figure.dpi"] = 100

# Set the path where npy files are stored
base_path = "../data/interim"

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

                # Plot the first 1000 samples of the data
                plt.figure()
                plt.plot(data[:10000])
                plt.title(f"Participant: {participant_dir}, Category: {file[:-4]}")
                plt.xlabel("Sample Index")
                plt.ylabel("ECG Signal")
                plt.show()
