import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from scipy.stats import zscore

# ---------------------------------------------------------------------------------------------------------------------
# Adjust plot settings
# ---------------------------------------------------------------------------------------------------------------------
mpl.style.use("fivethirtyeight")
mpl.rcParams["figure.figsize"] = (20, 5)
mpl.rcParams["figure.dpi"] = 100

# ---------------------------------------------------------------------------------------------------------------------
# Set the path where npy files are stored
# ---------------------------------------------------------------------------------------------------------------------
base_path = "../data/interim"

# ---------------------------------------------------------------------------------------------------------------------
# Plotting Outliers
# ---------------------------------------------------------------------------------------------------------------------
# Dictionary to store ECG data by category
category_data = {}

# Loop over all participant directories
for participant_dir in os.listdir(base_path)[:3]:
    participant_path = os.path.join(base_path, participant_dir)
    if os.path.isdir(participant_path):
        print(f"Processing participant: {participant_dir}")

        # Loop over all npy files in the participant directory
        for file in os.listdir(participant_path):
            if file.endswith(".npy"):
                file_path = os.path.join(participant_path, file)
                data = np.load(file_path)

                # Extract category from the file name
                category = file[:-4]

                # Append data to the corresponding category in the dictionary
                if category not in category_data:
                    category_data[category] = []
                category_data[category].append(data)

# Plot histogram of ECG data for each category
for category, data_list in category_data.items():
    concatenated_data = np.concatenate(data_list)
    plt.figure()
    plt.hist(concatenated_data, bins=100)
    plt.title(f"Histogram of ECG Data for Category: {category}")
    plt.xlabel("ECG Signal")
    plt.ylabel("Frequency")
    plt.show()

# Detect outliers using Z-score method
outliers = {}
for category, data_list in category_data.items():
    concatenated_data = np.concatenate(data_list)
    z_scores = zscore(concatenated_data)
    outliers[category] = concatenated_data[np.abs(z_scores) > 3]

# Plot boxplot without outliers
boxplot_data = []
labels = []
for category, data_list in category_data.items():
    concatenated_data = np.concatenate(data_list)
    z_scores = zscore(concatenated_data)
    filtered_data = concatenated_data[np.abs(z_scores) <= 3]
    boxplot_data.append(filtered_data)
    labels.append(category)

plt.figure()
plt.boxplot(boxplot_data, labels=labels)
plt.title("Boxplot of ECG Data by Category (Without Outliers)")
plt.xlabel("Category")
plt.ylabel("ECG Signal")
plt.show()
