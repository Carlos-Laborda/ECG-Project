import os
import pandas as pd
import pyedflib
from datetime import datetime

# Set paths
data_path = "../data/raw/Raw ECG project"
metadata_file = "../data/raw/Raw ECG project/TimeStamps_Merged.txt"
output_path = "../data/interim/01_data_processed.pkl"

# Read metadata
metadata = pd.read_csv(metadata_file, delimiter="\t")
metadata["LabelStart"] = pd.to_datetime(metadata["LabelStart"])
metadata["LabelEnd"] = pd.to_datetime(metadata["LabelEnd"])


# Function to load an EDF file
def load_edf(file_path):
    with pyedflib.EdfReader(file_path) as f:
        signals = []
        for i in range(f.signals_in_file):
            signals.append(f.readSignal(i))
        header = f.getHeader()
        signal_headers = f.getSignalHeaders()
    return signals, header, signal_headers


# Process EDF files
data = []
for edf_file in os.listdir(data_path):
    if edf_file.endswith(".edf"):
        # Extract Subject ID from the file name
        file_parts = edf_file.split("_")
        subject_id = file_parts[0]
        condition = file_parts[2]

        # Load ECG data
        edf_path = os.path.join(data_path, edf_file)
        signals, header, signal_headers = load_edf(edf_path)

        # Match metadata for the Subject ID
        subject_metadata = metadata[metadata["Subject_ID"] == int(subject_id)]
        for _, row in subject_metadata.iterrows():
            start_time = row["LabelStart"]
            end_time = row["LabelEnd"]

            # Extract segment of ECG data for the given time range
            # Assuming the sampling rate is available in the header
            sample_rate = signal_headers[0]["sample_rate"]
            start_idx = int(
                (
                    start_time
                    - datetime.strptime(header["startdate"], "%Y-%m-%d %H:%M:%S")
                ).total_seconds()
                * sample_rate
            )
            end_idx = int(
                (
                    end_time
                    - datetime.strptime(header["startdate"], "%Y-%m-%d %H:%M:%S")
                ).total_seconds()
                * sample_rate
            )

            # Append the matched data
            if 0 <= start_idx < len(signals[0]) and 0 <= end_idx <= len(signals[0]):
                segment = signals[0][start_idx:end_idx]
                data.append(
                    {
                        "Subject_ID": subject_id,
                        "Condition": condition,
                        "Category": row["Category"],
                        "Code": row["Code"],
                        "Segment": segment,
                    }
                )

# Convert to DataFrame
df = pd.DataFrame(data)

# Save the processed data for further use
df.to_pickle("processed_ecg_data.pkl")  # Save as a pickle file for efficiency
print("Data processing complete. Saved to 'processed_ecg_data.pkl'.")
