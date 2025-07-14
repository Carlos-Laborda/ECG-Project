import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

def inspect_hdf5(filepath, is_windowed=False):
    print(f"\n Inspecting: {filepath}")
    if not os.path.exists(filepath):
        print("  File not found")
        return

    with h5py.File(filepath, "r") as f:
        participants = list(f.keys())
        print(f"Participants: {len(participants)}")
        
        total_segments = 0
        total_windows = 0
        segment_lengths = []

        for pid in participants:
            group = f[pid]
            segments = list(group.keys())
            total_segments += len(segments)

            print(f"  - {pid}: {len(segments)} segments")

            for seg in segments:
                data = group[seg][...]
                if is_windowed:
                    # Expect shape (n_windows, window_len)
                    total_windows += data.shape[0]
                    segment_lengths.append(data.shape[1])  # window size
                else:
                    segment_lengths.append(len(data))  

        print(f"Total segments: {total_segments}")
        if is_windowed:
            print(f"Total windows: {total_windows}")
        
        segment_lengths = np.array(segment_lengths)
        print(f"Segment lengths (samples): min={segment_lengths.min()}, "
              f"max={segment_lengths.max()}, mean={segment_lengths.mean():.2f}")

# Paths (adjust if needed)
ROOT = "../data/interim"
inspect_hdf5(os.path.join(ROOT, "ppg_raw.h5"))
inspect_hdf5(os.path.join(ROOT, "ppg_clean.h5"))
inspect_hdf5(os.path.join(ROOT, "ppg_norm.h5"))
inspect_hdf5(os.path.join(ROOT, "ppg_windows.h5"), is_windowed=True)
