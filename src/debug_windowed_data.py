import h5py
import numpy as np
import matplotlib.pyplot as plt
from utils import prepare_cnn_data

def validate_windowed_data(window_data_path):
    """Validate the windowed data structure and labels"""
    
    print("\n=== Validating Windowed Data Structure ===")
    with h5py.File(window_data_path, 'r') as f:
        # 1. Check overall structure
        print("\nHDF5 Structure:")
        print("Available participants:", list(f.keys()))
        
        # Take first two participants for detailed inspection
        participants = list(f.keys())[:2]
        
        for participant_id in participants:
            print(f"\nInspecting Participant: {participant_id}")
            participant = f[participant_id]
            
            # Check categories
            print("Available categories:", list(participant.keys()))
            
            # Print details for each category
            for category in participant.keys():
                windows = participant[category][:]
                print(f"\n{category}:")
                print(f"Shape: {windows.shape}")
                print(f"Range: [{np.min(windows):.6f}, {np.max(windows):.6f}]")
                print(f"Mean: {np.mean(windows):.6f}")
                print(f"Std: {np.std(windows):.6f}")
                
                # Plot random window from each category
                plt.figure(figsize=(15, 3))
                random_idx = np.random.randint(0, windows.shape[0])
                plt.plot(windows[random_idx])
                plt.title(f"{participant_id} - {category} - Window {random_idx}")
                plt.show()

def compare_categories(window_data_path):
    """Compare baseline vs stress windows for same participant"""
    
    with h5py.File(window_data_path, 'r') as f:
        participant_id = list(f.keys())[1]  # Take first participant
        participant = f[participant_id]
        
        # Get one random window from each category
        baseline_windows = participant['baseline'][:]
        stress_windows = participant['mental_stress'][:]
        
        # Random indices
        b_idx = np.random.randint(0, baseline_windows.shape[0])
        s_idx = np.random.randint(0, stress_windows.shape[0])
        
        # Plot comparison
        plt.figure(figsize=(15, 6))
        
        plt.subplot(211)
        plt.plot(baseline_windows[b_idx])
        plt.title(f"{participant_id} - Baseline Window {b_idx}")
        plt.ylabel("Amplitude")
        
        plt.subplot(212)
        plt.plot(stress_windows[s_idx])
        plt.title(f"{participant_id} - Mental Stress Window {s_idx}")
        plt.ylabel("Amplitude")
        plt.xlabel("Sample")
        
        plt.tight_layout()
        plt.show()

def validate_prepare_cnn_data(window_data_path):
    """Validate the data preparation function"""
    
    print("\n=== Validating prepare_cnn_data Output ===")
    X, y, groups = prepare_cnn_data(
        hdf5_path=window_data_path,
        label_map={"baseline": 0, "mental_stress": 1}
    )
    
    print("\nShape Information:")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"groups shape: {groups.shape}")
    
    # Check label distribution per participant
    unique_participants = np.unique(groups)
    print("\nPer-Participant Label Distribution:")
    
    for participant in unique_participants:
        mask = groups == participant
        participant_y = y[mask]
        n_baseline = np.sum(participant_y == 0)
        n_stress = np.sum(participant_y == 1)
        total = len(participant_y)
        
        print(f"\nParticipant {participant}:")
        print(f"Baseline: {n_baseline} ({n_baseline/total*100:.1f}%)")
        print(f"Mental Stress: {n_stress} ({n_stress/total*100:.1f}%)")

if __name__ == "__main__":
    window_data_path = "../data/interim/windowed_data.h5"
    
    # Run all validations
    validate_windowed_data(window_data_path)
    compare_categories(window_data_path)
    validate_prepare_cnn_data(window_data_path)