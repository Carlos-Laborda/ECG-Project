import h5py
import numpy as np
from utils import load_ecg_data

# File paths
segmented_data_path = "../data/interim/ecg_data_segmented.h5"
cleaned_data_path = "../data/interim/ecg_data_cleaned.h5"
window_data_path = "../data/interim/windowed_data.h5"

def print_data_info(data, name):
    """Helper function to print data information"""
    print(f"\n=== {name} ===")
    print(f"Shape: {data.shape}")
    print(f"Type: {data.dtype}")
    print(f"Range: [{np.min(data):.3f}, {np.max(data):.3f}]")
    print(f"Mean: {np.mean(data):.3f}")
    print(f"Std: {np.std(data):.3f}")

def check_preprocessing_steps():
    """Check data types and properties at each preprocessing step"""
    
    # 1. Check segmented data
    print("\n🔍 Checking Segmented Data...")
    with h5py.File(segmented_data_path, 'r') as f:
        print("\nAvailable keys:", list(f.keys()))
        for participant in f.keys():
            print(f"\nParticipant: {participant}")
            # Access participant group
            participant_group = f[participant]
            print("Available datasets:", list(participant_group.keys()))
            
            # Access each dataset in the participant group
            for dataset_name in participant_group.keys():
                data = participant_group[dataset_name][:]
                print(f"\nDataset: {dataset_name}")
                print_data_info(data, f"Segmented Data - {participant}")
    
    # 2. Check cleaned data
    print("\n🔍 Checking Cleaned Data...")
    with h5py.File(cleaned_data_path, 'r') as f:
        print("\nAvailable keys:", list(f.keys()))
        for participant in f.keys():
            print(f"\nParticipant: {participant}")
            participant_group = f[participant]
            print("Available datasets:", list(participant_group.keys()))
            
            for dataset_name in participant_group.keys():
                data = participant_group[dataset_name][:]
                print(f"\nDataset: {dataset_name}")
                print_data_info(data, f"Cleaned Data - {participant}")
    
    # 3. Check windowed data
    print("\n🔍 Checking Windowed Data...")
    with h5py.File(window_data_path, 'r') as f:
        print("\nAvailable keys:", list(f.keys()))
        
        # Check first participant as sample
        sample_participant = list(f.keys())[0]
        print(f"\nDetailed check of {sample_participant}:")
        
        participant_group = f[sample_participant]
        categories = list(participant_group.keys())
        print(f"Categories: {categories}")
        
        for category in categories:
            windows = participant_group[category][:]
            print(f"\nCategory: {category}")
            print_data_info(windows, f"Windowed Data - {category}")
            print(f"Number of windows: {windows.shape[0]}")
            print(f"Window length: {windows.shape[1]}")
        
        # Print summary statistics
        total_windows = 0
        windows_per_category = {}
        
        for participant in f.keys():
            for category in f[participant].keys():
                n_windows = f[participant][category].shape[0]
                total_windows += n_windows
                windows_per_category[category] = windows_per_category.get(category, 0) + n_windows
        
        print("\n=== Summary Statistics ===")
        print(f"Total participants: {len(list(f.keys()))}")
        print(f"Total windows: {total_windows}")
        print("\nWindows per category:")
        for category, count in windows_per_category.items():
            print(f"{category}: {count} windows ({count/total_windows*100:.1f}%)")

if __name__ == "__main__":
    try:
        check_preprocessing_steps()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure all preprocessing steps have been completed.")
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise