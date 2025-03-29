import h5py
import numpy as np
import matplotlib.pyplot as plt

def inspect_known_stitching_point(segments_data_path, participant_id="30100", category="baseline", stitch_point=180000, window=5000):
    """
    Inspect a known stitching point in the ECG segments where different conditions are combined.
    
    Args:
        segments_data_path: Path to the segmented data file
        participant_id: ID of the participant to inspect
        category: Category to inspect (e.g., 'baseline')
        stitch_point: Sample index where stitching is expected to occur
        window: Number of samples to show before and after the stitch point
    """
    print(f"\n=== Inspecting Known Stitching Point for {participant_id}, {category} at {stitch_point} ===")
    
    with h5py.File(segments_data_path, 'r') as f:
        if participant_id not in f:
            print(f"Participant {participant_id} not found in the dataset.")
            return
            
        participant = f[participant_id]
        
        if category not in participant:
            print(f"Category {category} not found for participant {participant_id}.")
            return
            
        segment = participant[category][:]
        
        if stitch_point >= len(segment):
            print(f"Stitch point {stitch_point} is beyond the signal length {len(segment)}.")
            return
        
        # Determine window boundaries
        start = max(0, stitch_point - window)
        end = min(len(segment), stitch_point + window)
        
        # Plot a wider view first
        plt.figure(figsize=(15, 8))
        
        # Wide view (the full window)
        plt.subplot(311)
        plt.plot(range(start, end), segment[start:end])
        plt.axvline(x=stitch_point, color='r', linestyle='--', label="Expected Stitch Point")
        plt.title(f"{participant_id} - {category} - Wide View of Stitching Area")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.grid(True)
        
        # Zoomed in to 1 second before and after the stitch
        zoom_start = max(0, stitch_point - 1000)
        zoom_end = min(len(segment), stitch_point + 1000)
        
        plt.subplot(312)
        plt.plot(range(zoom_start, zoom_end), segment[zoom_start:zoom_end])
        plt.axvline(x=stitch_point, color='r', linestyle='--')
        plt.title("Zoomed View (1 second before and after)")
        plt.ylabel("Amplitude")
        plt.grid(True)
        
        # Ultra-zoomed to see the exact transition (100ms)
        ultra_zoom_start = max(0, stitch_point - 100)
        ultra_zoom_end = min(len(segment), stitch_point + 100)
        
        plt.subplot(313)
        plt.plot(range(ultra_zoom_start, ultra_zoom_end), segment[ultra_zoom_start:ultra_zoom_end], marker='o', markersize=3)
        plt.axvline(x=stitch_point, color='r', linestyle='--')
        plt.title("Ultra-Zoomed View (100ms before and after)")
        plt.ylabel("Amplitude")
        plt.xlabel("Sample Index")
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # Also check signal statistics before and after
        before = segment[max(0, stitch_point - 5000):stitch_point]
        after = segment[stitch_point:min(len(segment), stitch_point + 5000)]
        
        print("\nSignal Statistics Before and After Stitching Point:")
        print(f"Before - Mean: {np.mean(before):.6f}, Std: {np.std(before):.6f}, Range: [{np.min(before):.6f}, {np.max(before):.6f}]")
        print(f"After  - Mean: {np.mean(after):.6f}, Std: {np.std(after):.6f}, Range: [{np.min(after):.6f}, {np.max(after):.6f}]")

if __name__ == "__main__":
    segments_data_path = "../data/interim/ecg_data_segmented.h5"
    
    # Inspect the known stitching point
    inspect_known_stitching_point(
        segments_data_path, 
        participant_id='participant_30100',
        category="baseline", 
        stitch_point=180000,
    )