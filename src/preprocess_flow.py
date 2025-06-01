import os
import mlflow
from metaflow import FlowSpec, step, Parameter, current, project

from common2 import (
    process_ecg_data,
    process_save_cleaned_data,
    normalize_cleaned_data,
    segment_data_into_windows,
)

@project(name="ecg_preprocessing")
class ECGPreprocessFlow(FlowSpec):
    """
    Preprocessing Flow:
    - Segments raw ECG data
    - Cleans and normalizes signals
    - Segments into time windows
    """

    mlflow_tracking_uri = Parameter(
        "mlflow_tracking_uri",
        help="MLflow tracking URI",
        default=os.getenv("MLFLOW_TRACKING_URI", "https://127.0.0.1:5000")
    )

    segmented_data_path = Parameter(
        "segmented_data_path",
        help="Output path for segmented ECG",
        default="../data/interim/ecg_data_segmented.h5"
    )

    cleaned_data_path = Parameter(
        "cleaned_data_path",
        help="Output path for cleaned ECG",
        default="../data/interim/ecg_data_cleaned.h5"
    )

    normalized_data_path = Parameter(
        "normalized_data_path",
        help="Output path for normalized ECG",
        default="../data/interim/ecg_data_normalized.h5"
    )

    window_data_path = Parameter(
        "window_data_path",
        help="Final output: segmented & windowed HDF5",
        default="../data/interim/windowed_data.h5"
    )

    fs = Parameter(
        "fs",
        help="Sampling frequency (Hz)",
        default=1000
    )

    window_size = Parameter(
        "window_size",
        help="Size of each window (seconds)",
        default=10
    )

    step_size = Parameter(
        "step_size",
        help="Stride between windows (seconds)",
        default=5
    )

    @step
    def start(self):
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        mlflow.set_experiment("ECGPreprocessing")

        try:
            run = mlflow.start_run(run_name=current.run_id)
            self.mlflow_run_id = run.info.run_id
        except Exception as e:
            raise RuntimeError(f"MLflow connection failed: {str(e)}")

        print("Starting ECG preprocessing...")
        self.next(self.preprocess)

    @step
    def preprocess(self):
        """Run preprocessing pipeline from raw to windowed HDF5."""
        # Step 1: Segment raw data
        if not os.path.exists(self.segmented_data_path):
            print("Segmenting raw ECG data...")
            process_ecg_data(self.segmented_data_path)
        else:
            print(f"Using existing segmented data: {self.segmented_data_path}")

        # Step 2: Clean ECG signals
        if not os.path.exists(self.cleaned_data_path):
            print("Cleaning ECG data...")
            process_save_cleaned_data(self.segmented_data_path, self.cleaned_data_path)
        else:
            print(f"Using existing cleaned data: {self.cleaned_data_path}")

        # Step 3: Normalize cleaned signals
        if not os.path.exists(self.normalized_data_path):
            print("Normalizing ECG data...")
            normalize_cleaned_data(self.cleaned_data_path, self.normalized_data_path)
        else:
            print(f"Using existing normalized data: {self.normalized_data_path}")

        # Step 4: Segment into windows
        print("Creating windowed ECG data...")
        segment_data_into_windows(
            self.normalized_data_path,
            self.window_data_path,
            fs=self.fs,
            window_size=self.window_size,
            step_size=self.step_size,
        )
        print(f"Windowed data saved to: {self.window_data_path}")

    @step
    def end(self):
        mlflow.end_run() 
        print("ECG preprocessing complete.")

if __name__ == "__main__":
    ECGPreprocessFlow()