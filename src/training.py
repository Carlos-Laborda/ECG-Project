import os
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Metaflow imports
from metaflow import FlowSpec, step, card

# Local imports
from common import process_ecg_data, preprocess_features
from utils import load_ecg_data


class ECGTrainingFlow(FlowSpec):
    """
    Metaflow pipeline that:
      1. Loads raw ECG data & metadata to HDF5
      2. Splits data (by participant) into train & test
      3. Extracts features
      4. Trains model and evaluates
      5. (Optionally) logs to MLflow
    """

    @card
    @step
    def start(self):
        print("Starting the ECG training pipeline...")
        self.next(self.load_data)

    @card
    @step
    def load_data(self):
        """
        Step 1: Load or process ECG data into HDF5.
        """
        self.hdf5_path = "../data/interim/ecg_data.h5"
        if os.path.exists(self.hdf5_path):
            print(f"HDF5 file already exists at {self.hdf5_path} - skipping creation.")
        else:
            print(f"Processing raw ECG data -> creating {self.hdf5_path}...")
            process_ecg_data(self.hdf5_path)

        # Load into memory
        self.data = load_ecg_data(self.hdf5_path)
        print(f"Loaded {len(self.data)} (participant, category) pairs.")
        self.next(self.extract_features)

    @card
    @step
    def extract_features(self):
        """
        Step 2: Extract features from the ECG data.
        """
        features_path = "../data/processed/features.parquet"
        if os.path.exists(features_path):
            print(f"Features file found at {features_path}, loading...")
            self.features_df = pd.read_parquet(features_path)
        else:
            print("Extracting features (sliding window, etc.)...")
            self.features_df = preprocess_features(
                self.data, fs=1000, window_size=10, step_size=1
            )
            self.features_df.to_parquet(features_path, index=False)
            print(f"Features saved to {features_path}")

        print(f"Feature DataFrame shape: {self.features_df.shape}")
        self.next(self.train_model)

    @card
    @step
    def train_model(self):
        """
        Step 3: Train an ML model using the extracted features.
        """
        # Show sample
        print("Sample of features:")
        print(self.features_df.head())

        # Filter categories
        self.features_df = self.features_df[
            self.features_df["category"].isin(["high_physical_activity", "baseline"])
        ]
        # Binarize label
        self.features_df["label"] = self.features_df["category"].map(
            {"baseline": 0, "high_physical_activity": 1}
        )

        feature_cols = [
            "mean",
            "std",
            "min",
            "max",
            "rms",
            "iqr",
            "psd_mean",
            "psd_max",
            "dominant_freq",
            "shannon_entropy",
            "sample_entropy",
        ]
        X = self.features_df[feature_cols]
        y = self.features_df["label"]

        # Split by participant
        participants = self.features_df["participant_id"].unique()
        train_participants = participants[:8]
        test_participants = participants[8:]

        train_mask = self.features_df["participant_id"].isin(train_participants)
        test_mask = self.features_df["participant_id"].isin(test_participants)

        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train_scaled, y_train)

        # Evaluate
        y_pred = model.predict(X_test_scaled)
        print("Classification Report:")
        print(classification_report(y_test, y_pred))

        # Optional: store artifacts for next steps
        self.model = model
        self.scaler = scaler
        self.accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy = {self.accuracy * 100:.2f}%")

        self.next(self.end)

    @card
    @step
    def end(self):
        print("Flow completed successfully!")


if __name__ == "__main__":
    ECGTrainingFlow()
