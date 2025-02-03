import os
import logging
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Metaflow imports
from metaflow import FlowSpec, step, card, Parameter, current, project

# Local imports
from common import process_ecg_data, preprocess_features
from utils import load_ecg_data


@project(name="ecg_training")
class ECGTrainingFlow(FlowSpec):
    """
    Metaflow pipeline that:
      1. Loads raw ECG data & metadata to HDF5
      2. Splits data (by participant) into train & test
      3. Extracts features
      4. Trains model and evaluates
      5. Logs to MLflow
    """

    mlflow_tracking_uri = Parameter(
        "mlflow_tracking_uri",
        help="Location of the MLflow tracking server",
        default=os.getenv("MLFLOW_TRACKING_URI", "https://127.0.0.1:5000"),
    )

    hdf5_path = Parameter(
        "hdf5_path",
        help="Path to the HDF5 file containing ECG data",
        default="../data/interim/ecg_data.h5",
    )

    features_path = Parameter(
        "features_path",
        help="Path to the extracted features file",
        default="../data/processed/features.parquet",
    )

    num_train_participants = Parameter(
        "num_train_participants",
        help="Number of participants to use for training",
        default=8,
    )

    num_test_participants = Parameter(
        "num_test_participants",
        help="Number of participants to use for testing",
        default=2,
    )

    accuracy_threshold = Parameter(
        "accuracy_threshold",
        help="Minimum accuracy threshold required to register the model",
        default=0.8,
    )

    @card
    @step
    def start(self):
        """Start and prepare the Training pipeline."""
        # Set MLflow tracking URI
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        logging.info("MLflow tracking server: %s", self.mlflow_tracking_uri)

        try:
            # Start an MLflow run with the current Metaflow run ID
            run = mlflow.start_run(run_name=current.run_id)
            self.mlflow_run_id = run.info.run_id
        except Exception as e:
            message = f"Failed to connect to MLflow server {self.mlflow_tracking_uri}."
            raise RuntimeError(message) from e
        print("Starting the ECG training pipeline...")
        self.next(self.load_data)

    @card
    @step
    def load_data(self):
        """
        Step 1: Load or process ECG data into HDF5.
        """
        # self.hdf5_path = "../data/interim/ecg_data.h5"
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
        # features_path = "../data/processed/features.parquet"
        if os.path.exists(self.features_path):
            print(f"Features file found at {self.features_path}, loading...")
            self.features_df = pd.read_parquet(self.features_path)
        else:
            print("Extracting features (sliding window, etc.)...")
            self.features_df = preprocess_features(
                self.data, fs=1000, window_size=10, step_size=1
            )
            self.features_df.to_parquet(self.features_path, index=False)
            print(f"Features saved to {self.features_path}")

        print(f"Feature DataFrame shape: {self.features_df.shape}")
        self.next(self.train_model)

    @card
    @step
    def train_model(self):
        """
        Step 3: Train an ML model using the extracted features.
        """
        logging.info("Training model...")
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)

        # Show sample
        print("Sample of features:")
        print(self.features_df.head())

        # Filter for the two categories of interest and create binary labels.
        df_filtered = self.features_df[
            self.features_df["category"].isin(["high_physical_activity", "baseline"])
        ].copy()
        df_filtered["label"] = df_filtered["category"].map(
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
        X = df_filtered[feature_cols]
        y = df_filtered["label"]

        # Split data by participant.
        participants = np.sort(df_filtered["participant_id"].unique())
        train_participants = participants[: self.num_train_participants]
        test_participants = participants[
            self.num_train_participants : self.num_train_participants
            + self.num_test_participants
        ]

        mlflow.log_param("train_participants", list(train_participants))
        mlflow.log_param("test_participants", list(test_participants))

        train_mask = df_filtered["participant_id"].isin(train_participants)
        test_mask = df_filtered["participant_id"].isin(test_participants)
        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        # Standardize features.
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Use a nested MLflow run to log training details.
        with mlflow.start_run(run_id=self.mlflow_run_id, nested=True):
            mlflow.autolog(log_models=False)
            model = RandomForestClassifier(random_state=42)
            model.fit(X_train_scaled, y_train)

            y_pred = model.predict(X_test_scaled)
            report = classification_report(y_test, y_pred, output_dict=True)
            accuracy = accuracy_score(y_test, y_pred)
            print("Classification Report:")
            print(classification_report(y_test, y_pred))
            print("Confusion Matrix:")
            print(confusion_matrix(y_test, y_pred))
            print(f"Accuracy: {accuracy * 100:.2f}%")

            # Log key metrics.
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("f1_high_physical_activity", report["1"]["f1-score"])
            # Log the model artifact.
            mlflow.sklearn.log_model(
                model,
                registered_model_name="baseline_RF",
                artifact_path="random_forest_model",
            )

        # Optional: store artifacts for next steps
        self.model = model
        self.scaler = scaler
        self.accuracy = accuracy

        self.next(self.end)

    @card
    @step
    def end(self):
        print("Flow completed successfully!")


if __name__ == "__main__":
    ECGTrainingFlow()
