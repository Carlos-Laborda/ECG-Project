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


class ECGTrainingFlow(FlowSpec):
    """
    Metaflow pipeline that:
      1. Loads raw ECG data & metadata to HDF5 (dataset.py)
      2. Splits data (by participant) into train & test
      3. Extracts features (features.py)
      4. Trains model and evaluates (train.py)
      5. Logs everything to MLflow
    """

    @card
    @step
    def start(self):
        """
        Start of the flow.
        """
        print("Starting the ECG training pipeline...")
        self.next(self.load_data)

    @card
    @step
    def load_data(self):
        """
        Step 1: Run dataset creation code to create data
        using only 10 participants (first 10).
        """
        print("Data loaded and stored in HDF5 by dataset.py")

        # We store the path as an artifact
        self.data = process_ecg_data()
        self.next(self.extract_features)

    @card
    @step
    def extract_features(self):
        """
        Step 2: Extract features from the ECG data.
        """
        # Preprocess features
        self.features_df = preprocess_features(self.data)

        self.next(self.train_model)

    @card
    @step
    def train_model(self):
        """
        Step 3: Train an ML model using the extracted features.
        """
        # Example: Print the first few rows of the features DataFrame
        print(self.features_df.head())

        # Filter the dataset for the desired categories
        self.features_df = self.features_df[
            self.features_df["category"].isin(["high_physical_activity", "baseline"])
        ]

        # Map the target categories to binary labels
        self.features_df["label"] = self.features_df["category"].map(
            {"baseline": 0, "high_physical_activity": 1}
        )

        # Drop non-feature columns
        X = self.features_df[
            [
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
        ]
        y = self.features_df["label"]

        # Split the data by participant
        participants = self.features_df["participant_id"].unique()
        train_participants = participants[:8]  # First 8 participants for training
        test_participants = participants[8:]  # Last 2 participants for testing

        X_train = X[self.features_df["participant_id"].isin(train_participants)]
        y_train = y[self.features_df["participant_id"].isin(train_participants)]
        X_test = X[self.features_df["participant_id"].isin(test_participants)]
        y_test = y[self.features_df["participant_id"].isin(test_participants)]

        # Standardize the feature values (mean=0, std=1)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train the model
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train_scaled, y_train)

        # Evaluate the model
        y_pred = model.predict(X_test_scaled)
        print("Classification Report:")
        print(classification_report(y_test, y_pred))

        # Log the model with MLflow
        # mlflow.sklearn.log_model(model, "random_forest_model")

        self.next(self.end)

    @card
    @step
    def end(self):
        """
        End step: Flow completed.
        """
        print("Flow completed.")


if __name__ == "__main__":
    ECGTrainingFlow()
