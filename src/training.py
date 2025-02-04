import os
import logging
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import GroupKFold


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
        default=16,
    )

    num_test_participants = Parameter(
        "num_test_participants",
        help="Number of participants to use for testing",
        default=4,
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

        # Filter categories: only "baseline" vs. "high_physical_activity"
        df_filtered = self.features_df[
            self.features_df["category"].isin(["high_physical_activity", "baseline"])
        ].copy()
        df_filtered["label"] = df_filtered["category"].map(
            {"baseline": 0, "high_physical_activity": 1}
        )
        df_filtered.reset_index(drop=True, inplace=True)

        # Store this for cross-validation
        self.df_filtered = df_filtered
        self.next(self.prepare_cv)

    @card
    @step
    def prepare_cv(self):
        """
        Step 4: Prepare participant-level folds using GroupKFold with 5 splits.
        """
        # The full feature set (excluding participant/category info).
        feature_cols = [
            col
            for col in self.df_filtered.columns
            if col not in ["participant_id", "category", "label", "window_index"]
        ]

        self.X = self.df_filtered[feature_cols].values
        self.y = self.df_filtered["label"].values

        # Use participant_id as the grouping variable
        self.groups = self.df_filtered["participant_id"].values

        group_kf = GroupKFold(n_splits=5)
        self.folds = list(enumerate(group_kf.split(self.X, self.y, groups=self.groups)))

        # Use foreach to spawn one branch per fold
        self.next(self.cross_validate_fold, foreach="folds")

    @card
    @step
    def cross_validate_fold(self):
        """
        Step 5 (foreach): Train/evaluate a RandomForest for each participant-level fold.
        """
        fold_index, (train_idx, test_idx) = self.input
        self.fold_index = fold_index

        X_train_fold = self.X[train_idx]
        X_test_fold = self.X[test_idx]
        y_train_fold = self.y[train_idx]
        y_test_fold = self.y[test_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_fold)
        X_test_scaled = scaler.transform(X_test_fold)

        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        # Log everything under the parent run, create a nested run for each fold
        # with mlflow.start_run(run_id=self.mlflow_run_id, nested=True):
        with (
            mlflow.start_run(run_id=self.mlflow_run_id),
            mlflow.start_run(
                run_name=f"cross-validation-fold-{self.fold_index}",
                nested=True,
            ) as run,
        ):
            self.mlflow_fold_run_id = run.info.run_id
            mlflow.autolog(log_models=False)
            model = RandomForestClassifier(random_state=42)
            model.fit(X_train_scaled, y_train_fold)

            y_pred_fold = model.predict(X_test_scaled)
            self.fold_accuracy = accuracy_score(y_test_fold, y_pred_fold)
            fold_report = classification_report(
                y_test_fold, y_pred_fold, output_dict=True
            )

            # Log fold-specific metrics
            mlflow.log_metric(f"fold{fold_index}_accuracy", self.fold_accuracy)
            mlflow.log_metric(
                f"fold{fold_index}_f1_high_pa", fold_report["1"]["f1-score"]
            )

        print(f"Fold {fold_index} completed. Accuracy = {self.fold_accuracy:.3f}")
        self.next(self.join_folds)

    @card
    @step
    def join_folds(self, inputs):
        """
        Step 6: Join the fold branches and average the metrics.
        """
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)

        self.merge_artifacts(inputs, include=["mlflow_run_id"])

        # Gather fold accuracies
        accuracies = [inp.fold_accuracy for inp in inputs]
        self.cv_accuracy_mean = float(np.mean(accuracies))
        self.cv_accuracy_std = float(np.std(accuracies))

        print("\n=== Cross-Validation Results (Participant-Level, 5-Fold) ===")
        print(f"Fold Accuracies: {accuracies}")
        print(
            f"Mean Accuracy: {self.cv_accuracy_mean:.3f} ± {self.cv_accuracy_std:.3f}"
        )

        # Log overall CV metrics
        mlflow.log_metrics(
            {
                "cv_accuracy_mean": self.cv_accuracy_mean,
                "cv_accuracy_std": self.cv_accuracy_std,
            },
            run_id=self.mlflow_run_id,
        )

        # End the flow
        self.next(self.end)

    @card
    @step
    def end(self):
        """
        Step 7: Flow completed successfully, no final training step is performed.
        """
        print("Participant-Level Cross-Validation completed successfully!")
        print(f"Final CV Accuracy = {self.cv_accuracy_mean:.3f}")
        print("Flow ended.")


if __name__ == "__main__":
    ECGTrainingFlow()
