import os
import logging
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import GroupKFold
from mlflow.models import infer_signature

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
      2. Extracts features
      3. Performs participant-level 5-fold cross-validation
      4. Joins the folds, aggregates metrics
      5. Registers the final model in MLflow if accuracy >= threshold
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

    accuracy_threshold = Parameter(
        "accuracy_threshold",
        help="Minimum accuracy threshold required to register the model",
        default=0.7,
    )

    @card
    @step
    def start(self):
        """
        Start and prepare the pipeline (create parent MLflow run).
        """
        import mlflow

        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        logging.info("MLflow tracking server: %s", self.mlflow_tracking_uri)

        try:
            run = mlflow.start_run(run_name=current.run_id)
            self.mlflow_run_id = run.info.run_id  # parent run
        except Exception as e:
            raise RuntimeError(
                f"Cannot connect to MLflow at {self.mlflow_tracking_uri}"
            ) from e

        print(
            "Starting the participant-level CV pipeline with potential model registry..."
        )
        self.next(self.load_data)

    @card
    @step
    def load_data(self):
        """
        Step 1: Load or process ECG data into HDF5.
        """
        if os.path.exists(self.hdf5_path):
            print(f"HDF5 file already exists at {self.hdf5_path}, skipping creation.")
        else:
            print(f"Processing raw ECG data -> creating {self.hdf5_path}...")
            process_ecg_data(self.hdf5_path)

        # Load memory
        self.data = load_ecg_data(self.hdf5_path)
        print(f"Loaded {len(self.data)} (participant, category) pairs.")
        self.next(self.extract_features)

    @step
    def extract_features(self):
        """
        Step 2: Extract features from the ECG data.
        """
        if os.path.exists(self.features_path):
            print(f"Features file found at {self.features_path}, loading...")
            self.features_df = pd.read_parquet(self.features_path)
        else:
            print("Extracting features from data...")
            self.features_df = preprocess_features(
                self.data, fs=1000, window_size=10, step_size=1
            )
            self.features_df.to_parquet(self.features_path, index=False)
            print(f"Features saved to {self.features_path}")

        print(f"Feature DataFrame shape: {self.features_df.shape}")

        # Filter categories for baseline vs. high_physical_activity
        df_filtered = self.features_df[
            self.features_df["category"].isin(["high_physical_activity", "baseline"])
        ].copy()
        df_filtered["label"] = df_filtered["category"].map(
            {"baseline": 0, "high_physical_activity": 1}
        )
        df_filtered.reset_index(drop=True, inplace=True)

        self.df_filtered = df_filtered
        self.next(self.prepare_cv, self.train)

    @card
    @step
    def prepare_cv(self):
        """
        Step 3: Prepare participant-level folds via GroupKFold (5 splits).
        """
        from sklearn.model_selection import GroupKFold

        feature_cols = [
            col
            for col in self.df_filtered.columns
            if col not in ["participant_id", "category", "label", "window_index"]
        ]
        self.X = self.df_filtered[feature_cols].values
        self.y = self.df_filtered["label"].values
        self.groups = self.df_filtered["participant_id"].values

        gkf = GroupKFold(n_splits=5)
        self.folds = list(enumerate(gkf.split(self.X, self.y, groups=self.groups)))

        self.next(self.cross_validate_fold, foreach="folds")

    @card
    @step
    def cross_validate_fold(self):
        """
        Step 4 (foreach): Train/evaluate a RandomForest for each participant-level fold.
        """
        fold_index, (train_idx, test_idx) = self.input
        self.fold_index = fold_index

        # Prepare data
        X_train_fold = self.X[train_idx]
        X_test_fold = self.X[test_idx]
        y_train_fold = self.y[train_idx]
        y_test_fold = self.y[test_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_fold)
        X_test_scaled = scaler.transform(X_test_fold)

        # Start nested run for each fold
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        with (
            mlflow.start_run(run_id=self.mlflow_run_id),
            mlflow.start_run(run_name=f"fold-{fold_index}", nested=True) as run,
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
            mlflow.log_metric(f"fold{fold_index}_accuracy", self.fold_accuracy)
            mlflow.log_metric(
                f"fold{fold_index}_f1_high_pa", fold_report["1"]["f1-score"]
            )

        print(f"Fold {fold_index} done. Accuracy={self.fold_accuracy:.3f}")
        self.next(self.join_folds)

    @card
    @step
    def join_folds(self, inputs):
        """
        Step 5: Join folds, average metrics, store final model from first (or best) fold.
        """
        import mlflow

        # Merge artifacts so we can see self.mlflow_run_id from parent
        self.merge_artifacts(inputs, include=["mlflow_run_id"])

        # Collate fold accuracies
        accuracies = [inp.fold_accuracy for inp in inputs]
        self.cv_accuracy_mean = float(np.mean(accuracies))
        self.cv_accuracy_std = float(np.std(accuracies))

        print("\n===== Cross-Validation Results =====")
        print(f"Fold accuracies: {accuracies}")
        print(
            f"Mean CV accuracy: {self.cv_accuracy_mean:.3f} ± {self.cv_accuracy_std:.3f}"
        )

        # Log CV summary metrics
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        mlflow.log_metric(
            "cv_accuracy_mean", self.cv_accuracy_mean, run_id=self.mlflow_run_id
        )
        mlflow.log_metric(
            "cv_accuracy_std", self.cv_accuracy_std, run_id=self.mlflow_run_id
        )

        self.next(self.register)

    @card
    @step
    def train(self):
        """
        Train a model on the entire dataset.
        """
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)

        # # split data into features and labels
        self.X = self.df_filtered.drop(
            columns=["participant_id", "category", "label", "window_index"]
        )
        self.y = self.df_filtered["label"]

        # Log the training process under the current MLflow run
        with mlflow.start_run(run_id=self.mlflow_run_id):
            mlflow.autolog(log_models=False)

            # Train a model on the entire dataset
            self.model = RandomForestClassifier(random_state=42)
            self.model.fit(self.X, self.y)

        # After training, register the model
        self.next(self.register)

    @card
    @step
    def register(self, inputs):
        """Register the model if accuracy >= threshold"""

        mlflow.set_tracking_uri(self.mlflow_tracking_uri)

        self.merge_artifacts(inputs)

        if self.cv_accuracy_mean >= self.accuracy_threshold:
            with mlflow.start_run(run_id=self.mlflow_run_id):
                signature = infer_signature(self.X, self.model.predict(self.X))
                mlflow.sklearn.log_model(
                    sk_model=self.model,
                    artifact_path="model",
                    registered_model_name="Test_RF",
                    signature=signature,
                )
                print("Model successfully registered.")
        else:
            print("Model accuracy below threshold, skipping registration.")
        self.next(self.end)

    @card
    @step
    def end(self):
        """Step 7: Done."""
        print("Participant-Level CV completed.")
        print(
            f"Final CV Accuracy = {self.cv_accuracy_mean:.3f} (Threshold={self.accuracy_threshold})"
        )
        print("Flow ended.")


if __name__ == "__main__":
    ECGTrainingFlow()
