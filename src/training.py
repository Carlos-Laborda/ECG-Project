import os
import logging
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import GroupKFold
from mlflow.models import infer_signature

# Metaflow imports
from metaflow import FlowSpec, step, card, Parameter, current, project, environment

# Local imports
from common import (
    process_ecg_data,
    preprocess_features,
    process_save_cleaned_data,
    # segment_data_into_windows,
    baseline_1DCNN,
)
from utils import load_ecg_data, prepare_cnn_data


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

    segmented_data_path = Parameter(
        "segmented_data_path",
        help="Path to the HDF5 file containing ECG data segmented",
        default="../data/interim/ecg_data_segmented.h5",
    )

    cleaned_data_path = Parameter(
        "hdf5_path",
        help="Path to the HDF5 file containing ECG data cleaned",
        default="../data/interim/ecg_data_cleaned.h5",
    )

    window_data_path = Parameter(
        "window_data_path",
        help="Path to the windowed data",
        default="../data/interim/windowed_data.h5",
    )

    accuracy_threshold = Parameter(
        "accuracy_threshold",
        help="Minimum accuracy threshold required to register the model",
        default=0.7,
    )

    num_epochs = Parameter(
        "num_epochs",
        help="Number of training epochs for each fold / final model training.",
        default=5,
    )

    batch_size = Parameter(
        "batch_size",
        help="Batch size for CNN training.",
        default=32,
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
        if os.path.exists(self.segmented_data_path):
            print(
                f"HDF5 file already exists at {self.segmented_data_path}, skipping creation."
            )
        else:
            print(f"Processing raw ECG data -> creating {self.segmented_data_path}...")
            process_ecg_data(self.segmented_data_path)

        # Load memory
        #self.data = load_ecg_data(self.segmented_data_path)
        #print(f"Loaded {len(self.data)} (participant, category) pairs.")
        self.next(self.clean_data)

    @step
    def clean_data(self):
        """
        Clean the ECG segmented data.
        """
        if os.path.exists(self.cleaned_data_path):
            print(
                f"HDF5 cleaned data file already exists at {self.cleaned_data_path}, skipping creation."
            )
        else:
            print(
                f"Cleaning segmented ECG data -> creating {self.cleaned_data_path}..."
            )
            process_save_cleaned_data(self.segmented_data_path, self.cleaned_data_path)

        # Load memory
        #self.data = load_ecg_data(self.cleaned_data_path)
        #print(f"Loaded {len(self.data)} (participant, category) pairs.")
        self.next(self.segment_data_windows)

    @card
    @step
    def segment_data_windows(self):
        """
        Segment the ECG data into windows.
        """
        from common import segment_data_into_windows

        if os.path.exists(self.window_data_path):
            print(f"Windowed data file found at {self.window_data_path}, loading...")
        else:
            print("Segmenting ECG data into windows...")
            #self.data = load_ecg_data(self.cleaned_data_path)
            #self.fs = 1000
            #self.window_size = 10
            #self.step_size = 1
            segment_data_into_windows(
                self.data, self.window_data_path, fs=1000, window_size=10, step_size=1
            )
            print(f"Windowed data saved to {self.window_data_path}")

        #self.data = load_ecg_data(self.window_data_path)
        #print(f"Loaded {len(self.data)} (participant, category) pairs.")
        self.next(self.prepare_data_for_cnn)

    @card
    @step
    def prepare_data_for_cnn(self):
        """
        Step 5: Load the final windowed data from HDF5, build X, y, groups for the 1D CNN.
        We'll use the new utility function: prepare_cnn_data.
        """
        self.X, self.y, self.groups = prepare_cnn_data(
            hdf5_path=self.window_data_path,
            label_map={"baseline": 0, "high_physical_activity": 1},
        )
        print(
            f"Windowed data loaded: X shape={self.X.shape}, y shape={self.y.shape}, groups len={len(self.groups)}"
        )

        # We'll do participant-level cross validation with these arrays
        self.num_classes = 2
        self.next(self.prepare_cv, self.train_final)

    @card
    @step
    def prepare_cv(self):
        """
        Step 3: Prepare participant-level folds via GroupKFold (5 splits).
        """
        from sklearn.model_selection import GroupKFold

        gkf = GroupKFold(n_splits=5)
        self.folds = list(enumerate(gkf.split(self.X, self.y, groups=self.groups)))
        self.next(self.cross_validate_fold, foreach="folds")

    @card
    @environment(
        vars={
            "KERAS_BACKEND": os.getenv("KERAS_BACKEND", "jax"),
        },
    )
    @step
    def cross_validate_fold(self):
        """
        Step 4 (foreach): Train/evaluate a RandomForest for each participant-level fold.
        """
        import mlflow

        fold_index, (train_idx, test_idx) = self.input
        self.fold_index = fold_index
        X_train_fold = self.X[train_idx]
        y_train_fold = self.y[train_idx]
        X_test_fold = self.X[test_idx]
        y_test_fold = self.y[test_idx]

        mlflow.set_tracking_uri(self.mlflow_tracking_uri)

        # Nested MLflow run for each fold
        with (
            mlflow.start_run(run_id=self.mlflow_run_id),
            mlflow.start_run(run_name=f"fold-{fold_index}", nested=True) as run,
        ):
            self.mlflow_fold_run_id = run.info.run_id

            mlflow.autolog(log_models=False)
            model = baseline_1DCNN(
                input_shape=(X_train_fold.shape[1], 1))

            model.fit(
                X_train_fold,
                y_train_fold,
                validation_data=(X_test_fold, y_test_fold),
                epochs=self.num_epochs,
                batch_size=self.batch_size,
                verbose=0,
            )

            _, acc_test = model.evaluate(X_test_fold, y_test_fold, verbose=0)
            self.fold_accuracy = float(acc_test)

            mlflow.log_metric(f"fold{fold_index}_accuracy", self.fold_accuracy)

        print(f"Fold {fold_index} -> accuracy={self.fold_accuracy:.3f}")
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
    @environment(
        vars={
            "KERAS_BACKEND": os.getenv("KERAS_BACKEND", "jax"),
        },
    )
    @step
    def train_final(self):
        """
        Train a model on the entire dataset.
        """

        mlflow.set_tracking_uri(self.mlflow_tracking_uri)

        # Log the training process under the current MLflow run
        with mlflow.start_run(run_id=self.mlflow_run_id):
            mlflow.autolog(log_models=False)

            # Train a model on the entire dataset
            final_model = baseline_1DCNN(
                input_shape=(self.X.shape[1], 1))
            final_model.fit(
                self.X,
                self.y,
                epochs=self.num_epochs,
                batch_size=self.batch_size,
                verbose=0,
            )

            self.model = final_model

        # After training, register the model
        self.next(self.register)

    @environment(
        vars={
            "KERAS_BACKEND": os.getenv("KERAS_BACKEND", "jax"),
        },
    )
    @step
    def register(self, inputs):
        """Register the model if accuracy >= threshold"""

        mlflow.set_tracking_uri(self.mlflow_tracking_uri)

        self.merge_artifacts(inputs)

        if self.cv_accuracy_mean >= self.accuracy_threshold:
            with mlflow.start_run(run_id=self.mlflow_run_id):
                sample_input = self.X[:5]
                sample_output = self.model.predict(sample_input)
                signature = infer_signature(sample_input, sample_output)

                mlflow.keras.log_model(
                    self.model,
                    artifact_path="model",
                    registered_model_name="baseline_1DCNN",
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
