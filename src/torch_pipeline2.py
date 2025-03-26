import os
import logging
import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sktime.classification.kernel_based import RocketClassifier
from sktime.datatypes._panel._convert import from_2d_array_to_nested
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
from metaflow import (FlowSpec, step, card, Parameter, current, 
                      project, resources, conda_base, environment)

from common import (
    process_ecg_data,
    process_save_cleaned_data,
)

from torch_utilities import (
    load_processed_data, split_data_by_participant, ECGDataset,
    train, test, EarlyStopping, log_model_summary, prepare_model_signature,
    set_seed, Simple1DCNN, Simple1DCNN_v2, Improved1DCNN, Improved1DCNN_v2
)

from utils import load_ecg_data

@project(name="ecg_training_simple")
class ECGSimpleTrainingFlow(FlowSpec):
    """
    Simplified Metaflow pipeline that:
      1. Loads raw ECG data & metadata to HDF5
      2. Cleans the data
      3. Segments into windows
      4. Trains a model on train/test split
      5. Evaluates and registers if performance is good
    """

    mlflow_tracking_uri = Parameter(
        "mlflow_tracking_uri",
        help="MLflow tracking server location",
        default=os.getenv("MLFLOW_TRACKING_URI", "https://127.0.0.1:5000"),
    )
    
    seed = Parameter(
        "seed",
        help="Random seed for reproducibility",
        default=42
    )

    segmented_data_path = Parameter(
        "segmented_data_path",
        help="Path to segmented ECG data HDF5",
        default="../data/interim/ecg_data_segmented.h5",
    )

    cleaned_data_path = Parameter(
        "cleaned_data_path",
        help="Path to cleaned ECG data HDF5",
        default="../data/interim/ecg_data_cleaned.h5",
    )

    window_data_path = Parameter(
        "window_data_path",
        help="Path to windowed data HDF5",
        default="../data/interim/windowed_data.h5",
    )

    model_type = Parameter(
        "model_type",
        help="Type of model being trained (e.g., cnn, lstm, transformer)",
        default=""
    )

    model_description = Parameter(
        "model_description",
        help="Additional description for the model run (e.g., testing residual connections)",
        default=""
    )
    
    accuracy_threshold = Parameter(
        "accuracy_threshold",
        help="Minimum accuracy for model registration",
        default=0.75,
    )
    
    lr = Parameter("lr", default=0.00001, help="Learning rate")
    
    patience = Parameter(
        "patience",
        help="Early stopping patience",
        default=3,
    )

    num_epochs = Parameter(
        "num_epochs",
        help="Training epochs",
        default=25,
    )

    batch_size = Parameter(
        "batch_size",
        help="Training batch size",
        default=16,
    )

    @card
    @step
    def start(self):
        """Initialize MLflow tracking"""
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        logging.info("MLflow tracking server: %s", self.mlflow_tracking_uri)

        try:
            run = mlflow.start_run(run_name=current.run_id)
            self.mlflow_run_id = run.info.run_id
        except Exception as e:
            raise RuntimeError(f"MLflow connection failed: {str(e)}")

        print("Starting simple training pipeline...")
        self.next(self.load_data)

    @card
    @step
    def load_data(self):
        """Load or process raw ECG data"""
        if not os.path.exists(self.segmented_data_path):
            print(f"Processing raw ECG data -> {self.segmented_data_path}")
            process_ecg_data(self.segmented_data_path)
        else:
            print(f"Using existing segmented data: {self.segmented_data_path}")
        
        self.next(self.clean_data)

    @step
    def clean_data(self):
        """Clean the ECG data"""
        if not os.path.exists(self.cleaned_data_path):
            print(f"Cleaning ECG data -> {self.cleaned_data_path}")
            process_save_cleaned_data(self.segmented_data_path, self.cleaned_data_path)
        else:
            print(f"Using existing cleaned data: {self.cleaned_data_path}")
            
        self.next(self.segment_data_windows)

    @card
    @step
    def segment_data_windows(self):
        """Segment data into windows"""
        from common import segment_data_into_windows

        if not os.path.exists(self.window_data_path):
            print("Segmenting ECG data into windows...")
            self.data = load_ecg_data(self.cleaned_data_path)
            segment_data_into_windows(
                self.data, 
                self.window_data_path, 
                fs=1000, 
                window_size=10, 
                step_size=5
            )
        print(f"Using windowed data: {self.window_data_path}")
        self.next(self.prepare_data_for_cnn)

    @resources(memory=8000)
    @card
    @step
    def prepare_data_for_cnn(self):
        """Prepare data for CNN training"""        
        X, y, groups = load_processed_data(
            hdf5_path=self.window_data_path,
            label_map={"baseline": 0, "mental_stress": 1},
        )
        print(f"Data loaded: X shape={X.shape}, y shape={y.shape}")
        
        (self.X_train, self.y_train), (self.X_val, self.y_val), (self.X_test, self.y_test) = split_data_by_participant(
            X, 
            y, 
            groups)

        print(f"Train samples: {len(self.X_train)}")
        print(f"Validation samples: {len(self.X_val)}")
        print(f"Test samples: {len(self.X_test)}")
        
        self.next(self.train_rocket)

    @card
    @step
    def train_rocket(self):
        """
        Train the ROCKET classifier on windowed ECG data.
        Converts the numpy arrays (with shape (N, window_length, 1)) to a nested
        pandas DataFrame format expected by sktime, and logs the transformation progress.
        """
        logging.info("Starting transformation to nested format for training, validation, and test data.")
        
        # Sample a subset of the data for faster processing
        from sklearn.model_selection import train_test_split
        
        # Define subset sizes
        train_subset_size = 1000  # 1000 training samples
        val_subset_size = 500     # 500 validation samples
        test_subset_size = 500    # 500 test samples
        
        #stratify
        X_train_sample, _, y_train_sample, _ = train_test_split(
            self.X_train, self.y_train, 
            train_size=min(train_subset_size, len(self.X_train)), 
            stratify=self.y_train,
            random_state=self.seed
        )
        
        X_val_sample, _, y_val_sample, _ = train_test_split(
            self.X_val, self.y_val, 
            train_size=min(val_subset_size, len(self.X_val)), 
            stratify=self.y_val,
            random_state=self.seed
        )
        
        X_test_sample, _, y_test_sample, _ = train_test_split(
            self.X_test, self.y_test, 
            train_size=min(test_subset_size, len(self.X_test)), 
            stratify=self.y_test,
            random_state=self.seed
        )
        
        print(f"Using {len(X_train_sample)} training samples (from {len(self.X_train)} total)")
        print(f"Using {len(X_val_sample)} validation samples (from {len(self.X_val)} total)")
        print(f"Using {len(X_test_sample)} test samples (from {len(self.X_test)} total)")
        
        # Transform training data
        print("Starting transformation of training data...")
        X_train_squeezed = X_train_sample.squeeze(-1)
        print(f"Training data shape after squeeze: {X_train_squeezed.shape}")
        X_train_nested = from_2d_array_to_nested(X_train_squeezed)
        print("Training data transformation complete.")
    
        # Transform validation data
        print("Starting transformation of validation data...")
        X_val_squeezed = X_val_sample.squeeze(-1)
        print(f"Validation data shape after squeeze: {X_val_squeezed.shape}")
        X_val_nested = from_2d_array_to_nested(X_val_squeezed)
        print("Validation data transformation complete.")
    
        # Transform test data
        print("Starting transformation of test data...")
        X_test_squeezed = X_test_sample.squeeze(-1)
        print(f"Test data shape after squeeze: {X_test_squeezed.shape}")
        self.X_test_nested = from_2d_array_to_nested(X_test_squeezed)
        self.y_test_subset = y_test_sample  
        print("Test data transformation complete.")
        
        logging.info("Nested data transformation complete.")
    
        # Initialize the ROCKET classifier 
        self.rocket_clf = RocketClassifier(
            num_kernels=10,  # Reduced
            rocket_transform='rocket',
            max_dilations_per_kernel=32,
            n_features_per_kernel=4,
            use_multivariate='auto',
            n_jobs=-1,            # Use all cores
            random_state=self.seed
        )
        
        # Fit the ROCKET classifier on the training data
        print("Fitting the ROCKET classifier...")
        self.rocket_clf.fit(X_train_nested, y_train_sample)
        print("ROCKET classifier fitting complete.")
        
        # Evaluate on the validation set
        self.y_val_pred = self.rocket_clf.predict(X_val_nested)
        self.rocket_val_accuracy = accuracy_score(y_val_sample, self.y_val_pred)
        print(f"ROCKET Validation Accuracy: {self.rocket_val_accuracy*100:.2f}%")
        print("ROCKET Classification Report (Validation):")
        print(classification_report(y_val_sample, self.y_val_pred))
        
        self.next(self.evaluate_rocket)

    @card
    @step
    def evaluate_rocket(self):
        """
        Evaluate the ROCKET classifier on test data.
        Calculates accuracy, prints a classification report and confusion matrix,
        and logs metrics using MLflow.
        """
        # Predict on the test set
        self.y_test_pred = self.rocket_clf.predict(self.X_test_nested)
        self.rocket_test_accuracy = accuracy_score(self.y_test_subset, self.y_test_pred)
        print(f"ROCKET Test Accuracy: {self.rocket_test_accuracy*100:.2f}%")
        
        # Print detailed reports
        print("ROCKET Classification Report (Test):")
        print(classification_report(self.y_test_subset, self.y_test_pred))
        print("Confusion Matrix:")
        print(confusion_matrix(self.y_test_subset, self.y_test_pred))
        
        # Log metrics to MLflow
        mlflow.log_metric("rocket_val_accuracy", self.rocket_val_accuracy)
        mlflow.log_metric("rocket_test_accuracy", self.rocket_test_accuracy)
        
        self.next(self.register_rocket)

    @step
    def register_rocket(self):
        """
        Register the ROCKET classifier model in MLflow if its test accuracy
        meets or exceeds the specified threshold.
        """
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        
        if self.rocket_test_accuracy >= self.accuracy_threshold:
            self.registered_rocket = True
            print("Registering ROCKET classifier model...")
            mlflow.sklearn.log_model(
                self.rocket_clf,
                artifact_path="rocket_model",
                registered_model_name="ROCKET_ECG_Classifier"
            )
            print("ROCKET classifier model registered!")
        else:
            self.registered_rocket = False
            print(f"ROCKET classifier test accuracy {self.rocket_test_accuracy:.3f} below threshold {self.accuracy_threshold}")
            
        self.next(self.end)

    @card
    @step
    def end(self):
        """Finish the ROCKET training branch."""
        print("\n=== ROCKET Training Pipeline Complete ===")
        print(f"Final ROCKET Test Accuracy: {self.rocket_test_accuracy:.3f}")
        print(f"Accuracy Threshold: {self.accuracy_threshold}")
        # Optionally, delete large objects
        print("Done!")

if __name__ == "__main__":
    ECGSimpleTrainingFlow()
