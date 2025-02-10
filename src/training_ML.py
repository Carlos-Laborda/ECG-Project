import os
import logging
import mlflow
import mlflow.sklearn
import numpy as np
from metaflow import FlowSpec, step, card, Parameter, current, project, environment
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from mlflow.models import infer_signature

from common import (
    process_ecg_data,
    process_save_cleaned_data,
    baseline_1DCNN,
    baseline_1DCNN_improved,
)
from utils import load_ecg_data, prepare_cnn_data


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

    accuracy_threshold = Parameter(
        "accuracy_threshold",
        help="Minimum accuracy for model registration",
        default=0.7,
    )

    num_epochs = Parameter(
        "num_epochs",
        help="Training epochs",
        default=5,
    )

    batch_size = Parameter(
        "batch_size",
        help="Training batch size",
        default=32,
    )

    test_size = Parameter(
        "test_size",
        help="Proportion of data for testing",
        default=0.2,
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
                step_size=1
            )
        print(f"Using windowed data: {self.window_data_path}")
        self.next(self.prepare_data_for_cnn)

    @card
    @step
    def prepare_data_for_cnn(self):
        """Prepare data for RandomForest training"""
        self.X, self.y, self.groups = prepare_cnn_data(
            hdf5_path=self.window_data_path,
            label_map={"baseline": 0, "low_physical_activity": 1},
        )
        
        # Reshape the data for RandomForest (flatten the time series)
        self.X = self.X.reshape(self.X.shape[0], -1)  # Flatten the windows
        
        print(f"Data loaded: X shape={self.X.shape}, y shape={self.y.shape}")

        # Get unique participants
        unique_participants = np.unique(self.groups)
        n_participants = len(unique_participants)
        n_train = int(n_participants * 0.8)  # 80% for training
        
        # Randomly select participants for train/test (with fixed seed for reproducibility)
        np.random.seed(42)
        train_participants = np.random.choice(
            unique_participants, 
            size=n_train, 
            replace=False
        )
        test_participants = np.array([
            p for p in unique_participants if p not in train_participants
        ])
        
        # Create masks for train/test split
        train_mask = np.isin(self.groups, train_participants)
        test_mask = np.isin(self.groups, test_participants)
        
        # Split the data
        self.X_train = self.X[train_mask]
        self.y_train = self.y[train_mask]
        self.X_test = self.X[test_mask]
        self.y_test = self.y[test_mask]
        
        # Print detailed information about the split
        print("\nParticipant-level split details:")
        print(f"Total participants: {n_participants}")
        print(f"Training participants: {len(train_participants)} ({train_participants})")
        print(f"Test participants: {len(test_participants)} ({test_participants})")
        print(f"Train size: {len(self.X_train)}, Test size: {len(self.X_test)}")
        
        # Print class distribution for both sets
        train_dist = np.bincount(self.y_train) / len(self.y_train)
        test_dist = np.bincount(self.y_test) / len(self.y_test)
        print("\nClass distribution:")
        print(f"Train - baseline: {train_dist[0]:.2f}, high activity: {train_dist[1]:.2f}")
        print(f"Test  - baseline: {test_dist[0]:.2f}, high activity: {test_dist[1]:.2f}")
        
        self.next(self.train_model)

    @card
    @step
    def train_model(self):
        """Train the RandomForest model"""
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)

        with mlflow.start_run(run_id=self.mlflow_run_id):
            mlflow.autolog(log_models=False)
            
            # Scale the features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(self.X_train)
            X_test_scaled = scaler.transform(self.X_test)
            
            # Initialize and train RandomForest
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1  # Use all available cores
            )
            
            self.model.fit(X_train_scaled, self.y_train)
            
            # Store training metrics
            train_score = self.model.score(X_train_scaled, self.y_train)
            test_score = self.model.score(X_test_scaled, self.y_test)
            
            logging.info(
                "train_accuracy: %f - test_accuracy: %f",
                train_score,
                test_score
            )
            
            # Store scaler for later use
            self.scaler = scaler
            
        self.next(self.evaluate)

    @card
    @step
    def evaluate(self):
        """Evaluate RandomForest performance"""
        # Scale test data
        X_test_scaled = self.scaler.transform(self.X_test)
        
        # Get predictions
        self.test_accuracy = self.model.score(X_test_scaled, self.y_test)
        y_pred = self.model.predict(X_test_scaled)
        
        print(f"\nTest accuracy: {self.test_accuracy:.3f}")
        
        # Log metrics
        mlflow.log_metrics({
            "test_accuracy": self.test_accuracy,
        })
        
        self.next(self.register)

    @environment(vars={"KERAS_BACKEND": os.getenv("KERAS_BACKEND", "jax")})
    @step
    def register(self):
        """Register model if accuracy meets threshold"""
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)

        if self.test_accuracy >= self.accuracy_threshold:
            self.registered = True
            logging.info("Registering model...")
            
            with mlflow.start_run(run_id=self.mlflow_run_id):
                sample_input = self.scaler.transform(self.X_test[:5])
                sample_output = self.model.predict(sample_input)
                signature = infer_signature(sample_input, sample_output)

                mlflow.sklearn.log_model(
                    self.model,
                    artifact_path="model",
                    registered_model_name="baseline_randomforest",
                    signature=signature,
                )
                print("Model successfully registered!")
        else:
            self.registered = False
            print(f"Model accuracy {self.test_accuracy:.3f} below threshold {self.accuracy_threshold}")
        
        self.next(self.end)

    @card
    @step
    def end(self):
        """Finish the pipeline"""
        print("\n=== Training Pipeline Complete ===")
        print(f"Final Test Accuracy: {self.test_accuracy:.3f}")
        print(f"Threshold: {self.accuracy_threshold}")
        print("Done!")


if __name__ == "__main__":
    ECGSimpleTrainingFlow()