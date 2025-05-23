import os
import logging
import keras
import mlflow
import mlflow.sklearn
import mlflow.keras 
import numpy as np
from tqdm.keras import TqdmCallback
from sklearn.metrics import confusion_matrix, classification_report
from metaflow import FlowSpec, step, card, Parameter, current, project, environment
from mlflow.models import infer_signature

from common import (
    process_ecg_data,
    process_save_cleaned_data,
)

from keras_models import (
    improved_1DCNN,
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
        default=0.6,
    )

    num_epochs = Parameter(
        "num_epochs",
        help="Training epochs",
        default=25,
    )

    batch_size = Parameter(
        "batch_size",
        help="Training batch size",
        default=32,
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

    @card
    @step
    def prepare_data_for_cnn(self):
        """Prepare data for CNN training with validation split"""
        self.X, self.y, self.groups = prepare_cnn_data(
            hdf5_path=self.window_data_path,
            label_map={"baseline": 0, "mental_stress": 1},
        )
        print(f"Data loaded: X shape={self.X.shape}, y shape={self.y.shape}")

        # Get unique participants
        unique_participants = np.unique(self.groups)
        n_participants = len(unique_participants)
        
        # Split participants: 60% train, 20% validation, 20% test
        n_train = int(n_participants * 0.6)
        n_val = int(n_participants * 0.2)
        
        # Randomly select participants with fixed seed
        np.random.seed(42)
        train_participants = np.random.choice(
            unique_participants, 
            size=n_train, 
            replace=False
        )
        
        remaining_participants = np.array([
            p for p in unique_participants if p not in train_participants
        ])
        
        val_participants = np.random.choice(
            remaining_participants,
            size=n_val,
            replace=False
        )
        
        test_participants = np.array([
            p for p in remaining_participants if p not in val_participants
        ])
        
        # Create masks for splits
        train_mask = np.isin(self.groups, train_participants)
        val_mask = np.isin(self.groups, val_participants)
        test_mask = np.isin(self.groups, test_participants)
        
        # Split the data
        self.X_train = self.X[train_mask]
        self.y_train = self.y[train_mask]
        self.X_val = self.X[val_mask]
        self.y_val = self.y[val_mask]
        self.X_test = self.X[test_mask]
        self.y_test = self.y[test_mask]
        
        print("\nData split sizes:")
        print(f"Train: {len(self.X_train)} samples")
        print(f"Validation: {len(self.X_val)} samples")
        print(f"Test: {len(self.X_test)} samples")
    
        # check class distributions:
        def print_class_distribution(y, split_name):
            unique, counts = np.unique(y, return_counts=True)
            total = len(y)
            print(f"\n{split_name} Class Distribution:")
            for label, count in zip(unique, counts):
                percentage = (count / total) * 100
                class_name = "Baseline" if label == 0 else "Mental Stress"
                print(f"{class_name}: {count} samples ({percentage:.1f}%)")
        
        # Print distributions for each split
        print("\n=== Class Distribution Analysis ===")
        print_class_distribution(self.y_train, "Training")
        print_class_distribution(self.y_val, "Validation")
        print_class_distribution(self.y_test, "Test")
        
        self.next(self.train_model)

    @card
    @environment(vars={"KERAS_BACKEND": os.getenv("KERAS_BACKEND", "jax")})
    @step
    def train_model(self):
        """Train the CNN model"""
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)

        with mlflow.start_run(run_id=self.mlflow_run_id):
            # Add tags for better organization
            mlflow.set_tags({
                "model_type": self.model_type,
                "description": self.model_description,
            })
            
            mlflow.autolog(log_models=False)
            
            self.model = improved_1DCNN(input_length=10000)
            class PrintEpochMetricsCallback(keras.callbacks.Callback):
                def on_epoch_end(self, epoch, logs=None):
                    print(
                        f"Epoch {epoch+1}: "
                        f"loss={logs.get('loss'):.3f}, "
                        f"binary_accuracy={logs.get('binary_accuracy'):.3f}, "
                        f"val_loss={logs.get('val_loss'):.3f}, "
                        f"val_binary_accuracy={logs.get('val_binary_accuracy'):.3f}"
                    )
            
            history = self.model.fit(
                self.X_train,
                self.y_train,
                validation_data=(self.X_val, self.y_val),
                epochs=self.num_epochs,
                batch_size=self.batch_size,
                verbose=0, 
                callbacks=[
                    TqdmCallback(verbose=1),  
                    PrintEpochMetricsCallback(),
                ],
            )
                
            self.train_history = history.history
            
            logging.info(
                "train_loss: %f - train_accuracy: %f - val_loss: %f - val_accuracy: %f",
                history.history["loss"][-1],
                history.history["binary_accuracy"][-1],
                history.history["val_loss"][-1],
                history.history["val_binary_accuracy"][-1],
            )
            
        self.next(self.evaluate)

    @card
    @step
    def evaluate(self):
        """Evaluate model performance"""
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        
        y_pred = (self.model.predict(self.X_test) > 0.5).astype(int)
        self.test_loss, self.test_accuracy = self.model.evaluate(
            self.X_test, 
            self.y_test, 
            verbose=0,
        )
        
        # Print confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        print("\nConfusion Matrix:")
        print("--------------- Predicted")
        print("--------------- Baseline  Mental Stress")
        print(f"Actual Baseline    {cm[0][0]:4d}         {cm[0][1]:4d}")
        print(f"Mental Stress      {cm[1][0]:4d}         {cm[1][1]:4d}")
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred, 
                                target_names=['Baseline', 'Mental Stress']))
        
        print(f"\nTest accuracy: {self.test_accuracy:.3f}")
        
        logging.info(
            "test_loss: %f - test_accuracy: %f",
            self.test_loss,
            self.test_accuracy,
        )

        mlflow.log_metrics({
            "test_loss": self.test_loss,
            "test_accuracy": self.test_accuracy,
        }, run_id=self.mlflow_run_id)
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
                sample_input = self.X_test[:5]
                sample_output = self.model.predict(sample_input)
                signature = infer_signature(sample_input, sample_output)

                mlflow.keras.log_model(
                    self.model,
                    artifact_path="model",
                    registered_model_name="baseline_1DCNN",
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