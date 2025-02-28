import os
import logging
import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
from metaflow import FlowSpec, step, card, Parameter, current, project, environment
from mlflow.models import infer_signature

from common import (
    process_ecg_data,
    process_save_cleaned_data,
)

from torch_utilities import (
    load_processed_data, split_data_by_participant, ECGDataset,
    Improved1DCNN, train, test, log_model_summary
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
    
    lr = Parameter("lr", default=0.001, help="Learning rate")

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

    @card
    @step
    def prepare_data_for_cnn(self):
        """Prepare data for CNN training"""
        self.X, self.y, self.groups = load_processed_data(
            hdf5_path=self.window_data_path,
            label_map={"baseline": 0, "mental_stress": 1},
        )
        print(f"Data loaded: X shape={self.X.shape}, y shape={self.y.shape}")
        
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_data_by_participant(self.X, 
                                                                                         self.y, 
                                                                                         self.groups)

        print(f"Train samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Test samples: {len(X_test)}")
        
        # Create PyTorch datasets
        train_dataset = ECGDataset(X_train, y_train)
        val_dataset = ECGDataset(X_val, y_val)
        test_dataset = ECGDataset(X_test, y_test)
        
        # Create DataLoaders
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        self.next(self.train_model)

    @card
    @step
    def train_model(self):
        """Train the CNN model using PyTorch"""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Training on device: {device}")
        
        # Instantiate the model and move it to device
        self.model = Improved1DCNN().to(device)
        
        # Define loss function (using BCELoss for binary classification)
        loss_fn = torch.nn.BCELoss()
        # Define optimizer and scheduler
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.95)
        
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        with mlflow.start_run(run_id=self.mlflow_run_id):
            mlflow.log_param("lr", self.lr)
            mlflow.log_param("epochs", self.num_epochs)
            mlflow.log_param("batch_size", self.batch_size)
            
            # Log model summary artifact; assume input shape is (batch, 1, window_length)
            #input_size = (self.batch_size, 1, self.X.shape[1])
            #log_model_summary(self.model, input_size)
            
            for epoch in range(1, self.num_epochs + 1):
                print(f"\nEpoch {epoch}/{self.num_epochs}")
                train(self.model, self.train_loader, optimizer, loss_fn, device, epoch)
                _ = test(self.model, self.val_loader, loss_fn, device)
                scheduler.step()
            
            # Log the final model artifact
            mlflow.pytorch.log_model(self.model, "model")
    
        self.next(self.evaluate)

    @card
    @step
    def evaluate(self):
        """Evaluate model performance on test data"""
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        """Evaluate the trained model on the test set"""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        loss_fn = torch.nn.BCELoss()
        test_accuracy = test(self.model, self.test_loader, loss_fn, device)
        print(f"Final Test Accuracy: {test_accuracy*100:.2f}%")
        self.next(self.register)

    @step
    def register(self):
        """Register model if accuracy meets threshold"""
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.test_accuracy >= self.accuracy_threshold:
            self.registered = True
            logging.info("Registering model...")
            
            with mlflow.start_run(run_id=self.mlflow_run_id):
                sample_input = torch.tensor(self.X_test[:5], dtype=torch.float32).to(device)
                sample_output = self.model(sample_input).detach().cpu().numpy()
                signature = infer_signature(self.X_test[:5], sample_output)

                mlflow.pytorch.log_model(
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