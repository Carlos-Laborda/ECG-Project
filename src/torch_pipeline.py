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
from mlflow.types import Schema, TensorSpec
from mlflow.models import ModelSignature

from common import (
    process_ecg_data,
    process_save_cleaned_data,
)

from torch_utilities import (
    load_processed_data, split_data_by_participant, ECGDataset,
    Improved1DCNN, train, test, log_model_summary, Simple1DCNN,
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
        default=0.5,
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
        
        (X_train, y_train), (X_val, y_val), (self.X_test, y_test) = split_data_by_participant(
            self.X, 
            self.y, 
            self.groups)

        print(f"Train samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Test samples: {len(self.X_test)}")
        
        # Create PyTorch datasets
        train_dataset = ECGDataset(X_train, y_train)
        val_dataset = ECGDataset(X_val, y_val)
        test_dataset = ECGDataset(self.X_test, y_test)
        
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
        self.model = Simple1DCNN().to(device)
        
        # Define loss function (using BCELoss for binary classification)
        loss_fn = torch.nn.BCELoss()
        # Define optimizer and scheduler
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.95)
                
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        with mlflow.start_run(run_id=self.mlflow_run_id):
            # Log training parameters
            params = {
                # Model parameters
                "model_type": self.model_type,
                "model_description": self.model_description,
                
                # Training hyperparameters
                "epochs": self.num_epochs,
                "learning_rate": self.lr,
                "batch_size": self.batch_size,
                "optimizer": optimizer.__class__.__name__,
                "loss_function": loss_fn.__class__.__name__,
                
                # Scheduler parameters
                "scheduler": scheduler.__class__.__name__,
                "scheduler_step_size": scheduler.step_size,
                "scheduler_gamma": scheduler.gamma,
                
                # Data parameters
                "input_shape": f"{self.X.shape}",
                "train_samples": len(self.train_loader.dataset),
                "val_samples": len(self.val_loader.dataset),
                "test_samples": len(self.test_loader.dataset),
                
                # Hardware
                "device": device
            }
            mlflow.log_params(params)
            
            # Log model summary artifact; input shape is (batch, 1, window_length)
            input_size = (self.batch_size, 1, self.X.shape[1])
            log_model_summary(self.model, input_size)
            
            # Training loop with validation
            for epoch in range(1, self.num_epochs + 1):
                print(f"\nEpoch {epoch}/{self.num_epochs}")
                
                # Train and log metrics
                self.train_loss, self.train_acc = train(
                    self.model, self.train_loader, optimizer, 
                    loss_fn, device, epoch)
                
                # Validate and log metrics
                self.val_loss, self.val_acc = test(
                    self.model, self.val_loader, loss_fn, 
                    device, phase='val', epoch=epoch)
                
                # Update learning rate
                scheduler.step()
        
        self.next(self.evaluate)
    @card
    @step
    def evaluate(self):
        """Evaluate model performance on test data"""
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        loss_fn = torch.nn.BCELoss()
        with mlflow.start_run(run_id=self.mlflow_run_id):
            # Evaluate and log test metrics
            self.test_loss, self.test_accuracy = test(
                self.model, self.test_loader, loss_fn, 
                device, phase='test')
            print(f"Final Test Accuracy: {self.test_accuracy*100:.2f}%")
        self.next(self.register)

    @step
    def register(self):
        """Register model if accuracy meets threshold"""
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        
        if self.test_accuracy >= self.accuracy_threshold:
            self.registered = True
            logging.info("Registering model...")
            
            with mlflow.start_run(run_id=self.mlflow_run_id):
                # Prepare sample input and get prediction
                sample_input = self.X_test[:5] 
                self.model.cpu().eval()
                
                with torch.no_grad():
                    # Convert to tensor, permute to (batch, 1, window_length), and get prediction
                    tensor_input = torch.tensor(sample_input, dtype=torch.float32).permute(0, 2, 1)
                    sample_output = self.model(tensor_input).numpy()
                
                # Define input/output schema
                input_schema = Schema([TensorSpec(np.dtype(np.float32), (-1, self.X.shape[1], 1))])
                output_schema = Schema([TensorSpec(np.dtype(np.float32), (-1, 1))])
                signature = ModelSignature(inputs=input_schema, outputs=output_schema)
    
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