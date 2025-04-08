import os
import logging
import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
from metaflow import (FlowSpec, step, card, Parameter, current, 
                      project, resources, conda_base, environment)

from common2 import (
    process_ecg_data,
    process_save_cleaned_data,
    normalize_cleaned_data,
)

from torch_utilities import (
    load_processed_data, split_data_by_participant, ECGDataset,
    train, test, EarlyStopping, log_model_summary, prepare_model_signature,
    set_seed, Simple1DCNN, Simple1DCNN_v2, Improved1DCNN, Improved1DCNN_v2,
    EmotionRecognitionCNN, xresnet1d101, TCNClassifier, TransformerECGClassifier
)

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
    
    normalized_data_path = Parameter(
            "normalized_data_path",
            help="Path to normalized ECG data HDF5",
            default="../data/interim/ecg_data_normalized.h5",
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
        default=0.74,
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
            
        self.next(self.normalize_data)
        
    @step
    def normalize_data(self):
        """Perform user-specific z-score normalization on cleaned data"""        
        if not os.path.exists(self.normalized_data_path):
            print(f"Normalizing cleaned data -> {self.normalized_data_path}")
            normalize_cleaned_data(self.cleaned_data_path, self.normalized_data_path)
        else:
            print(f"Using existing normalized data: {self.normalized_data_path}")
        self.next(self.segment_data_windows)

    @step
    def segment_data_windows(self):
        """Segment data into windows"""
        from common2 import segment_data_into_windows

        if not os.path.exists(self.window_data_path):
            print("Segmenting ECG data into windows...")
            segment_data_into_windows(
                self.normalized_data_path, 
                self.window_data_path, 
                fs=1000, 
                window_size=10, 
                step_size=5
            )
        print(f"Using windowed data: {self.window_data_path}")
        self.next(self.prepare_data_for_cnn)

    @resources(memory=16000)
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
        
        self.next(self.train_model)
        
    @resources(memory=16000)    
    @step
    def train_model(self):
        """Train the CNN model using PyTorch"""
        set_seed(self.seed)
        
        # Create PyTorch datasets
        train_dataset = ECGDataset(self.X_train, self.y_train)
        val_dataset = ECGDataset(self.X_val, self.y_val)
        test_dataset = ECGDataset(self.X_test, self.y_test)
        
        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, 
                                       shuffle=True, num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, 
                                     num_workers=0, pin_memory=True)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, 
                                      num_workers=0, pin_memory=True)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Training on device: {device}")
        
        # Model setup
        self.model = TransformerECGClassifier().to(device)
        loss_fn = torch.nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        # ReduceLROnPlateau scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',           # reduce LR when the validation loss stops decreasing
            factor=0.1,           # multiply LR by this factor when reducing
            patience=2,           # number of epochs with no improvement after which LR is reduced
            verbose=False,         # print message when LR is reduced
            min_lr=1e-11           # lower bound on the learning rate
        )
        
        early_stopping = EarlyStopping(patience=self.patience)
                
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        with mlflow.start_run(run_id=self.mlflow_run_id):
            # Log training parameters
            params = {
                # Model parameters
                "model_type": self.model_type,
                "model_description": self.model_description,
                "model_name": self.model.__class__.__name__,
                "random_seed": self.seed,
                "deterministic": True,
                
                # Training hyperparameters
                "epochs": self.num_epochs,
                "learning_rate": self.lr,
                "batch_size": self.batch_size,
                "optimizer": optimizer.__class__.__name__,
                "loss_function": loss_fn.__class__.__name__,
                
                # Scheduler parameters
                "scheduler": scheduler.__class__.__name__,
                #"scheduler_gamma": scheduler.gamma,
                "scheduler_mode": "min",
                "scheduler_factor": 0.1,
                "scheduler_patience": 2,
                "scheduler_min_lr": 1e-11,
                
                # Early stopping
                "patience": self.patience,
                
                # Data parameters
                #"input_shape": f"{self.X.shape}",
                "train_samples": len(train_loader.dataset),
                "val_samples": len(val_loader.dataset),
                "test_samples": len(self.test_loader.dataset),
                
                # Hardware
                "device": device
            }
            mlflow.log_params(params)
            
            # Log model summary artifact; input shape is (batch, 1, window_length)
            #input_size = (self.batch_size, 1, self.X.shape[1])
            #log_model_summary(self.model, input_size)
            
            # Training loop with validation
            for epoch in range(1, self.num_epochs + 1):
                print(f"\nEpoch {epoch}/{self.num_epochs}")
                
                # Train and log metrics
                self.train_loss, self.train_acc, self.train_auc = train(
                    self.model, train_loader, optimizer, 
                    loss_fn, device, epoch)
                
                # Validate and log metrics
                self.val_loss, self.val_acc, self.val_auc = test(
                    self.model, val_loader, loss_fn, 
                    device, phase='val', epoch=epoch)

                # Early stopping
                early_stopping(self.val_loss)
                if early_stopping.early_stop:
                    print("Early stopping triggered")
                    break
                
                # Update learning rate
                scheduler.step(self.val_loss)
                
        # Cleanup: Remove heavy training data from the flow state to save disk space
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None
        
        self.next(self.evaluate)

    @step
    def evaluate(self):
        """Evaluate model performance on test data"""
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        loss_fn = torch.nn.BCELoss()
        with mlflow.start_run(run_id=self.mlflow_run_id):
            # Evaluate and log test metrics
            self.test_loss, self.test_accuracy, self.test_auc = test(
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
                signature = prepare_model_signature(
                self.model, 
                self.X_test[:5]
                )
    
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

    @step
    def end(self):
        """Finish the pipeline"""
        print("\n=== Training Pipeline Complete ===")
        print(f"Final Test Accuracy: {self.test_accuracy:.3f}")
        print(f"Threshold: {self.accuracy_threshold}")

        print("Done!")

if __name__ == "__main__":
    ECGSimpleTrainingFlow()