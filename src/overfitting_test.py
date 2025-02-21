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
    baseline_1DCNN,
    baseline_1DCNN_improved,
    baseline_1DCNN_improved2,
    neural_network,
    baseline_LSTM,
    cnn_overfit,
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
        default=0.60,
    )

    num_epochs = Parameter(
        "num_epochs",
        help="Training epochs",
        default=100,
    )

    batch_size = Parameter(
        "batch_size",
        help="Training batch size",
        default=8,
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
        # Load and verify initial data
        self.X, self.y, self.groups = prepare_cnn_data(
            hdf5_path=self.window_data_path,
            label_map={"baseline": 0, "mental_stress": 1},
        )
        
        # Log initial data state
        print("\n=== Initial Data State ===")
        print(f"Initial X shape: {self.X.shape}")
        print(f"Initial y shape: {self.y.shape}")
        print(f"Sample from X[0,:50,0]: {self.X[0,:50,0]}")
        print(f"Label for sample: {self.y[0]} ({'Baseline' if self.y[0]==0 else 'Mental Stress'})")
        
        # Store original data for validation
        self._original_X = self.X.copy()
        self._original_y = self.y.copy()
        
        # Get unique participants
        unique_participants = np.unique(self.groups)
        
        # For overfitting test, use only 2 participants
        np.random.seed(42)  # For reproducibility
        n_participants = 2
        train_participants = np.random.choice(
            unique_participants, 
            size=n_participants, 
            replace=False
        )
        
        print(f"\nSelected participants: {train_participants}")
        
        # Create mask for small training set
        train_mask = np.isin(self.groups, train_participants)
        
        # Use same data for train/val/test to check overfitting
        self.X_train = self.X[train_mask]
        self.y_train = self.y[train_mask]
        self.X_val = self.X_train.copy()  # Explicit copy
        self.y_val = self.y_train.copy()  # Explicit copy
        self.X_test = self.X_train.copy()  # Explicit copy
        self.y_test = self.y_train.copy()  # Explicit copy
        
        # Verify data hasn't been modified
        assert np.array_equal(self.X, self._original_X), "Original X was modified!"
        assert np.array_equal(self.y, self._original_y), "Original y was modified!"
        
        # Log final data state
        print("\n=== Final Data State ===")
        print(f"X_train shape: {self.X_train.shape}")
        print(f"y_train shape: {self.y_train.shape}")
        
        # Log random samples for verification
        idx = np.random.randint(0, len(self.y_train))
        print(f"\nRandom sample (index {idx}):")
        print(f"First 50 points: {self.X_train[idx,:50,0]}")
        print(f"Label: {self.y_train[idx]} ({'Baseline' if self.y_train[idx]==0 else 'Mental Stress'})")
        
        self.next(self.train_model)
    
    @card
    @environment(vars={"KERAS_BACKEND": os.getenv("KERAS_BACKEND", "jax")})
    @step
    def train_model(self):
        """Train the CNN model with additional data validation"""
        # Verify data consistency across steps
        print("\n=== Pre-training Data Verification ===")
        
        # Check shapes
        print(f"X_train shape: {self.X_train.shape}")
        print(f"y_train shape: {self.y_train.shape}")
        
        # Log multiple random samples
        for i in range(3):
            idx = np.random.randint(0, len(self.y_train))
            print(f"\nRandom sample {i+1} (index {idx}):")
            print(f"Signal snippet: {self.X_train[idx,:50,0]}")
            print(f"Label: {self.y_train[idx]} ({'Baseline' if self.y_train[idx]==0 else 'Mental Stress'})")
            print(f"Signal stats - Min: {np.min(self.X_train[idx]):.6f}, Max: {np.max(self.X_train[idx]):.6f}")
        
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)

        # Add debugging information
        print("\n=== Training Data Debug Info ===")
        print(f"X_train shape: {self.X_train.shape}")
        print(f"y_train shape: {self.y_train.shape}")
        
        # Check data range and statistics
        print("\nX_train statistics:")
        print(f"Range: [{np.min(self.X_train):.6f}, {np.max(self.X_train):.6f}]")
        print(f"Mean: {np.mean(self.X_train):.6f}")
        print(f"Std: {np.std(self.X_train):.6f}")
        
        # Check unique values in first window to ensure variety
        print("\nFirst window statistics:")
        first_window = self.X_train[0, :, 0]
        print(f"Unique values in first window: {len(np.unique(first_window))}")
        print(f"First window range: [{np.min(first_window):.6f}, {np.max(first_window):.6f}]")
        
        # Check class distribution
        unique_labels, counts = np.unique(self.y_train, return_counts=True)
        print("\nClass distribution in y_train:")
        for label, count in zip(unique_labels, counts):
            class_name = "Baseline" if label == 0 else "Mental Stress"
            percentage = (count / len(self.y_train)) * 100
            print(f"{class_name} (label {label}): {count} samples ({percentage:.1f}%)")
        
        # Verify no NaN values
        print("\nData quality checks:")
        print(f"NaN in X_train: {np.isnan(self.X_train).any()}")
        print(f"NaN in y_train: {np.isnan(self.y_train).any()}")

        with mlflow.start_run(run_id=self.mlflow_run_id):
            mlflow.autolog(log_models=False)
            
            self.model = cnn_overfit(input_length=10000)
            
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
                    PrintEpochMetricsCallback() 
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