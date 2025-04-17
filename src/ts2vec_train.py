import os
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
import mlflow.pytorch
from torch.utils.data import DataLoader, TensorDataset
from metaflow import FlowSpec, step, Parameter, current, project, resources

from ts2vec import TS2Vec, SimpleClassifier

from torch_utilities import load_processed_data, split_data_by_participant, set_seed

# -------------------------
# Metaflow Pipeline for TS2Vec pretraining and classifier fine-tuning
# -------------------------
@project(name="ecg_training_ts2vec")
class ECGTS2VecFlow(FlowSpec):
    mlflow_tracking_uri = Parameter(
        "mlflow_tracking_uri",
        help="MLflow tracking server location",
        default=os.getenv("MLFLOW_TRACKING_URI", "https://127.0.0.1:5000")
    )
    
    window_data_path = Parameter(
        "window_data_path",
        help="Path to windowed ECG data HDF5",
        default="../data/interim/windowed_data.h5"
    )

    seed = Parameter("seed", help="Random seed", default=42)
    ts2vec_epochs = Parameter("ts2vec_epochs", help="Epochs for TS2Vec pretraining", default=50)
    ts2vec_lr = Parameter("ts2vec_lr", help="Learning rate for TS2Vec", default=0.001)
    ts2vec_batch_size = Parameter("ts2vec_batch_size", help="Batch size for TS2Vec", default=8)
    classifier_epochs = Parameter("classifier_epochs", help="Epochs for classifier training", default=25)
    classifier_lr = Parameter("classifier_lr", help="Learning rate for classifier", default=0.0001)
    classifier_batch_size = Parameter("classifier_batch_size", help="Batch size for classifier", default=32)
    accuracy_threshold = Parameter("accuracy_threshold", help="Minimum accuracy for model registration", default=0.74)
    
    label_fraction = Parameter(
        "label_fraction", 
        help="Fraction of labeled training data for classifier (e.g., 0.01, 0.05, 0.1, 1.0)", 
        default=1.0 
    )

    @step
    def start(self):
        """Initialize MLflow and set seed."""
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        logging.info("MLflow tracking server: %s", self.mlflow_tracking_uri)
        try:
            run = mlflow.start_run(run_name=current.run_id)
            self.mlflow_run_id = run.info.run_id
        except Exception as e:
            raise RuntimeError(f"MLflow connection failed: {str(e)}")
        
        set_seed(self.seed)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        self.next(self.preprocess_data)

    @step
    def preprocess_data(self):
        """Load or generate the windowed data for training.
        The expected shape is (n_instances, sequence_length, n_features)."""
        X, y, groups = load_processed_data(
            hdf5_path=self.window_data_path,
            label_map={"baseline": 0, "mental_stress": 1}
        )
        print(f"Windowed data loaded: X.shape={X.shape}, y.shape={y.shape}")

        # Split the data into train/validation/test by participant if needed
        (self.X_train, self.y_train), (self.X_val, self.y_val), (self.X_test, self.y_test) = split_data_by_participant(
            X, y, groups
        )
        print(f"Train samples: {len(self.X_train)}")
        self.next(self.train_ts2vec)

    @resources(memory=16000)
    @step
    def train_ts2vec(self):
        """Train the TS2Vec model in a self-supervised way."""
        print(f"Training TS2Vec on device: {self.device}")

        # Assume each window has shape (window_length, 1)
        # TS2Vec requires a numpy array of shape (n_instances, sequence_length, n_features)
        input_dims = self.X_train.shape[2]  # e.g., 1 for univariate ECG
        
        # Set hyperparameters for TS2Vec
        output_dims=320
        hidden_dims=64
        depth=10
        max_train_length=5000  # can be adjusted
        temporal_unit=0
        
        # Create TS2Vec model instance with the chosen hyperparameters
        self.ts2vec = TS2Vec(
            input_dims=input_dims,
            output_dims=output_dims,
            hidden_dims=hidden_dims,
            depth=depth,
            device=self.device,
            lr=self.ts2vec_lr,
            batch_size=self.ts2vec_batch_size,
            max_train_length=max_train_length,
            temporal_unit=temporal_unit
        )
        
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        with mlflow.start_run(run_id=self.mlflow_run_id):
            params = {
                "ts2vec_epochs": self.ts2vec_epochs,
                "ts2vec_lr": self.ts2vec_lr,
                "ts2vec_batch_size": self.ts2vec_batch_size,
                "ts2vec_output_dims": output_dims, 
                "ts2vec_hidden_dims": hidden_dims,
                "ts2vec_depth": depth,
                "ts2vec_max_train_length": max_train_length,
                "ts2vec_temporal_unit": temporal_unit,
            }
            mlflow.log_params(params)
        
            # Train TS2Vec using self-supervised learning
            loss_log = self.ts2vec.fit(self.X_train, n_epochs=self.ts2vec_epochs, verbose=True)
            print("TS2Vec training complete.")

            # Save the trained TS2Vec model checkpoint
            self.ts2vec_model_path = f"ts2vec_{self.mlflow_run_id}.pth"
            self.ts2vec.save(self.ts2vec_model_path)
            mlflow.log_artifact(self.ts2vec_model_path, artifact_path="ts2vec_model")
        self.next(self.extract_representations)

    @step
    def extract_representations(self):
        """Extract feature representations using the trained TS2Vec encoder."""
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        # Use TS2Vec.encode() to compute representations for train, val and test sets.
        self.train_repr = self.ts2vec.encode(self.X_train, encoding_window="full_series")
        self.val_repr = self.ts2vec.encode(self.X_val, encoding_window="full_series")
        self.test_repr = self.ts2vec.encode(self.X_test, encoding_window="full_series")
        print(f"Extracted TS2Vec representations: train_repr shape={self.train_repr.shape}")
        self.next(self.train_classifier)

    @resources(memory=16000)
    @step
    def train_classifier(self):
        """Train a classifier using the TS2Vec representations."""
        set_seed(self.seed)
        print(f"Training classifier on device: {self.device}")
        print(f"Using {self.label_fraction * 100:.1f}% of labeled training data.")

        # --- Subset the training data ---
        num_train_samples = len(self.train_repr)
        indices = np.arange(num_train_samples)
        np.random.shuffle(indices) # Shuffle indices randomly (but deterministically due to set_seed)
        
        subset_size = int(num_train_samples * self.label_fraction)
        if subset_size == 0 and self.label_fraction > 0:
             subset_size = 1 # Ensure at least one sample if fraction > 0
        elif subset_size > num_train_samples:
             subset_size = num_train_samples # Cap at the total number of samples
             
        subset_indices = indices[:subset_size]
        
        train_repr_subset = self.train_repr[subset_indices]
        y_train_subset = self.y_train[subset_indices]
        print(f"Classifier training subset size: {len(train_repr_subset)}")

        # For the classifier, we use the size of the TS2Vec output features.
        feature_dim = train_repr_subset.shape[-1]
        self.classifier = SimpleClassifier(input_dim=feature_dim).to(self.device)
        loss_fn = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(self.classifier.parameters(), lr=self.classifier_lr)
        
        # Create TensorDatasets for train and validation sets
        train_dataset = TensorDataset(torch.from_numpy(train_repr_subset).float(), 
                                      torch.from_numpy(y_train_subset).float())
        val_dataset = TensorDataset(torch.from_numpy(self.val_repr).float(), 
                                    torch.from_numpy(self.y_val).float())
        
        train_loader = DataLoader(train_dataset, batch_size=self.classifier_batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.classifier_batch_size, shuffle=False)
        
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        with mlflow.start_run(run_id=self.mlflow_run_id):
            # Log classifier hyperparameters
            params = {
                "classifier_lr": self.classifier_lr,
                "classifier_epochs": self.classifier_epochs,
                "classifier_batch_size": self.classifier_batch_size,
                "label_fraction": self.label_fraction
            }
            mlflow.log_params(params)
            
            # Training loop for the classifier
            for epoch in range(1, self.classifier_epochs + 1):
                self.classifier.train()
                running_loss = 0.0
                for features, labels in train_loader:
                    features, labels = features.to(self.device), labels.to(self.device)
                    optimizer.zero_grad()
                    outputs = self.classifier(features).squeeze()
                    loss = loss_fn(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item() * features.size(0)
                epoch_loss = running_loss / len(train_loader.dataset)
                print(f"Classifier Epoch {epoch}/{self.classifier_epochs}, Loss: {epoch_loss:.4f}")
                mlflow.log_metric("classifier_train_loss", epoch_loss, step=epoch)
                
                # Evaluate on validation set
                self.classifier.eval()
                correct = 0
                total = 0
                with torch.no_grad():
                    for features, labels in val_loader:
                        features, labels = features.to(self.device), labels.to(self.device)
                        outputs = self.classifier(features).squeeze()
                        # using sigmoid to get probabilities
                        preds = (torch.sigmoid(outputs) > 0.5).float()
                        correct += (preds == labels).sum().item()
                        total += labels.size(0)
                val_acc = correct / total
                print(f"Validation Accuracy: {val_acc:.4f}")
                mlflow.log_metric("classifier_val_accuracy", val_acc, step=epoch)
            
        self.next(self.evaluate)

    @step
    def evaluate(self):
        """Evaluate the classifier performance on the test data."""
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        print(f"Evaluating classifier on device: {self.device}")
        test_dataset = TensorDataset(torch.from_numpy(self.test_repr).float(), 
                                     torch.from_numpy(self.y_test).float())
        test_loader = DataLoader(test_dataset, batch_size=self.classifier_batch_size, shuffle=False)
        
        with mlflow.start_run(run_id=self.mlflow_run_id):
            self.classifier.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for features, labels in test_loader:
                    features, labels = features.to(self.device), labels.to(self.device)
                    outputs = self.classifier(features).squeeze()
                    preds = (torch.sigmoid(outputs) > 0.5).float()
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)
            self.test_accuracy = correct / total
            print(f"Final Test Accuracy: {self.test_accuracy:.4f}")
            mlflow.log_metric("classifier_test_accuracy", self.test_accuracy)
        self.next(self.register)

    @step
    def register(self):
        """Register the classifier model if the test accuracy is above the threshold."""
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        if self.test_accuracy >= self.accuracy_threshold:
            self.registered = True
            logging.info("Registering classifier model...")
            with mlflow.start_run(run_id=current.run_id):
                # Create a sample signature from a batch of test representations
                sample_input = torch.from_numpy(self.test_repr[:5]).float()
                mlflow.pytorch.log_model(
                    self.classifier,
                    artifact_path="classifier_model",
                    registered_model_name="ts2vec_classifier",
                )
                print("Classifier model successfully registered!")
        else:
            self.registered = False
            print(f"Test accuracy ({self.test_accuracy:.4f}) below threshold ({self.accuracy_threshold})—model not registered.")
        self.next(self.end)

    @step
    def end(self):
        """Finalize the pipeline."""
        print("=== TS2Vec Training and Classifier Pipeline Complete ===")
        print(f"Final Test Accuracy: {self.test_accuracy:.4f}")
        print("Done!")

if __name__ == "__main__":
    ECGTS2VecFlow()
