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

# Swap in Soft TS2Vec and utilities
from ts2vec_soft import TS2Vec_soft, LinearClassifier, save_sim_mat, densify, train_linear_classifier, evaluate_classifier

from torch_utilities import load_processed_data, split_data_by_participant, set_seed

# -------------------------
# Metaflow Pipeline for Soft TS2Vec pretraining and classifier fine-tuning
# -------------------------
@project(name="ecg_training_ts2vec_soft")
class ECGTS2VecFlow(FlowSpec):
    # MLflow and data parameters
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

    # General parameters
    seed = Parameter("seed", help="Random seed", default=42)

    # Soft TS2Vec pretraining hyperparameters
    ts2vec_epochs = Parameter("ts2vec_epochs", help="Epochs for TS2Vec pretraining", default=50)
    ts2vec_lr = Parameter("ts2vec_lr", help="Learning rate for TS2Vec", default=0.001)
    ts2vec_batch_size = Parameter("ts2vec_batch_size", help="Batch size for TS2Vec", default=8)
    ts2vec_output_dims = Parameter("ts2vec_output_dims", help="Representation dimension (Co)", default=320)
    ts2vec_hidden_dims = Parameter("ts2vec_hidden_dims", help="Hidden dimension (Ch)", default=64)
    ts2vec_depth = Parameter("ts2vec_depth", help="Depth (# dilated conv blocks)", default=10)
    ts2vec_max_train_length = Parameter("ts2vec_max_train_length", help="Max training length", default=5000)
    ts2vec_temporal_unit = Parameter("ts2vec_temporal_unit", help="Temporal unit for hierarchical pooling", default=0)

    # Soft contrastive learning hyperparameters
    ts2vec_dist_type = Parameter(
        "ts2vec_dist_type",
        help="Distance metric for soft labels (DTW, EUC, COS, TAM, GAK)",
        default="EUC"
    )
    ts2vec_tau_inst = Parameter(
        "ts2vec_tau_inst",
        help="Temperature parameter tau_inst for soft instance CL", default=50.0
    )
    ts2vec_tau_temp = Parameter(
        "ts2vec_tau_temp",
        help="Temperature parameter tau_temp for soft temporal CL", default=2.5
    )
    ts2vec_alpha = Parameter(
        "ts2vec_alpha",
        help="Alpha for densification of soft labels", default=0.5
    )
    ts2vec_lambda = Parameter(
        "ts2vec_lambda",
        help="Weight lambda for instance vs temporal CL", default=0.5
    )

    # Classifier training hyperparameters
    classifier_epochs = Parameter("classifier_epochs", help="Epochs for classifier training", default=25)
    classifier_lr = Parameter("classifier_lr", help="Learning rate for classifier", default=0.0001)
    classifier_batch_size = Parameter("classifier_batch_size", help="Batch size for classifier", default=32)
    accuracy_threshold = Parameter("accuracy_threshold", help="Minimum accuracy for model registration", default=0.74)
    label_fraction = Parameter(
        "label_fraction",
        help="Fraction of labeled training data for classifier (0.01-1.0)",
        default=1.0
    )

    @step
    def start(self):
        """Initialize MLflow and set seed."""
        set_seed(self.seed)
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        logging.info("MLflow tracking server: %s", self.mlflow_tracking_uri)
        # Set a custom MLflow experiment name
        mlflow.set_experiment("SoftTS2Vec")
        try:
            run = mlflow.start_run(run_name=current.run_id)
            self.mlflow_run_id = run.info.run_id
        except Exception as e:
            raise RuntimeError(f"MLflow connection failed: {str(e)}")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        self.next(self.preprocess_data)

    @step
    def preprocess_data(self):
        """Load or generate the windowed data for training."""
        X, y, groups = load_processed_data(
            hdf5_path=self.window_data_path,
            label_map={"baseline": 0, "mental_stress": 1}
        )
        print(f"Windowed data loaded: X.shape={X.shape}, y.shape={y.shape}")   
    
        (self.X_train, self.y_train), (self.X_val, self.y_val), (self.X_test, self.y_test) = split_data_by_participant(
            X, y, groups
        )
        print(f"Train samples: {len(self.X_train)}")
        self.next(self.train_ts2vec)

    @resources(memory=16000)
    @step
    def train_ts2vec(self):
        """
        If a TS2Vec encoder with identical hyper-parameters is already stored in
        MLflow (SoftTS2Vec experiment), load it. Otherwise train and log it.
        """
        import tempfile, os, mlflow
        from mlflow.tracking import MlflowClient

        set_seed(self.seed)
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        mlflow.set_experiment("SoftTS2Vec")          # same as in start()

        # ------------------------------------------------------------------
        # Build the fingerprint that defines *one unique encoder*
        # ------------------------------------------------------------------
        fp = {
            "model_name": "TS2Vec_soft",
            "seed": self.seed,
            "ts2vec_epochs": self.ts2vec_epochs,
            "ts2vec_output_dims": self.ts2vec_output_dims,
            "ts2vec_hidden_dims": self.ts2vec_hidden_dims,
            "ts2vec_depth": self.ts2vec_depth,
            "ts2vec_dist_type": self.ts2vec_dist_type,
            "ts2vec_tau_inst": self.ts2vec_tau_inst,
            "ts2vec_tau_temp": self.ts2vec_tau_temp,
            "ts2vec_alpha": self.ts2vec_alpha,
            "ts2vec_lambda": self.ts2vec_lambda,
            "ts2vec_max_train_length": self.ts2vec_max_train_length,
        }
        # convert everything to str for MLflow query
        fp_str = {k: str(v) for k, v in fp.items()}

        # ------------------------------------------------------------------
        # Search MLflow
        # ------------------------------------------------------------------
        client = MlflowClient()
        exp = client.get_experiment_by_name("SoftTS2Vec")
        match_run = None
        if exp:
            filter_parts = ["attributes.status = 'FINISHED'"]
            filter_parts += [f"params.{k} = '{v}'" for k, v in fp_str.items()]
            query = " and ".join(filter_parts)

            hits = mlflow.search_runs([exp.experiment_id],
                                    filter_string=query, max_results=1)
            if not hits.empty:
                match_run = hits.iloc[0]["run_id"]

        # ------------------------------------------------------------------
        # Load or train
        # ------------------------------------------------------------------
        if match_run is not None:
            print(f"Re-using encoder from run {match_run}")
            model_uri = f"runs:/{match_run}/ts2vec_model"
            self.ts2vec_soft = mlflow.pytorch.load_model(model_uri,
                                                        map_location=self.device)

        else:
            print("No compatible encoder found – training from scratch")
            # ---- create encoder ----
            self.ts2vec_soft = TS2Vec_soft(
                input_dims=self.X_train.shape[2],
                output_dims=self.ts2vec_output_dims,
                hidden_dims=self.ts2vec_hidden_dims,
                depth=self.ts2vec_depth,
                device=self.device,
                lr=self.ts2vec_lr,
                batch_size=self.ts2vec_batch_size,
                lambda_=self.ts2vec_lambda,
                tau_temp=self.ts2vec_tau_temp,
                max_train_length=self.ts2vec_max_train_length,
                temporal_unit=self.ts2vec_temporal_unit,
                soft_instance=(self.ts2vec_tau_inst > 0),
                soft_temporal=(self.ts2vec_tau_temp > 0)
            )

            # Compute soft labels matrix if instance soft CL is enabled
            if self.ts2vec_tau_inst > 0:
                print("Computing similarity matrix for soft instance contrastive learning...")
                # For univariate series, squeeze last dim
                if self.X_train.ndim == 3 and self.X_train.shape[2] == 1:
                    X_flat = self.X_train.squeeze(-1)
                    multivariate = False
                else:
                    # Flatten multivariate or select first channel
                    X_flat = self.X_train.reshape(self.X_train.shape[0], self.X_train.shape[1], -1)
                    X_flat = X_flat[..., 0]
                    multivariate = False
                # compute normalized similarity matrix
                sim_mat = save_sim_mat(
                    X_flat,
                    min_=0,
                    max_=1,
                    multivariate=multivariate,
                    type_=self.ts2vec_dist_type,
                )
                # convert to distance
                dist_mat = 1 - sim_mat
                # densify to get soft labels
                soft_labels = densify(-dist_mat, self.ts2vec_tau_inst, self.ts2vec_alpha)
            else:
                soft_labels = None
            
            # Expand the soft‑label matrix to match the splits
            if self.ts2vec_tau_inst > 0 and self.ts2vec_max_train_length is not None:
                S = self.X_train.shape[1] // self.ts2vec_max_train_length
                if S > 1:
                    # tile both axes
                    soft_labels = np.repeat(
                        np.repeat(soft_labels, repeats=S, axis=0),
                        repeats=S, axis=1
                    )

            # Log and run pretraining
            mlflow.set_tracking_uri(self.mlflow_tracking_uri)
            with mlflow.start_run(run_id=self.mlflow_run_id):
                mlflow.log_params(fp_str)

                run_dir = tempfile.mkdtemp(prefix="ts2vec_")
                self.ts2vec_soft.fit(
                    self.X_train, soft_labels,
                    run_dir=run_dir,
                    n_epochs=self.ts2vec_epochs,
                    verbose=True
                )
                # log pytorch model to MLflow
                mlflow.pytorch.log_model(
                    pytorch_model=self.ts2vec_soft,
                    artifact_path="ts2vec_model"
                )

        self.next(self.extract_representations)

    @step
    def extract_representations(self):
        """Extract feature representations using the trained TS2Vec encoder."""
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        self.train_repr = self.ts2vec_soft.encode(self.X_train, encoding_window="full_series")
        self.val_repr = self.ts2vec_soft.encode(self.X_val, encoding_window="full_series")
        self.test_repr = self.ts2vec_soft.encode(self.X_test, encoding_window="full_series")
        print(f"Extracted TS2Vec representations: train_repr shape={self.train_repr.shape}")
        self.next(self.train_classifier)

    @resources(memory=16000)
    @step
    def train_classifier(self):
        set_seed(self.seed)
        print(f"Training classifier on device: {self.device}")
        print(f"Using {self.label_fraction * 100:.1f}% of labeled training data.")

        # Subset training data
        indices = np.arange(len(self.train_repr))
        np.random.shuffle(indices)
        subset_size = int(len(indices) * self.label_fraction)
        subset_size = max(1, min(subset_size, len(indices))) if self.label_fraction>0 else 0
        subset_indices = indices[:subset_size]

        train_repr_subset = self.train_repr[subset_indices]
        y_train_subset = self.y_train[subset_indices]

        feature_dim = train_repr_subset.shape[-1]
        self.classifier = LinearClassifier(input_dim=feature_dim).to(self.device)
        loss_fn = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(self.classifier.parameters(), lr=self.classifier_lr)

        train_dataset = TensorDataset(
            torch.from_numpy(train_repr_subset).float(),
            torch.from_numpy(y_train_subset).float()
        )
        val_dataset = TensorDataset(
            torch.from_numpy(self.val_repr).float(),
            torch.from_numpy(self.y_val).float()
        )
        train_loader = DataLoader(train_dataset, batch_size=self.classifier_batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.classifier_batch_size, shuffle=False)

        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        with mlflow.start_run(run_id=self.mlflow_run_id):
            params = {
                "classifier_lr": self.classifier_lr,
                "classifier_epochs": self.classifier_epochs,
                "classifier_batch_size": self.classifier_batch_size,
                "label_fraction": self.label_fraction
            }
            mlflow.log_params(params)
            
            classifier, val_accuracies, val_aurocs, val_pr_aucs, val_f1_scores = train_linear_classifier(
                model=self.classifier,
                train_loader=train_loader,
                val_loader=val_loader,
                loss_fn=loss_fn,
                optimizer=optimizer,
                epochs=self.classifier_epochs,
                device=self.device
            )

        print("Classifier training complete.")
        self.next(self.evaluate)

    @step
    def evaluate(self):
        """Evaluate the classifier performance on the test data."""
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        test_dataset = TensorDataset(
            torch.from_numpy(self.test_repr).float(),
            torch.from_numpy(self.y_test).float()
        )
        test_loader = DataLoader(test_dataset, batch_size=self.classifier_batch_size, shuffle=False)

        with mlflow.start_run(run_id=self.mlflow_run_id):
            self.test_accuracy, test_auroc, test_pr_auc, test_f1 = evaluate_classifier(
                model=self.classifier,
                test_loader=test_loader,
                device=self.device
            )
            
        self.next(self.register)

    @step
    def register(self):
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        if self.test_accuracy >= self.accuracy_threshold:
            with mlflow.start_run(run_id=current.run_id):
                mlflow.pytorch.log_model(
                    self.classifier,
                    artifact_path="classifier_model",
                    registered_model_name="ts2vec_classifier_soft"
                )
            self.registered = True
        else:
            self.registered = False
        self.next(self.end)

    @step
    def end(self):
        print("=== Soft TS2Vec Training and Classifier Pipeline Complete ===")
        print(f"Final Test Accuracy: {self.test_accuracy:.4f}")
        print("Done!")

if __name__ == "__main__":
    ECGTS2VecFlow()
