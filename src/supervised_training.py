import os, tempfile, logging, json
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
import mlflow, mlflow.pytorch
from mlflow.tracking import MlflowClient
from torch.utils.data import DataLoader, TensorDataset
from metaflow import FlowSpec, step, Parameter, current, project, resources

from torch_utilities import (             
    load_processed_data,
    split_indices_by_participant,
    build_supervised_fingerprint,
    search_encoder_fp,
    ECGDataset,
    set_seed,
    train,
    test,
    EarlyStopping,
    Improved1DCNN_v2,
    TCNClassifier,
    TransformerECGClassifier,
)

#  Metaflow pipeline
@project(name="ecg_training_supervised")
class ECGSupervisedFlow(FlowSpec):

    # generic parameters
    mlflow_tracking_uri = Parameter("mlflow_tracking_uri",
        default=os.getenv("MLFLOW_TRACKING_URI", "https://127.0.0.1:5000")
    )
    window_data_path = Parameter("window_data_path",
        default="../data/interim/windowed_data.h5"
    )
    seed = Parameter("seed", default=42)
    
    model_type  = Parameter("model_type", help= "cnn, tcn or transformer",
        default="cnn"
    )

    # training hyper-parameters
    lr = Parameter("lr", default=1e-4)
    batch_size = Parameter("batch_size", default=32)
    num_epochs = Parameter("num_epochs", default=25)
    patience = Parameter("patience",default=5)
    scheduler_mode = Parameter("scheduler_mode", default="min")
    scheduler_factor = Parameter("scheduler_factor", default=0.1)
    scheduler_patience = Parameter("scheduler_patience", default=2)
    scheduler_min_lr = Parameter("scheduler_min_lr", default=1e-11)

    # Metaflow steps
    @step
    def start(self):
        """Set seed, choose MLflow experiment name, open run."""
        set_seed(self.seed)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        exp_map = {
            "cnn": "Supervised_CNN",
            "tcn": "Supervised_TCN",
            "transformer": "Supervised_Transformer",
        }
        self.experiment_name = exp_map.get(self.model_type.lower())
        if self.experiment_name is None:
            raise ValueError(f"Unknown model_type '{self.model_type}'")

        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        try:
            run = mlflow.start_run(run_name=current.run_id)
            self.mlflow_run_id = run.info.run_id
        except Exception as e:
            raise RuntimeError(f"MLflow connection failed: {str(e)}")

        print(f"Starting simple training pipeline... MLflow experiment: {self.experiment_name}")
        self.next(self.load_and_split)

    @step
    def load_and_split(self):
        """Load windowed data and produce participant-level train/val/test idx."""
        X, y, groups = load_processed_data(
            self.window_data_path,
            label_map={"baseline": 0, "mental_stress": 1}
        )
        self.n_features = X.shape[2]
        self.y = y.astype(np.float32)

        tr_idx, val_idx, te_idx = split_indices_by_participant(groups, seed=0)
        self.train_idx, self.val_idx, self.test_idx = tr_idx, val_idx, te_idx
        print(f"windows: train {len(tr_idx)}, val {len(val_idx)}, test {len(te_idx)}")
        self.next(self.train_model)

    @resources(memory=16000)
    @step
    def train_model(self):
        """
        Train (or load if same fingerprint) the supervised model specified by 'model_type'.
        """
        set_seed(self.seed)
        X, _, _ = load_processed_data(self.window_data_path)

        # build fingerprint and MLflow lookup
        self.fp = build_supervised_fingerprint({
            "model_name": self.model_type.lower(),
            "seed": self.seed,
            "lr": self.lr,
            "batch_size": self.batch_size,
            "num_epochs": self.num_epochs,
            "patience": self.patience,
            "scheduler_mode": self.scheduler_mode,
            "scheduler_factor": self.scheduler_factor,
            "scheduler_patience": self.scheduler_patience,
            "scheduler_min_lr": self.scheduler_min_lr,
        })
        run_id = search_encoder_fp(self.fp, self.experiment_name, self.mlflow_tracking_uri)

        if run_id is not None:
            # Re-use existing model
            print(f"Found matching run {run_id}: loading model.")
            uri = f"runs:/{run_id}/supervised_model"
            self.model = mlflow.pytorch.load_model(uri, map_location=self.device)

        else:
            # train from scratch
            print("No matching run: training from scratch.")
            # build datasets / loaders
            tr_ds = ECGDataset(X[self.train_idx], self.y[self.train_idx])
            va_ds = ECGDataset(X[self.val_idx],   self.y[self.val_idx])

            num_workers = min(8, os.cpu_count() or 2)
            tr_loader = DataLoader(tr_ds, self.batch_size, shuffle=True,
                                   num_workers=num_workers, pin_memory=True)
            va_loader = DataLoader(va_ds, self.batch_size, shuffle=False,
                                   num_workers=num_workers, pin_memory=True)

            # choose model
            if self.model_type.lower() == "cnn":
                self.model = Improved1DCNN_v2().to(self.device)
            elif self.model_type.lower() == "tcn":
                self.model = TCNClassifier().to(self.device)
            elif self.model_type.lower() == "transformer":
                self.model = TransformerECGClassifier().to(self.device)
            else:
                raise ValueError(f"Unknown model_type '{self.model_type}'")

            loss_fn = nn.BCEWithLogitsLoss()
            optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode=self.scheduler_mode, factor= self.scheduler_factor,
                patience=self.scheduler_patience, min_lr=self.scheduler_min_lr, verbose=False
            )
            es = EarlyStopping(self.patience)

            mlflow.set_tracking_uri(self.mlflow_tracking_uri)
            with mlflow.start_run(run_id=self.mlflow_run_id):
                mlflow.log_params(self.fp | {
                    "optimizer": "Adam",
                    "loss_fn":   "BCEWithLogitsLoss",
                })

                # Training loop with validation
                for ep in range(1, self.num_epochs+1):
                    print(f"\nEpoch {ep}/{self.num_epochs}")
                    
                    # Train and log metrics
                    train_loss, train_acc, train_auc, train_pr_auc, train_f1 = train(
                        self.model, tr_loader, optimizer, loss_fn,
                        self.device, epoch=ep
                    )
                    
                    # Validate and log metrics
                    val_loss, val_acc, val_auc, val_pr_auc, val_f1 = test(
                        self.model, va_loader, loss_fn,
                        self.device, phase='val', epoch=ep
                    )

                    scheduler.step(val_loss)
                    es(val_loss)
                    if es.early_stop:
                        print("Early stopping triggered")
                        break

                # store model artifact
                mlflow.pytorch.log_model(
                    self.model, artifact_path="supervised_model"
                )
                
        self.next(self.evaluate)

    @step
    def evaluate(self):
        """Test-set evaluation."""
        set_seed(self.seed)
        X, _, _ = load_processed_data(self.window_data_path)
        te_ds = ECGDataset(X[self.test_idx], self.y[self.test_idx])
        num_workers = min(8, os.cpu_count() or 2)
        self.test_loader = DataLoader(te_ds, self.batch_size, shuffle=False,
                                          num_workers=num_workers, pin_memory=True)
        
        loss_fn = nn.BCEWithLogitsLoss()
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        with mlflow.start_run(run_id=self.mlflow_run_id):
            mlflow.log_params(self.fp | {
                    "optimizer": "Adam",
                    "loss_fn":   "BCEWithLogitsLoss",
                })
            test_loss, self.test_accuracy, test_auc, test_pr_auc, test_f1  = test(
                self.model, self.test_loader, loss_fn,
                self.device, phase='test'
            )

        print(f"Test accuracy: {self.test_accuracy:.3f}")
        self.next(self.end)

    @step
    def end(self):
        print(f"Supervised training flow with {self.model_type} complete.")
        print(f"Test accuracy: {self.test_accuracy:.4f}")
        mlflow.end_run() 
        print("Done!")

if __name__ == "__main__":
    ECGSupervisedFlow()