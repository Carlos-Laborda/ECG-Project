import os, logging, numpy as np, torch, torch.nn as nn, torch.optim as optim
import mlflow, mlflow.pytorch
from torch.utils.data import DataLoader, TensorDataset
from metaflow import FlowSpec, step, Parameter, current, project, resources

from models.ts2vec_soft import TS2Vec_soft
from models.ts2vec_soft import save_sim_mat, densify, train_linear_classifier, evaluate_classifier, load_ppg_windows
from training_pipelines.torch_utilities import load_processed_data, split_data_by_participant, set_seed
from models.supervised import (
    LinearClassifier,
    MLPClassifier,
)

@project(name="ppg2ecg_ts2vec_soft")
class PPG2ECG_TS2VecFlow(FlowSpec):

    # MLflow + DATA PATHS 
    mlflow_tracking_uri = Parameter("mlflow_tracking_uri",
        default=os.getenv("MLFLOW_TRACKING_URI", "https://127.0.0.1:5000"))
    ppg_window_path = Parameter("ppg_window_path",
        help="Path to windowed PPG data",
        default="../data/interim/ppg_windows.h5")               
    ecg_window_path = Parameter("ecg_window_path",
        help="Path to windowed ECG data",
        default="../data/interim/windowed_data.h5")      

    seed = Parameter("seed", default=42)

    # TS2Vec hyper-params 
    ts2vec_epochs          = Parameter("ts2vec_epochs", default=50)
    ts2vec_lr              = Parameter("ts2vec_lr", default=1e-3)
    ts2vec_batch_size      = Parameter("ts2vec_batch_size", default=8) 
    ts2vec_output_dims     = Parameter("ts2vec_output_dims", default=320)
    ts2vec_hidden_dims     = Parameter("ts2vec_hidden_dims", default=64)
    ts2vec_depth           = Parameter("ts2vec_depth", default=10)
    ts2vec_max_train_length= Parameter("ts2vec_max_train_length", default=None)
    ts2vec_temporal_unit   = Parameter("ts2vec_temporal_unit", default=0)

    # Soft-contrastive hyper-params 
    ts2vec_dist_type = Parameter("ts2vec_dist_type", default="EUC")
    ts2vec_tau_inst  = Parameter("ts2vec_tau_inst", default=50.0)
    ts2vec_tau_temp  = Parameter("ts2vec_tau_temp", default=2.5)
    ts2vec_alpha     = Parameter("ts2vec_alpha",    default=0.5)
    ts2vec_lambda    = Parameter("ts2vec_lambda",   default=0.5)

    # Classifier hyper-params 
    classifier_epochs      = Parameter("classifier_epochs", default=25)
    classifier_lr          = Parameter("classifier_lr", default=1e-4)
    classifier_batch_size  = Parameter("classifier_batch_size", default=32)
    accuracy_threshold     = Parameter("accuracy_threshold", default=0.74)
    label_fraction         = Parameter("label_fraction", default=1.0)

    @step
    def start(self):
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
        self.next(self.load_data)

    @resources(memory=16000)
    @step
    def load_data(self):
        """Load PPG windows (unlabeled) + ECG windows (labeled)."""
        print("Loading PPG windows…")
        self.X_ppg, _ = load_ppg_windows(self.ppg_window_path)          
        print(f"PPG windows: {self.X_ppg.shape}")

        print("Loading ECG windows + labels…")
        X_ecg, y_ecg, groups = load_processed_data(
            hdf5_path=self.ecg_window_path,
            label_map={"baseline": 0, "mental_stress": 1}
        )
        (self.X_train_ecg, self.y_train), \
        (self.X_val_ecg,  self.y_val), \
        (self.X_test_ecg, self.y_test) = split_data_by_participant(X_ecg, y_ecg, groups)
        print(f"ECG train windows: {self.X_train_ecg.shape}")
        self.next(self.train_ts2vec)

    @resources(memory=16000)
    @step
    def train_ts2vec(self):
        """Pre-train TS2Vec-Soft on PPG windows only."""
        input_dims = 1                    
        # soft-label matrix (instance-wise)
        tau_inst  = self.ts2vec_tau_inst
        if tau_inst > 0:
            print("Computing soft-label matrix (may take a while)…")
            sim_mat   = save_sim_mat(self.X_ppg.squeeze(-1), type_=self.ts2vec_dist_type,
                                     min_=0, max_=1, multivariate=False)
            soft_lab  = densify(- (1 - sim_mat), tau_inst, self.ts2vec_alpha)
        else:
            soft_lab  = None

        self.ts2vec_soft_ppg = TS2Vec_soft(
            input_dims=input_dims,
            output_dims=self.ts2vec_output_dims,
            hidden_dims=self.ts2vec_hidden_dims,
            depth=self.ts2vec_depth,
            device=self.device,
            lr=self.ts2vec_lr,
            batch_size=self.ts2vec_batch_size,
            lambda_=self.ts2vec_lambda,
            tau_temp=self.ts2vec_tau_temp,
            temporal_unit=self.ts2vec_temporal_unit,
            soft_instance=(tau_inst>0),
            soft_temporal=(self.ts2vec_tau_temp>0)
        )
        # Log and run pretraining
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        with mlflow.start_run(run_id=self.mlflow_run_id):
            params = {
                "model_name": self.ts2vec_soft_ppg.__class__.__name__, 
                "ts2vec_epochs": self.ts2vec_epochs,
                "ts2vec_lr": self.ts2vec_lr,
                "ts2vec_batch_size": self.ts2vec_batch_size,
                "ts2vec_output_dims": self.ts2vec_output_dims,
                "ts2vec_hidden_dims": self.ts2vec_hidden_dims,
                "ts2vec_depth": self.ts2vec_depth,
                "ts2vec_max_train_length": self.ts2vec_max_train_length,
                "ts2vec_temporal_unit": self.ts2vec_temporal_unit,
                # soft CL params
                "ts2vec_dist_type": self.ts2vec_dist_type,
                "ts2vec_tau_inst": self.ts2vec_tau_inst,
                "ts2vec_tau_temp": self.ts2vec_tau_temp,
                "ts2vec_alpha": self.ts2vec_alpha,
                "ts2vec_lambda": self.ts2vec_lambda,
            }
            mlflow.log_params(params)
        
            run_dir = f"ts2vec_soft_ppg_{self.mlflow_run_id}"
            os.makedirs(run_dir, exist_ok=True)
            self.ts2vec_soft_ppg.fit(self.X_ppg, soft_lab, run_dir,
                            n_epochs=self.ts2vec_epochs, verbose=True)
            self.encoder_path = f"{run_dir}/encoder.pth"
            self.ts2vec_soft_ppg.save(self.encoder_path)
            mlflow.log_artifact(self.encoder_path, artifact_path="ts2vec_encoder")
        self.next(self.extract_representations)

    @step
    def extract_representations(self):
        """Encode ECG windows with the frozen PPG-trained encoder."""
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        self.train_repr = self.ts2vec_soft_ppg.encode(self.X_train_ecg, encoding_window="full_series")
        self.val_repr   = self.ts2vec_soft_ppg.encode(self.X_val_ecg,   encoding_window="full_series")
        self.test_repr  = self.ts2vec_soft_ppg.encode(self.X_test_ecg,  encoding_window="full_series")
        print("Representations extracted:", self.train_repr.shape)
        self.next(self.train_classifier)

    @resources(memory=16000)
    @step
    def train_classifier(self):
        set_seed(self.seed)
        idx = np.random.permutation(len(self.train_repr))[: int(len(self.train_repr)*self.label_fraction)]
        feat_dim = self.train_repr.shape[-1]
        self.classifier = LinearClassifier(input_dim=feat_dim).to(self.device)
        opt = optim.AdamW(self.classifier.parameters(), lr=self.classifier_lr)
        loss_fn = nn.BCEWithLogitsLoss()

        train_ds = TensorDataset(torch.from_numpy(self.train_repr[idx]).float(),
                                 torch.from_numpy(self.y_train[idx]).float())
        val_ds   = TensorDataset(torch.from_numpy(self.val_repr).float(),
                                 torch.from_numpy(self.y_val).float())
        train_loader = DataLoader(train_ds, batch_size=self.classifier_batch_size, shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=self.classifier_batch_size)
        
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        with mlflow.start_run(run_id=self.mlflow_run_id):
            params = {
                "classifier_lr": self.classifier_lr,
                "classifier_epochs": self.classifier_epochs,
                "classifier_batch_size": self.classifier_batch_size,
                "label_fraction": self.label_fraction
            }
            mlflow.log_params(params)

            self.classifier, _, _, _, _ = train_linear_classifier(
                self.classifier, train_loader, val_loader, opt, loss_fn,
                self.classifier_epochs, self.device
            )
        print("Classifier training complete.")
        self.next(self.evaluate)

    @step
    def evaluate(self):
        """Evaluate the classifier performance on the test data."""
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        test_ds = TensorDataset(torch.from_numpy(self.test_repr).float(),
                                torch.from_numpy(self.y_test).float())
        test_loader = DataLoader(test_ds, batch_size=self.classifier_batch_size)
        with mlflow.start_run(run_id=self.mlflow_run_id):
            self.test_accuracy, *_ = evaluate_classifier(self.classifier, test_loader, self.device)
        self.next(self.register)

    @step
    def register(self):
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        if self.test_accuracy >= self.accuracy_threshold:
            with mlflow.start_run(run_id=current.run_id):
                mlflow.pytorch.log_model(self.classifier,
                    artifact_path="classifier_model",
                    registered_model_name="PPG-pretrained_TS2Vec_classifier_soft")
            self.registered = True
        else:
            self.registered = False
        self.next(self.end)

    @step
    def end(self):
        print("=== PPG→ECG TS2Vec-Soft Pipeline finished ===")
        print(f"Test accuracy: {self.test_accuracy:.4f}")

if __name__ == "__main__":
    PPG2ECG_TS2VecFlow()
