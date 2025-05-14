import os, mlflow, torch, numpy as np, torch.nn as nn, torch.optim as optim
import mlflow.pytorch
from torch.utils.data import TensorDataset, DataLoader
from metaflow import FlowSpec, step, Parameter, current, project, resources
from torch_utilities import (load_processed_data, split_data_by_participant,
                             set_seed)

from simclr import (get_simclr_model, NTXentLoss, simclr_data_loaders, pretrain_one_epoch,
                    encode_representations, LinearClassifier, train_linear_classifier,
                    evaluate_classifier, show_shape)

@project(name="ecg_training_simclr")
class ECGSimCLRFlow(FlowSpec):

    # pipeline parameters
    mlflow_tracking_uri = Parameter("mlflow_tracking_uri",
                                    default=os.getenv("MLFLOW_TRACKING_URI",
                                                      "https://127.0.0.1:5000"))
    window_data_path = Parameter("window_data_path",
                                    default="../data/interim/windowed_data.h5")
    seed = Parameter("seed", default=42)

    # SimCLR pre‑training
    epochs = Parameter("epochs", default=100)
    lr = Parameter("lr",default=1e-3)
    batch_size = Parameter("batch_size", default=128)
    temperature = Parameter("temperature", default=0.5)

    # linear classifier
    clf_epochs = Parameter("clf_epochs", default=25)
    clf_lr = Parameter("clf_lr",       default=1e-4)
    clf_batch_size = Parameter("clf_batch_size", default=32)
    label_fraction = Parameter("label_fraction", default=1.0)
    accuracy_threshold = Parameter("accuracy_threshold", default=0.74)

    @step
    def start(self):
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        try:
            run = mlflow.start_run(run_name=current.run_id)
            self.mlflow_run_id = run.info.run_id
        except Exception as e:
            raise RuntimeError(f"MLflow connection failed: {e}")
        set_seed(self.seed)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        self.next(self.preprocess_data)

    @step
    def preprocess_data(self):
        X, y, groups = load_processed_data(
            hdf5_path=self.window_data_path,
            label_map={"baseline": 0, "mental_stress": 1}
        )
        print(f"Windowed data loaded: X.shape={X.shape}, y.shape={y.shape}")
        (self.X_train, self.y_train), (self.X_val, self.y_val), (self.X_test, self.y_test) = \
            split_data_by_participant(X, y, groups)
        print(f"Train samples: {len(self.X_train)}")
        print(f"[DBG] preprocess_data  train/val/test shapes:",
              self.X_train.shape, self.X_val.shape, self.X_test.shape)
        self.next(self.pretrain_simclr)

    @resources(memory=16000)
    @step
    def pretrain_simclr(self):
        model = get_simclr_model(window=self.X_train.shape[-1],
                                 device=self.device)
        loss_fn = NTXentLoss(self.batch_size, self.temperature)
        opt     = optim.AdamW(model.parameters(), lr=self.lr, weight_decay=1e-4)
        train_dl, val_dl = simclr_data_loaders(self.X_train, self.X_val,
                                               self.batch_size)
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        with mlflow.start_run(run_id=self.mlflow_run_id):
            params = {
                "epochs": self.epochs, "lr": self.lr,
                "batch_size": self.batch_size,
                "temperature": self.temperature}
            mlflow.log_params(params)

            for ep in range(1, self.epochs+1):
                tr_loss = pretrain_one_epoch(
                    model, train_dl, loss_fn, opt, self.device)
                mlflow.log_metric("ssl_train_loss", tr_loss, step=ep)
                if ep % 50 == 0 or ep==1:
                    print(f"Epoch {ep}/{self.epochs}: loss={tr_loss:.4f}")

            # save ssl checkpoint
            ckpt = f"simclr_encoder_{current.run_id}.pt"
            torch.save(model.state_dict(), ckpt)
            mlflow.log_artifact(ckpt, "ssl_model")
            self.ssl_model_state = ckpt
            self.simclr_model = model  
        self.next(self.extract_representations)

    @step
    def extract_representations(self):
        # freeze encoder for linear evaluation
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        with torch.no_grad():
            self.train_repr = encode_representations(
                self.simclr_model, self.X_train,
                self.batch_size, self.device)
            self.val_repr   = encode_representations(
                self.simclr_model, self.X_val,
                self.batch_size, self.device)
            self.test_repr  = encode_representations(
                self.simclr_model, self.X_test,
                self.batch_size, self.device)
        show_shape("repr shapes", (self.train_repr,
                                   self.val_repr, self.test_repr))
        self.next(self.train_classifier)

    @resources(memory=16000)
    @step
    def train_classifier(self):
        set_seed(self.seed)
        print(f"Using {self.label_fraction * 100:.1f}% of labeled training data.")

        # subsample labelled training windows
        idx = np.arange(len(self.train_repr))
        np.random.shuffle(idx)
        keep = max(1, int(len(idx)*self.label_fraction))
        idx  = idx[:keep]

        Xtr, ytr = self.train_repr[idx], self.y_train[idx]
        feat_dim = Xtr.shape[1]

        clf = LinearClassifier(feat_dim).to(self.device)
        loss_fn  = nn.BCEWithLogitsLoss()
        opt      = optim.AdamW(clf.parameters(), lr=self.clf_lr)

        train_dl = DataLoader(
            TensorDataset(torch.from_numpy(Xtr).float(),
                          torch.from_numpy(ytr).float()),
            batch_size=self.clf_batch_size, shuffle=True)

        val_dl   = DataLoader(
            TensorDataset(torch.from_numpy(self.val_repr).float(),
                          torch.from_numpy(self.y_val).float()),
            batch_size=self.clf_batch_size, shuffle=False)
        
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        with mlflow.start_run(run_id=self.mlflow_run_id):
            params = {
                "classifier_lr": self.clf_lr,
                "classifier_epochs": self.clf_epochs,
                "classifier_batch_size": self.clf_batch_size,
                "label_fraction": self.label_fraction
            }
            mlflow.log_params(params)
    
            clf, *_ = train_linear_classifier(
                clf, train_dl, val_dl, opt, loss_fn,
                self.clf_epochs, self.device)
            self.classifier = clf
        
        print("Classifier training complete.")
        self.next(self.evaluate)

    @step
    def evaluate(self):
        test_dl = DataLoader(
            TensorDataset(torch.from_numpy(self.test_repr).float(),
                          torch.from_numpy(self.y_test).float()),
            batch_size=self.clf_batch_size, shuffle=False)
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        with mlflow.start_run(run_id=self.mlflow_run_id):
            self.test_acc, *_ = evaluate_classifier(
                self.classifier, test_dl, self.device)
        self.next(self.register)

    @step
    def register(self):
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        if self.test_acc >= self.accuracy_threshold:
            with mlflow.start_run(run_id=current.run_id):
                mlflow.pytorch.log_model(
                    self.classifier,
                    artifact_path="classifier_model",
                    registered_model_name="simclr_classifier")
            self.registered = True
        else: self.registered = False
        self.next(self.end)

    @step
    def end(self):
        print(f"=== SimCLR pipeline done – test accuracy {self.test_acc:.4f} ===")

if __name__ == "__main__":
    ECGSimCLRFlow()
