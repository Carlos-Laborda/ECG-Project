import os
import sys
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
import mlflow.pytorch

from torch.utils.data import DataLoader, TensorDataset
from metaflow import FlowSpec, step, Parameter, current, project, resources

from torch_utilities import load_processed_data, split_data_by_participant, set_seed

from tstcc.py import data
from dataloader.dataloader import data_generator
from models.model import base_Model
from models.TC import TC
from models.loss import NTXentLoss
from utils import _logger

@project(name="ecg_training_tstcc")
class ECGTSTCCFlow(FlowSpec):

    # MLflow & data
    mlflow_tracking_uri = Parameter("mlflow_tracking_uri",
        default=os.getenv("MLFLOW_TRACKING_URI", "https://127.0.0.1:5000"))
    window_data_path = Parameter("window_data_path",
        default="../data/interim/windowed_data.h5")

    # general
    seed = Parameter("seed", default=42)

    # TS-TCC pretraining
    tcc_epochs = Parameter("tcc_epochs", default=40)
    tcc_lr = Parameter("tcc_lr", default=3e-4)
    tcc_batch_size = Parameter("tcc_batch_size", default=128)

    # temporal contrasting
    tc_timesteps = Parameter("tc_timesteps", default=10)
    tc_hidden_dim = Parameter("tc_hidden_dim", default=100)
    tc_depth = Parameter("tc_depth", default=4)
    tc_heads = Parameter("tc_heads", default=4)
    tc_mlp_dim = Parameter("tc_mlp_dim", default=64)
    tc_dropout = Parameter("tc_dropout", default=0.1)

    # contextual contrasting
    cc_temperature = Parameter("cc_temperature", default=0.2)
    cc_use_cosine = Parameter("cc_use_cosine", default=True)

    # classifier fine-tuning
    classifier_epochs = Parameter("classifier_epochs", default=25)
    classifier_lr = Parameter("classifier_lr", default=1e-4)
    classifier_batch_size = Parameter("classifier_batch_size", default=32)
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
        self.next(self.train_tstcc)

    @resources(memory=16000)
    @step
    def train_tstcc(self):
        print(f"Training TS-TCC on {self.device}")

        # get data loaders
        train_dl, val_dl, test_dl = data_generator(
            data_path=os.path.dirname(self.window_data_path),
            configs=type("C", (), {"batch_size": self.tcc_batch_size, "drop_last": True}),
            training_mode="self_supervised"
        )

        # assemble a minimal config for Trainer
        class ContextCont: pass
        ctx = ContextCont()
        ctx.temperature = self.cc_temperature
        ctx.use_cosine_similarity = self.cc_use_cosine
        class Config: pass
        self.configs = Config()
        self.configs.batch_size = self.tcc_batch_size
        self.configs.num_epoch = self.tcc_epochs
        self.configs.Context_Cont = ctx

        # models
        self.model = base_Model(self.configs).to(self.device)
        self.temporal_contr_model = TC(self.configs, self.device).to(self.device)

        # optimizers
        model_opt = optim.AdamW(self.model.parameters(), lr=self.tcc_lr, weight_decay=3e-4)
        tc_opt    = optim.AdamW(self.temporal_contr_model.parameters(), lr=self.tcc_lr, weight_decay=3e-4)

        # logger
        logger = _logger(f"ts_tcc_{self.mlflow_run_id}.log")

        # MLflow params
        mlflow.log_params({
            "tcc_epochs": self.tcc_epochs,
            "tcc_lr": self.tcc_lr,
            "tcc_batch_size": self.tcc_batch_size,
            "tc_timesteps": self.tc_timesteps,
            "tc_hidden_dim": self.tc_hidden_dim,
            "tc_depth": self.tc_depth,
            "tc_heads": self.tc_heads,
            "tc_mlp_dim": self.tc_mlp_dim,
            "tc_dropout": self.tc_dropout,
            "cc_temperature": self.cc_temperature,
            "cc_use_cosine": self.cc_use_cosine
        })

        # run training
        from trainer import Trainer  # your Trainer function from TS-TCC repo
        run_dir = f"tstcc_{self.mlflow_run_id}"
        os.makedirs(run_dir, exist_ok=True)
        Trainer(
            model=self.model,
            temporal_contr_model=self.temporal_contr_model,
            model_optimizer=model_opt,
            temp_cont_optimizer=tc_opt,
            train_dl=train_dl, valid_dl=val_dl, test_dl=test_dl,
            device=self.device,
            logger=logger,
            config=self.configs,
            experiment_log_dir=run_dir,
            training_mode="self_supervised"
        )

        # save checkpoint
        ckpt = os.path.join(run_dir, f"tstcc_{self.mlflow_run_id}.pt")
        torch.save({
            'model': self.model.state_dict(),
            'tc': self.temporal_contr_model.state_dict()
        }, ckpt)
        mlflow.log_artifact(ckpt, artifact_path="tstcc_model")

        self.tstcc_run_dir = run_dir
        self.next(self.extract_representations)

    @step
    def extract_representations(self):
        print("Extracting TS-TCC representations …")
        self.model.eval()
        self.temporal_contr_model.eval()

        def _encode(X, y):
            loader = DataLoader(
                TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).long()),
                batch_size=self.tcc_batch_size, shuffle=False
            )
            reprs, labs = [], []
            with torch.no_grad():
                for xb, yb in loader:
                    xb = xb.to(self.device)
                    # conv encoder
                    _, feats = self.model(xb)
                    feats = F.normalize(feats, dim=1)
                    # TC projection head
                    _, c_proj = self.temporal_contr_model(feats, feats)
                    reprs.append(c_proj.cpu().numpy())
                    labs.append(yb.numpy())
            return np.concatenate(reprs, axis=0), np.concatenate(labs, axis=0)

        self.train_repr, _ = _encode(self.X_train, self.y_train)
        self.val_repr, _   = _encode(self.X_val,   self.y_val)
        self.test_repr, _  = _encode(self.X_test,  self.y_test)

        print(f"train_repr shape = {self.train_repr.shape}")
        self.next(self.train_classifier)

    @resources(memory=16000)
    @step
    def train_classifier(self):
        set_seed(self.seed)
        print(f"Training classifier on {self.device}")
        # subset if requested
        n = len(self.train_repr)
        idx = np.random.permutation(n)[:max(1, int(n*self.label_fraction))]
        X_sub, y_sub = self.train_repr[idx], self.y_train[idx]

        # build classifier
        feat_dim = X_sub.shape[1]
        classifier = nn.Linear(feat_dim, 1).to(self.device)
        loss_fn = nn.BCEWithLogitsLoss()
        opt = optim.AdamW(classifier.parameters(), lr=self.classifier_lr)

        train_ds = TensorDataset(torch.from_numpy(X_sub).float(), torch.from_numpy(y_sub).float())
        val_ds   = TensorDataset(torch.from_numpy(self.val_repr).float(), torch.from_numpy(self.y_val).float())
        train_dl = DataLoader(train_ds, batch_size=self.classifier_batch_size, shuffle=True)
        val_dl   = DataLoader(val_ds,   batch_size=self.classifier_batch_size, shuffle=False)

        mlflow.log_params({
            "classifier_epochs": self.classifier_epochs,
            "classifier_lr": self.classifier_lr,
            "classifier_batch_size": self.classifier_batch_size,
            "label_fraction": self.label_fraction
        })

        best_acc = 0.
        for epoch in range(1, self.classifier_epochs+1):
            # train
            classifier.train()
            for xb, yb in train_dl:
                xb, yb = xb.to(self.device), yb.to(self.device)
                opt.zero_grad()
                logits = classifier(xb).squeeze(1)
                loss = loss_fn(logits, yb)
                loss.backward(); opt.step()
            # validate
            classifier.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for xb, yb in val_dl:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    pred = torch.sigmoid(classifier(xb)).round()
                    correct += (pred.cpu()==yb.cpu()).sum().item()
                    total += yb.size(0)
            acc = correct/total
            best_acc = max(best_acc, acc)
            print(f"[Epoch {epoch}] val acc={acc:.4f}")

        self.classifier = classifier
        self.next(self.evaluate)

    @step
    def evaluate(self):
        test_ds = TensorDataset(torch.from_numpy(self.test_repr).float(),
                                torch.from_numpy(self.y_test).float())
        test_dl = DataLoader(test_ds, batch_size=self.classifier_batch_size, shuffle=False)
        self.classifier.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for xb, yb in test_dl:
                xb, yb = xb.to(self.device), yb.to(self.device)
                pred = torch.sigmoid(self.classifier(xb)).round()
                correct += (pred.cpu()==yb.cpu()).sum().item()
                total += yb.size(0)
        self.test_accuracy = correct/total
        print(f"Test accuracy = {self.test_accuracy:.4f}")
        self.next(self.register)

    @step
    def register(self):
        if self.test_accuracy >= self.accuracy_threshold:
            mlflow.pytorch.log_model(
                self.classifier,
                artifact_path="classifier_model",
                registered_model_name="ts_tcc_classifier"
            )
            self.registered = True
        else:
            self.registered = False
        self.next(self.end)

    @step
    def end(self):
        print("=== TS-TCC pipeline complete ===")
        print(f"Test accuracy: {self.test_accuracy:.4f}")
        print(f"Classifier registered: {self.registered}")

if __name__ == "__main__":
    ECGTSTCCFlow()
