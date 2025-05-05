import os
import sys
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import mlflow
import mlflow.pytorch

from torch.utils.data import DataLoader, TensorDataset
from metaflow import FlowSpec, step, Parameter, current, project, resources

from torch_utilities import load_processed_data, split_data_by_participant, set_seed

from tstcc import data_generator_from_arrays, Trainer, base_Model, TC, NTXentLoss, Config as ECGConfig, LinearClassifier, train_linear_classifier, evaluate_classifier, encode_representations, show_shape

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
    tcc_batch_size = Parameter("tcc_batch_size", default=64)

    # temporal contrasting
    tc_timesteps = Parameter("tc_timesteps", default=50)
    tc_hidden_dim = Parameter("tc_hidden_dim", default=100)

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
        print(f"[DBG] preprocess_data  train/val/test shapes:",
              self.X_train.shape, self.X_val.shape, self.X_test.shape)
        self.next(self.train_tstcc)

    @resources(memory=16000)
    @step
    def train_tstcc(self):
        print(f"Training TS-TCC on {self.device}")
        
        # Hyperparameters
        self.configs = ECGConfig()
        self.configs.num_epoch   = self.tcc_epochs    
        self.configs.batch_size  = self.tcc_batch_size
        self.configs.Context_Cont.temperature           = self.cc_temperature
        self.configs.Context_Cont.use_cosine_similarity = self.cc_use_cosine
        self.configs.TC.timesteps = self.tc_timesteps   
        self.configs.TC.hidden_dim = self.tc_hidden_dim

        # Data loaders
        train_dl, val_dl, test_dl = data_generator_from_arrays(
            self.X_train, self.y_train, 
            self.X_val, self.y_val, 
            self.X_test, self.y_test,
            self.configs,
            training_mode="self_supervised"
        )

        # models
        self.model = base_Model(self.configs).to(self.device)
        self.temporal_contr_model = TC(self.configs, self.device).to(self.device)

        # optimizers
        model_opt = optim.AdamW(self.model.parameters(), lr=self.tcc_lr, weight_decay=3e-4)
        tc_opt    = optim.AdamW(self.temporal_contr_model.parameters(), lr=self.tcc_lr, weight_decay=3e-4)

        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        with mlflow.start_run(run_id=self.mlflow_run_id):
        # MLflow params
            params = {
                "model_name": self.model.__class__.__name__,
                "tcc_epochs": self.tcc_epochs,
                "tcc_lr": self.tcc_lr,
                "tcc_batch_size": self.tcc_batch_size,
                "tc_timesteps": self.tc_timesteps,
                "tc_hidden_dim": self.tc_hidden_dim,
                "cc_temperature": self.cc_temperature,
                "cc_use_cosine": self.cc_use_cosine
            }
            mlflow.log_params(params)

            # run training
            run_dir = f"tstcc_{self.mlflow_run_id}"
            os.makedirs(run_dir, exist_ok=True)
            Trainer(
                model=self.model,
                temporal_contr_model=self.temporal_contr_model,
                model_optimizer=model_opt,
                temp_cont_optimizer=tc_opt,
                train_dl=train_dl, valid_dl=val_dl, test_dl=test_dl,
                device=self.device,
                #logger=logger,
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
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        self.model.eval()
        self.temporal_contr_model.eval()
        
        self.train_repr, _ = encode_representations(
            self.X_train, self.y_train,
            self.model, self.temporal_contr_model,
            self.tcc_batch_size, self.device
        )
        
        self.val_repr, _ = encode_representations(
            self.X_val, self.y_val,
            self.model, self.temporal_contr_model,
            self.tcc_batch_size, self.device
        )
        
        self.test_repr, _ = encode_representations(
            self.X_test, self.y_test,
            self.model, self.temporal_contr_model,
            self.tcc_batch_size, self.device
        )

        print(f"train_repr shape = {self.train_repr.shape}")
        show_shape("val_repr / test_repr",
                   (self.val_repr, self.test_repr))
        self.next(self.train_classifier)

    @resources(memory=16000)
    @step
    def train_classifier(self):
        set_seed(self.seed)
        print(f"Training classifier on {self.device}")
        print(f"Using {self.label_fraction * 100:.1f}% of labeled training data.")

        # subsample labelled windows
        indices = np.arange(len(self.train_repr))
        np.random.shuffle(indices)
        keep = max(1, int(len(indices) * self.label_fraction))
        subset_idx = indices[:keep]

        X_train_sub = self.train_repr[subset_idx]
        y_train_sub = self.y_train[subset_idx]

        # build model and optimiser 
        feat_dim = X_train_sub.shape[-1]
        self.classifier = LinearClassifier(input_dim=feat_dim).to(self.device)
        loss_fn = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(self.classifier.parameters(), lr=self.classifier_lr)

        # DataLoaders
        train_ds = TensorDataset(torch.from_numpy(X_train_sub).float(),
                                torch.from_numpy(y_train_sub).float())
        val_ds = TensorDataset(torch.from_numpy(self.val_repr).float(),
                                torch.from_numpy(self.y_val).float())

        train_dl = DataLoader(train_ds, batch_size=self.classifier_batch_size, shuffle=True)
        val_dl = DataLoader(val_ds,   batch_size=self.classifier_batch_size, shuffle=False)

        # MLflow logging and training
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        with mlflow.start_run(run_id=self.mlflow_run_id):
            params = {
                "classifier_lr": self.classifier_lr,
                "classifier_epochs": self.classifier_epochs,
                "classifier_batch_size": self.classifier_batch_size,
                "label_fraction": self.label_fraction
            }
            mlflow.log_params(params)
            
            self.classifier, val_accs, val_aurocs, val_pr_aucs, val_f1s = train_linear_classifier(
                model = self.classifier,
                train_loader = train_dl,
                val_loader = val_dl,
                loss_fn = loss_fn,
                optimizer = optimizer,
                epochs = self.classifier_epochs,
                device = self.device
            )
        
        print("Classifier training complete.")
        self.next(self.evaluate)

    @step
    def evaluate(self):
        """Evaluate the linear classifier on the held-out test windows."""
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        test_ds = TensorDataset(torch.from_numpy(self.test_repr).float(),
                                torch.from_numpy(self.y_test).float())
        test_dl = DataLoader(test_ds, batch_size=self.classifier_batch_size, shuffle=False)

        with mlflow.start_run(run_id=self.mlflow_run_id):
            self.test_accuracy, test_auroc, test_pr_auc, test_f1 = evaluate_classifier(
                model = self.classifier,
                test_loader = test_dl,
                device = self.device
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
        print("Done!")

if __name__ == "__main__":
    ECGTSTCCFlow()
