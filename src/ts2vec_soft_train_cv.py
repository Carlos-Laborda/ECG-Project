# ecg_ts2vec_cv_flow.py
#
# 5-fold participant-wise CV for Soft-TS2Vec + linear classifier
# =============================================================

import os, logging, numpy as np, torch, mlflow, torch.nn as nn, torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from metaflow import FlowSpec, step, Parameter, project, current, resources

# ──────────────────────────────────────────────────────────────
#  Import your own helpers / model code
# ──────────────────────────────────────────────────────────────
from ts2vec_soft import TS2Vec_soft, LinearClassifier, save_sim_mat, densify
from torch_utilities import load_processed_data, set_seed

# ──────────────────────────────────────────────────────────────
#  Flow
# ──────────────────────────────────────────────────────────────
@project(name="ecg_training_ts2vec_soft")
class ECGTS2VecCVFlow(FlowSpec):
    # -------------  generic params -------------
    k_folds                 = Parameter("k_folds",         default=5)
    seed                    = Parameter("seed",            default=42)
    window_data_path        = Parameter("window_data_path", default="../data/interim/windowed_data.h5")
    mlflow_tracking_uri     = Parameter("mlflow_tracking_uri",
                                        default=os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))

    # -------------  TS2Vec hyper-params (identical to your old flow) -------------
    ts2vec_epochs           = Parameter("ts2vec_epochs",   default=50)
    ts2vec_lr               = Parameter("ts2vec_lr",       default=1e-3)
    ts2vec_batch_size       = Parameter("ts2vec_batch_size", default=8)
    ts2vec_output_dims      = Parameter("ts2vec_output_dims", default=320)
    ts2vec_hidden_dims      = Parameter("ts2vec_hidden_dims", default=64)
    ts2vec_depth            = Parameter("ts2vec_depth",    default=10)
    ts2vec_max_train_length = Parameter("ts2vec_max_train_length", default=5000)
    ts2vec_temporal_unit    = Parameter("ts2vec_temporal_unit", default=0)

    # -------------  Soft CL params -------------
    ts2vec_dist_type  = Parameter("ts2vec_dist_type", default="EUC")
    ts2vec_tau_inst   = Parameter("ts2vec_tau_inst",  default=50.0)
    ts2vec_tau_temp   = Parameter("ts2vec_tau_temp",  default=2.5)
    ts2vec_alpha      = Parameter("ts2vec_alpha",     default=0.5)
    ts2vec_lambda     = Parameter("ts2vec_lambda",    default=0.5)

    # -------------  Classifier params -------------
    classifier_epochs      = Parameter("classifier_epochs",  default=25)
    classifier_lr          = Parameter("classifier_lr",      default=1e-4)
    classifier_batch_size  = Parameter("classifier_batch_size", default=32)
    label_fraction         = Parameter("label_fraction",     default=1.0)   # 1 = 100 %

    # -------------  FLOW: start -------------
    @step
    def start(self):
        """Initialise RNGs and MLflow."""
        set_seed(self.seed)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        logging.info("MLflow tracking set to %s", self.mlflow_tracking_uri)
        self.next(self.load_data)

    # -------------  load raw windowed data -------------
    @step
    def load_data(self):
        X, y, groups = load_processed_data(
            hdf5_path=self.window_data_path,
            label_map={"baseline": 0, "mental_stress": 1},
        )
        self.X, self.y, self.groups = X, y, groups
        self.next(self.make_splits)

    # -------------  create participant-wise splits and fork -------------
    @step
    def make_splits(self):
        gkf = GroupKFold(n_splits=self.k_folds)
        self.splits = [(train_idx, test_idx)
                       for train_idx, test_idx in gkf.split(self.X, self.y, self.groups)]
        self.next(self.cv_fold, foreach="splits")   # -> fan-out 

    # -------------  CHILD BRANCH: one fold -------------
    @resources(memory=16000)
    @step
    def cv_fold(self):
        """
        Everything from preprocessing → pre-training → fine-tune → evaluate
        happens inside this branch, completely isolated per fold.
        """
        # ----------------  indices ----------------
        train_idx_all, test_idx = self.input  # supplied by foreach
        # make *participant-wise* 20 % validation split inside train part
        gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=self.seed)
        val_rel, train_rel = next(gss.split(train_idx_all,
                                           self.y[train_idx_all],
                                           self.groups[train_idx_all]))
        train_idx = train_idx_all[train_rel]
        val_idx   = train_idx_all[val_rel]

        # ----------------  tensors ----------------
        self.X_train, self.y_train = self.X[train_idx], self.y[train_idx]
        self.X_val,   self.y_val   = self.X[val_idx],   self.y[val_idx]
        self.X_test,  self.y_test  = self.X[test_idx],  self.y[test_idx]

        # ----------------  pre-train Soft-TS2Vec ----------------
        self._pretrain_ts2vec()               # helper (defined below)

        # ----------------  extract reps ----------------
        self.train_repr = self.ts2vec.encode(self.X_train, encoding_window="full_series")
        self.val_repr   = self.ts2vec.encode(self.X_val,   encoding_window="full_series")
        self.test_repr  = self.ts2vec.encode(self.X_test,  encoding_window="full_series")

        # ----------------  train classifier ----------------
        self._train_classifier()              # helper (defined below)

        # ----------------  evaluate ----------------
        self._evaluate()                      # helper (defined below)

        self.next(self.join_folds)

    # -------------  JOIN -------------
    @step
    def join_folds(self, inputs):
        """
        Aggregate metrics across all folds.
        """
        import numpy as np

        # Collect metrics
        self.accs  = [i.test_acc  for i in inputs]
        self.aucs  = [i.test_auc  for i in inputs]
        self.f1s   = [i.test_f1   for i in inputs]
        self.prs   = [i.test_pr   for i in inputs]

        # mean ± std
        self.mean_acc, self.std_acc = np.mean(self.accs), np.std(self.accs)
        self.mean_auc, self.std_auc = np.mean(self.aucs), np.std(self.aucs)
        self.mean_f1,  self.std_f1  = np.mean(self.f1s),  np.std(self.f1s)
        self.mean_pr,  self.std_pr  = np.mean(self.prs),  np.std(self.prs)

        # Log once (parent run)
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        with mlflow.start_run(run_name="SoftTS2Vec_5fold_CV", nested=False):
            mlflow.log_metrics({
                "cv_accuracy_mean": self.mean_acc,
                "cv_accuracy_std":  self.std_acc,
                "cv_auc_mean":      self.mean_auc,
                "cv_auc_std":       self.std_auc,
                "cv_f1_mean":       self.mean_f1,
                "cv_f1_std":        self.std_f1,
                "cv_pr_mean":       self.mean_pr,
                "cv_pr_std":        self.std_pr,
            })

        self.next(self.end)

    # -------------  END -------------
    @step
    def end(self):
        print("\n=== 5-Fold CV finished ===")
        print(f"Accuracy  : {self.mean_acc:.4f} ± {self.std_acc:.4f}")
        print(f"ROC-AUC   : {self.mean_auc:.4f} ± {self.std_auc:.4f}")
        print(f"F1-Score  : {self.mean_f1 :.4f} ± {self.std_f1 :.4f}")
        print("Done!")

    # ──────────────────────────────────────────────────────────
    #  ↓↓↓  helper methods (kept outside flow steps)  ↓↓↓
    # ──────────────────────────────────────────────────────────
    def _pretrain_ts2vec(self):
        # ── compute (optional) soft label matrix ─────────────────────────
        if self.ts2vec_tau_inst > 0:
            X_flat = self.X_train.squeeze(-1) if self.X_train.ndim == 3 else self.X_train
            sim = save_sim_mat(X_flat, min_=0, max_=1, multivariate=False, type_=self.ts2vec_dist_type)
            soft_labels = densify(-(1 - sim), self.ts2vec_tau_inst, self.ts2vec_alpha)
        else:
            soft_labels = None

        # ── model ────────────────────────────────────────────────────────
        self.ts2vec = TS2Vec_soft(
            input_dims       = self.X_train.shape[2],
            output_dims      = self.ts2vec_output_dims,
            hidden_dims      = self.ts2vec_hidden_dims,
            depth            = self.ts2vec_depth,
            device           = self.device,
            lr               = self.ts2vec_lr,
            batch_size       = self.ts2vec_batch_size,
            lambda_          = self.ts2vec_lambda,
            tau_temp         = self.ts2vec_tau_temp,
            max_train_length = self.ts2vec_max_train_length,
            temporal_unit    = self.ts2vec_temporal_unit,
            soft_instance    = (self.ts2vec_tau_inst > 0),
            soft_temporal    = (self.ts2vec_tau_temp > 0),
        )

        # ── MLflow: one run per fold (tagged) ────────────────────────────
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        with mlflow.start_run(run_name=f"fold_{current.index}", nested=True):
            self.ts2vec.fit(self.X_train,
                            soft_labels,
                            run_dir=f"ts2vec_fold_{current.index}",
                            n_epochs=self.ts2vec_epochs,
                            verbose=False)
            mlflow.log_param("fold", current.index)

    def _train_classifier(self):
        # label-sub-sampling
        idx   = np.random.permutation(len(self.train_repr))
        nkeep = max(1, int(len(idx) * self.label_fraction))
        idx   = idx[:nkeep]

        Xtr, ytr = self.train_repr[idx], self.y_train[idx]
        Xval, yval = self.val_repr, self.y_val

        feat_dim = Xtr.shape[-1]
        clf = LinearClassifier(feat_dim).to(self.device)
        opt = optim.AdamW(clf.parameters(), lr=self.classifier_lr)
        loss_fn = nn.BCEWithLogitsLoss()

        # loaders
        tr_loader  = DataLoader(TensorDataset(torch.from_numpy(Xtr).float(),
                                              torch.from_numpy(ytr).float()),
                                batch_size=self.classifier_batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(torch.from_numpy(Xval).float(),
                                              torch.from_numpy(yval).float()),
                                batch_size=self.classifier_batch_size)

        # simple training loop
        for _ in range(self.classifier_epochs):
            clf.train()
            for xb, yb in tr_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                loss = loss_fn(clf(xb).squeeze(), yb)
                opt.zero_grad(); loss.backward(); opt.step()

        self.classifier = clf

    def _evaluate(self):
        from torcheval.metrics import BinaryAUROC, BinaryF1Score, BinaryAveragePrecision
        loader = DataLoader(TensorDataset(torch.from_numpy(self.test_repr).float(),
                                          torch.from_numpy(self.y_test).float()),
                            batch_size=self.classifier_batch_size)

        preds, gts = [], []
        self.classifier.eval()
        with torch.no_grad():
            for xb, yb in loader:
                logits = self.classifier(xb.to(self.device)).squeeze().cpu()
                preds.append(torch.sigmoid(logits))
                gts.append(yb)
        preds = torch.cat(preds)
        gts   = torch.cat(gts)

        self.test_acc = ((preds > 0.5) == gts).float().mean().item()
        self.test_auc = BinaryAUROC()(preds, gts).item()
        self.test_f1  = BinaryF1Score()(preds, gts).item()
        self.test_pr  = BinaryAveragePrecision()(preds, gts).item()

        # log per-fold metrics
        mlflow.log_metrics({
            "fold_acc": self.test_acc,
            "fold_auc": self.test_auc,
            "fold_f1":  self.test_f1,
            "fold_pr":  self.test_pr,
        })

if __name__ == "__main__":
    ECGTS2VecCVFlow()
