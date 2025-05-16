import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import cv2
import mlflow
from torch.utils.data import Dataset, DataLoader

from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision, BinaryF1Score
from typing import Union, Sequence

# ----------------------------------------------------------------------
# helpers (put right after the imports in simclr.py)
# ----------------------------------------------------------------------
def to_1d(x: np.ndarray) -> np.ndarray:
    """Remove channel/extra dims so that augmentations see (L,) arrays."""
    return np.asarray(x).squeeze()        # guarantees positive length axis=-1

def same_length(arr: np.ndarray, target_len: int) -> np.ndarray:
    """Resample (with linear interp) so that len(arr)==target_len."""
    if arr.shape[-1] == target_len:
        return arr
    return cv2.resize(arr.reshape(-1, 1), (1, target_len),
                      interpolation=cv2.INTER_LINEAR).flatten()

def safe_contiguous(x: np.ndarray) -> np.ndarray:
    """Ensure positive stride & contiguous memory."""
    return np.ascontiguousarray(x)

# ----------------------------------------------------------------------
# Augmentations (following paper’s guidelines (15, 0.9, 20, 9, 1.05))
# ----------------------------------------------------------------------
def add_noise(signal, noise_amount):
    noise = np.random.normal(0, noise_amount, size=signal.shape)
    return signal + noise


def add_noise_with_SNR(signal, snr_db=15):
    x_watts = signal ** 2
    sig_avg_watts = np.mean(x_watts)
    sig_avg_db = 10 * np.log10(sig_avg_watts)
    noise_avg_db = sig_avg_db - snr_db
    noise_avg_watts = 10 ** (noise_avg_db / 10)
    noise = np.random.normal(0, math.sqrt(noise_avg_watts), size=signal.shape)
    return signal + noise


def scaled(signal, factor=0.9):
    return signal * factor


def negate(signal):
    return -signal


def hor_flip(signal):
    return np.flip(signal)


def permute(signal, pieces=20):
    L = signal.shape[-1]
    piece_len = L // pieces
    segments = [signal[i*piece_len:(i+1)*piece_len] for i in range(pieces)]
    np.random.shuffle(segments)
    return np.concatenate(segments, axis=-1)

def time_warp(signal, pieces=9, stretch_factor=1.05, squeeze_factor=0.95):
    signal = to_1d(signal)                
    L = signal.shape[-1]
    if L < pieces:                 
        return signal.copy()

    seg_len = L // pieces
    output = []
    for i in range(pieces):
        seg = signal[i*seg_len:(i+1)*seg_len]
        new_len = int(np.ceil(len(seg) * (stretch_factor if np.random.rand() < 0.5
                                          else squeeze_factor)))
        warped = cv2.resize(seg.reshape(-1, 1), (1, new_len),
                            interpolation=cv2.INTER_LINEAR).flatten()
        output.append(warped)

    warped_all = np.concatenate(output, axis=-1)
    return same_length(warped_all, L) 

# List of available augmentations
AUGMENTATIONS = [
    add_noise_with_SNR,   # uses snr_db=15
    scaled,               # uses factor=0.9
    permute,              # uses pieces=20
    time_warp,            # uses pieces=9, stretch=1.05, squeeze=0.95
    negate,
    hor_flip,
]

def DataTransform(signal):
    """Generate two augmented views of the same ECG signal."""
    signal = to_1d(signal)       
    target_len = signal.shape[-1]

    view1, view2 = signal.copy(), signal.copy()
    for _ in range(2):
        view1 = np.random.choice(AUGMENTATIONS)(view1)
        view2 = np.random.choice(AUGMENTATIONS)(view2)

    # guarantee fixed length & positive strides
    view1 = safe_contiguous(same_length(view1, target_len))
    view2 = safe_contiguous(same_length(view2, target_len))
    return view1, view2

# -------------------------------------------------------------
# Tuned augmentations for 10k-sample ECG windows
# -------------------------------------------------------------
# def add_noise_with_SNR(x, snr_db=None):
#     snr = snr_db if snr_db is not None else np.random.uniform(18, 30)
#     p_signal = np.mean(x**2)
#     p_noise  = p_signal / (10**(snr/10))
#     return x + np.random.normal(0, np.sqrt(p_noise), size=x.shape)

# def random_scaling(x):
#     return x * np.random.uniform(0.9, 1.1)

# def random_crop_shift(x, crop_len=8000):
#     if len(x) <= crop_len:
#         return x
#     start = np.random.randint(0, len(x)-crop_len)
#     cropped = x[start:start+crop_len]
#     pad_left  = np.random.randint(0, len(x)-crop_len)
#     pad_right = len(x)-crop_len-pad_left
#     return np.pad(cropped, (pad_left, pad_right))

# def local_jitter(x):
#     return x + np.random.normal(0, 0.003*np.std(x), size=x.shape)

# def baseline_wander(x):
#     f = np.random.uniform(0.05, 0.3)
#     t = np.arange(len(x)) / 1000   
#     amp = np.random.uniform(0.01, 0.05) * np.std(x)
#     return x + amp * np.sin(2*np.pi*f*t)

# def negate(x):   return -x
# def hor_flip(x): return np.ascontiguousarray(np.flip(x))

# AUGS = [
#     (add_noise_with_SNR, 0.25),
#     (random_scaling,     0.20),
#     (random_crop_shift,  0.20),
#     (local_jitter,       0.15),
#     (baseline_wander,    0.10),
#     (negate,             0.05),
#     (hor_flip,           0.05),
# ]

# funcs, probs = zip(*AUGS)

# def sample_augmented(x: np.ndarray) -> np.ndarray:
#     idx = np.random.choice(len(funcs), p=probs)
#     return funcs[idx](x)

# def DataTransform(signal):
#     sig = to_1d(signal)
#     v1, v2 = sig.copy(), sig.copy()
#     v1 = sample_augmented(v1); v1 = sample_augmented(v1)
#     v2 = sample_augmented(v2); v2 = sample_augmented(v2)
#     # ensure shape & contiguity
#     L = sig.shape[-1]
#     return safe_contiguous(same_length(v1, L)), safe_contiguous(same_length(v2, L))


# ----------------------------------------------------------------------
# Encoder f(.)
# ----------------------------------------------------------------------
class ECGEncoder(nn.Module):
    """1D CNN encoder with 3 blocks, each block: Conv1d -> ReLU -> MaxPool -> Dropout"""
    def __init__(self, input_channels=1, dropout=0.3, window=10000):
        super(ECGEncoder, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=32, stride=1, padding=16, bias=False),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(dropout)
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=16, stride=1, padding=8, bias=False),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(dropout)
        )
        self.block3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=8, stride=1, padding=4, bias=False),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(dropout)
        )
        # After conv blocks, flatten and FC to 80 units
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(256 *  (window // 8), 80) 
        # L2 regularization to be set via optimizer weight_decay

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.flatten(x)
        h = self.fc(x)
        return h

# ----------------------------------------------------------------------
# Projection head g(.)
# ----------------------------------------------------------------------
class ProjectionHead(nn.Module):
    """MLP → 256‑dim, followed by L2‑norm (no softmax)."""
    def __init__(self, in_dim=80, proj_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, proj_dim), nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim))

    def forward(self, h):
        z = self.net(h)
        return F.normalize(z, dim=1)

# ----------------------------------------------------------------------
# SimCLR Model
# ----------------------------------------------------------------------
class SimCLR(nn.Module):
    def __init__(self, encoder:ECGEncoder, projector:ProjectionHead):
        super().__init__()
        self.f, self.g = encoder, projector

    def forward(self, x1, x2):                 # views (B,1,L)
        h1, h2 = self.f(x1), self.f(x2)        # (B,80)
        z1, z2 = self.g(h1), self.g(h2)        # (B,256)
        return h1, h2, z1, z2
    
# ----------------------------------------------------------------------
# NT-Xent Loss
# ----------------------------------------------------------------------
class NTXentLoss(nn.Module):
    def __init__(self, batch_size:int, T:float=0.5):
        super().__init__()
        self.B, self.T = batch_size, T
        self.register_buffer("labels", torch.arange(batch_size).repeat(2))

    def forward(self, z1, z2):                 # (B,D)
        z = torch.cat([z1, z2], dim=0)         # (2B,D)
        sim = torch.mm(z, z.t()) / self.T      # cosine sims after norm
        sim.fill_diagonal_(-1e9)               # mask self‑contrast
        l_pos = torch.diag(sim, self.B)        # positives
        r_pos = torch.diag(sim, -self.B)
        positives = torch.cat([l_pos, r_pos], 0)[:,None]     # (2B,1)
        loss = -torch.mean(torch.log(torch.exp(positives) / torch.exp(sim).sum(dim=1, keepdim=True)))
        return loss

# ----------------------------------------------------------------------
# Dataset wrapper
# ----------------------------------------------------------------------
class SimCLRDataset(Dataset):
    def __init__(self, signals):
        self.signals = signals

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        x = self.signals[idx]
        v1, v2 = DataTransform(x)
        
        # reshape to (C,L)
        v1 = torch.tensor(v1, dtype=torch.float).unsqueeze(0)
        v2 = torch.tensor(v2, dtype=torch.float).unsqueeze(0)
        return v1, v2

def get_simclr_model(window:int=10000, device:str="cpu"):
    enc = ECGEncoder(window=window).to(device)
    proj = ProjectionHead().to(device)
    return SimCLR(enc, proj)

def simclr_data_loaders(X_train, X_val, batch_size:int):
    return (DataLoader(SimCLRDataset(X_train), batch_size, shuffle=True,  drop_last=True),
            DataLoader(SimCLRDataset(X_val),   batch_size, shuffle=False, drop_last=True))

def pretrain_one_epoch(model, loader, loss_fn, opt, device):
    model.train()
    tot = 0
    for v1,v2 in loader:
        v1,v2 = v1.to(device), v2.to(device)
        _,_,z1,z2 = model(v1,v2)
        loss = loss_fn(z1, z2)
        opt.zero_grad()
        loss.backward()
        opt.step()
        tot += loss.item()
    return tot/len(loader)

@torch.no_grad()
def encode_representations(model, X, batch_size:int, device:str):
    """Return L2‑normalised encoder outputs h."""
    loader = DataLoader(torch.from_numpy(X).float(), batch_size, shuffle=False)
    reps = []
    model.eval()
    for xb in loader:
        xb = xb.permute(0, 2, 1).to(device) 
        h = F.normalize(model.f(xb), dim=1)
        reps.append(h.cpu().numpy())
    return np.concatenate(reps,0)

# ----------------------------------------------------------------------        
# linear classifier
# ----------------------------------------------------------------------
class LinearClassifier(nn.Module):
    def __init__(self, input_dim, num_classes=1):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        # x is expected to be of shape (batch, feature_dim)
        return self.fc(x)
    
# --------------------------------------------------------
# Classifier Training Loop
# --------------------------------------------------------
def train_linear_classifier(
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_fn,
    epochs,
    device,
):
    """
    Trains a linear classifier with validation loop and MLflow logging.

    Args:
        model: The classifier model (subclass of nn.Module).
        train_loader: DataLoader for the training set.
        val_loader: DataLoader for the validation set.
        optimizer: The optimizer for training.
        loss_fn: The loss function.
        epochs: Number of training epochs.
        device: The device to train on ('cpu' or 'cuda').

    Returns:
        Tuple: (Trained model, list of validation accuracies, list of validation AUROCs, list of validation PR-AUCs, list of validation F1-Scores)
    """
    val_accuracies = []
    val_aurocs = []
    val_pr_aucs = []
    val_f1_scores = []
    
    train_auroc_metric = BinaryAUROC()
    val_auroc_metric = BinaryAUROC()
    train_pr_auc_metric = BinaryAveragePrecision()
    val_pr_auc_metric = BinaryAveragePrecision()
    train_f1_metric = BinaryF1Score() 
    val_f1_metric = BinaryF1Score()

    for epoch in range(1, epochs + 1):
        # Training phase
        model.train()
        running_loss = 0.0
        correct_train = total_train = 0
        train_auroc_metric.reset()
        train_pr_auc_metric.reset()
        train_f1_metric.reset()
        
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features).squeeze() 
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * features.size(0)
            # Accuracy calculation
            preds_train = (torch.sigmoid(outputs) > 0.5).float()
            correct_train += (preds_train == labels).sum().item()
            total_train += labels.size(0)
            # Update metrics
            outputs_cpu = outputs.detach().cpu()
            labels_cpu_int = labels.cpu().int()
            preds_train_cpu_int = preds_train.cpu().int()
            
            train_auroc_metric.update(outputs_cpu, labels_cpu_int)
            train_pr_auc_metric.update(outputs_cpu, labels_cpu_int)
            train_f1_metric.update(preds_train_cpu_int, labels_cpu_int)
                
        epoch_loss = running_loss / total_train if total_train > 0 else 0.0
        epoch_train_acc = correct_train / total_train if total_train > 0 else 0.0
        epoch_train_auroc = train_auroc_metric.compute().item()
        epoch_train_pr_auc = train_pr_auc_metric.compute().item()
        epoch_train_f1 = train_f1_metric.compute().item()
        
        mlflow.log_metrics({
            "classifier_train_loss": epoch_loss,
            "classifier_train_accuracy": epoch_train_acc,
            "classifier_train_auroc": epoch_train_auroc,
            "classifier_train_pr_auc": epoch_train_pr_auc,
            "classifier_train_f1": epoch_train_f1,
        }, step=epoch)

        print(f"Epoch {epoch}/{epochs} - Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_train_acc:.4f}, Train AUROC: {epoch_train_auroc:.4f}, Train PR-AUC: {epoch_train_pr_auc:.4f}, Train F1: {epoch_train_f1:.4f}")

        # Validation phase
        model.eval()
        correct_val = total_val = 0
        val_auroc_metric.reset()
        val_pr_auc_metric.reset()
        val_f1_metric.reset()
        
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features).squeeze()
                # Accuracy calculation
                preds_val = (torch.sigmoid(outputs) > 0.5).float()
                correct_val += (preds_val == labels).sum().item()
                total_val += labels.size(0)
                # Update metrics
                outputs_cpu = outputs.cpu()
                labels_cpu_int = labels.cpu().int()
                preds_val_cpu_int = preds_val.cpu().int()
                
                val_auroc_metric.update(outputs_cpu, labels_cpu_int)
                val_pr_auc_metric.update(outputs_cpu, labels_cpu_int)
                val_f1_metric.update(preds_val_cpu_int, labels_cpu_int)
        
        val_acc = correct_val / total_val if total_val > 0 else 0.0
        val_auroc = val_auroc_metric.compute().item()
        val_pr_auc = val_pr_auc_metric.compute().item()
        val_f1 = val_f1_metric.compute().item()
        val_accuracies.append(val_acc)
        val_aurocs.append(val_auroc)
        val_pr_aucs.append(val_pr_auc)
        val_f1_scores.append(val_f1)
        
        mlflow.log_metrics({
            "val_accuracy": val_acc,
            "val_auroc": val_auroc,
            "val_pr_auc": val_pr_auc,
            "val_f1": val_f1,
        }, step=epoch)
        
        print(f"Epoch {epoch}/{epochs} - Val Acc: {val_acc:.4f}, Val AUROC: {val_auroc:.4f}, Val PR-AUC: {val_pr_auc:.4f}, Val F1: {val_f1:.4f}")

    return model, val_accuracies, val_aurocs, val_pr_aucs, val_f1_scores

# --------------------------------------------------------
# Classifier Evaluation Loop
# --------------------------------------------------------
def evaluate_classifier(
    model,
    test_loader,
    device
):
    """
    Evaluates the classifier on the test set and logs the accuracy to MLflow.

    Args:
        model: The trained classifier model.
        test_loader: DataLoader for the test set.
        device: The device to evaluate on ('cpu' or 'cuda').

    Returns:
        float: The test accuracy, test AUROC, and test PR-AUC.
    """
    model.eval()
    correct = total = 0
    test_auroc_metric = BinaryAUROC()
    test_pr_auc_metric = BinaryAveragePrecision()
    test_f1_metric = BinaryF1Score()
    test_auroc_metric.reset()
    test_pr_auc_metric.reset()
    test_f1_metric.reset()
    
    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features).squeeze() 
            # Accuracy calculation
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            # Update metrics
            outputs_cpu = outputs.cpu()
            labels_cpu_int = labels.cpu().int()
            preds_cpu_int = preds.cpu().int()
            
            test_auroc_metric.update(outputs_cpu, labels_cpu_int)
            test_pr_auc_metric.update(outputs_cpu, labels_cpu_int)
            test_f1_metric.update(preds_cpu_int, labels_cpu_int)

    test_accuracy = correct / total if total > 0 else 0.0
    test_auroc = test_auroc_metric.compute().item()
    test_pr_auc = test_pr_auc_metric.compute().item()
    test_f1 = test_f1_metric.compute().item()
    
    mlflow.log_metrics({
        "test_accuracy": test_accuracy,
        "test_auroc": test_auroc,
        "test_pr_auc": test_pr_auc,
        "test_f1": test_f1,
    })
    
    print(f"Test Accuracy: {test_accuracy:.4f}, Test AUROC: {test_auroc:.4f}, Test PR-AUC: {test_pr_auc:.4f}, Test F1: {test_f1:.4f}")
    return test_accuracy, test_auroc, test_pr_auc, test_f1



DEBUG_SHAPES = True # flip to False to mute everything

_printed_once: set[str] = set()  

def _shape(x):
    """convenience – works for tensors / ndarrays / lists"""
    if isinstance(x, (torch.Tensor, np.ndarray)):
        return tuple(x.shape)
    return str(type(x))

def show_shape(label: str, obj: Union[torch.Tensor, np.ndarray, Sequence], *,
               once: bool = True) -> None:
    """
    Print `<label>: <shape>` in a *single* line.
    If `once=True` (default) the same label is never printed again
    """
    if not DEBUG_SHAPES:
        return
    if once and label in _printed_once:
        return
    _printed_once.add(label)

    if isinstance(obj, (list, tuple)):
        shapes = [_shape(o) for o in obj]
    else:
        shapes = _shape(obj)
    print(f"[DBG] {label:<26s} : {shapes}", flush=True)