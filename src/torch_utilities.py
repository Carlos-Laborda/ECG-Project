import tempfile
from pathlib import Path

import os
import random
import numpy as np
import h5py
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchinfo import summary
from sklearn.metrics import precision_recall_curve, auc, f1_score, roc_curve
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision
from torcheval.metrics.functional import multiclass_f1_score

# MLflow imports
import mlflow
import mlflow.pytorch
from mlflow.types import Schema, TensorSpec
from mlflow.models import ModelSignature
from mlflow.tracking import MlflowClient

# def load_processed_data(hdf5_path, label_map=None):
#     """
#     Load windowed ECG data from an HDF5 file.
    
#     Returns:
#         X (np.ndarray): shape (N, window_length, 1)
#         y (np.ndarray): shape (N,)
#         groups (np.ndarray): shape (N,) - each window's participant ID
#     """
#     if label_map is None:
#         label_map = {"baseline": 0, "mental_stress": 1}

#     X_list, y_list, groups_list = [], [], []
#     with h5py.File(hdf5_path, "r") as f:
#         participants = list(f.keys())
#         for participant_key in participants:
#             participant_id = participant_key.replace("participant_", "")
#             for cat in f[participant_key].keys():
#                 if cat not in label_map:
#                     continue
#                 windows = f[participant_key][cat][:]
#                 n_windows = windows.shape[0]
#                 groups_arr = np.array([participant_id] * n_windows, dtype=object)

#                 X_list.append(windows)
#                 y_list.append(np.full((n_windows,), label_map[cat], dtype=int))
#                 groups_list.append(groups_arr)

#     if len(X_list) == 0:
#         raise ValueError(f"No valid data found in {hdf5_path} with label_map {label_map}.")

#     # Concatenate all data
#     X = np.concatenate(X_list, axis=0)
#     y = np.concatenate(y_list, axis=0)
#     groups = np.concatenate(groups_list, axis=0)

#     # Expand dims for CNN: (N, window_length, 1)
#     X = np.expand_dims(X, axis=-1)
#     return X, y, groups

# MLflow helpers
def build_supervised_fingerprint(cfg: dict[str, object]) -> dict[str, str]:
    """
    Create an immutable dictionary (all str) uniquely describing one training.
    """
    keys = (
        "model_name", "seed", "lr", "batch_size", "num_epochs",
        "patience", "scheduler_mode", "scheduler_factor",
        "scheduler_patience", "scheduler_min_lr", "label_fraction"
    )
    return {k: str(cfg[k]) for k in keys}

def search_encoder_fp(fp: dict[str, str], experiment_name: str,
                      tracking_uri: str) -> str | None:
    """Return run_id of an MLflow run whose params exactly match 'fp'."""
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    exp    = client.get_experiment_by_name(experiment_name)
    if exp is None:
        return None
    clauses = ["attributes.status = 'FINISHED'"]
    clauses += [f"params.{k} = '{v}'" for k, v in fp.items()]
    hits = mlflow.search_runs([exp.experiment_id],
                              filter_string=" and ".join(clauses),
                              max_results=1)
    return None if hits.empty else hits.iloc[0]["run_id"]

def load_processed_data(hdf5_path, label_map=None):
    """
    Load windowed ECG data from an HDF5 file.
    
    Returns:
        X (np.ndarray): shape (N, window_length, 1)
        y (np.ndarray): shape (N,)
        groups (np.ndarray): shape (N,) - each window's participant ID
    """
    if label_map is None:
        label_map = {"baseline": 0, "mental_stress": 1}

    X_list, y_list, groups_list = [], [], []
    with h5py.File(hdf5_path, "r") as f:
        participants = list(f.keys())
        for participant_key in participants:
            participant_id = participant_key.replace("participant_", "")
            for cat in f[participant_key].keys():
                if cat not in label_map:
                    continue
                cat_group = f[participant_key][cat]
                segment_windows_list = []
                for segment_name in cat_group.keys():
                    windows = cat_group[segment_name][...]
                    segment_windows_list.append(windows)
                if len(segment_windows_list) == 0:
                    continue
                # Concatenate windows from all segments in this category
                windows_all = np.concatenate(segment_windows_list, axis=0)
                n_windows = windows_all.shape[0]
                groups_arr = np.array([participant_id] * n_windows, dtype=object)

                X_list.append(windows_all)
                y_list.append(np.full((n_windows,), label_map[cat], dtype=int))
                groups_list.append(groups_arr)

    if len(X_list) == 0:
        raise ValueError(f"No valid data found in {hdf5_path} with label_map {label_map}.")

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    groups = np.concatenate(groups_list, axis=0)

    # Expand dims for CNN: (N, window_length, 1)
    X = np.expand_dims(X, axis=-1)
    return X, y, groups


def split_data_by_participant(X, y, groups, train_ratio=0.6, val_ratio=0.2, seed=42):
    """
    Split data by unique participant IDs.
    
    Returns:
        (X_train, y_train), (X_val, y_val), (X_test, y_test)
    """
    unique_participants = np.unique(groups)
    n_participants = len(unique_participants)
    
    # Set seed and shuffle participants
    np.random.seed(seed)
    shuffled = np.random.permutation(unique_participants)
    
    n_train = int(n_participants * train_ratio)
    n_val = int(n_participants * val_ratio)
    
    train_participants = shuffled[:n_train]
    val_participants = shuffled[n_train:n_train+n_val]
    test_participants = shuffled[n_train+n_val:]
    
    train_mask = np.isin(groups, train_participants)
    val_mask = np.isin(groups, val_participants)
    test_mask = np.isin(groups, test_participants)
    
    return (X[train_mask], y[train_mask]), (X[val_mask], y[val_mask]), (X[test_mask], y[test_mask])

def split_indices_by_participant(groups, train_ratio=0.6, val_ratio=0.2, seed=42):
    """
    Return index arrays for train / val / test
    """
    uniq = np.unique(groups)
    rng  = np.random.default_rng(seed)
    rng.shuffle(uniq)

    n_train = int(len(uniq) * train_ratio)
    n_val   = int(len(uniq) * val_ratio)

    train_p, val_p, test_p = np.split(uniq, [n_train, n_train + n_val])

    train_idx = np.flatnonzero(np.isin(groups, train_p))
    val_idx   = np.flatnonzero(np.isin(groups, val_p))
    test_idx  = np.flatnonzero(np.isin(groups, test_p))

    return train_idx, val_idx, test_idx

class ECGDataset(Dataset):
    """
    PyTorch Dataset for ECG data.
    """
    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        sample = self.X[idx]
        label = self.y[idx]
        if self.transform:
            sample = self.transform(sample)
        # Convert to torch.Tensor
        sample = torch.tensor(sample, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)
        return sample, label


# -------------------------
# Simple classifier
# -------------------------
class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_classes=1):
        super(SimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x is expected to be of shape (batch, feature_dim)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# liniar classifier
class LinearClassifier(nn.Module):
    def __init__(self, input_dim, num_classes=1):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        # x is expected to be of shape (batch, feature_dim)
        return self.fc(x)
    
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, dropout=0.2, num_classes=1):
        super(MLPClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# ----------------------
# CNN Model Class
# ----------------------
class Simple1DCNN(nn.Module):
    """
    A very simple 1D CNN with two convolutional layers and a single dense layer.
    Intended for quick tests and validation with minimal training time.
    """
    def __init__(self):
        super(Simple1DCNN, self).__init__()
        # Conv block 1: 1 -> 8 channels
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        # Conv block 2: 8 -> 16 channels
        self.conv2 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        # Global average pooling to get a single feature per channel
        self.gap = nn.AdaptiveAvgPool1d(output_size=1)
        # Final fully-connected layer for binary classification
        self.fc1 = nn.Linear(16, 1)

    def forward(self, x):
        # x shape: (batch_size, 1, sequence_length)
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.gap(x)  # shape: (batch_size, 16, 1)
        x = x.view(x.size(0), -1)  # flatten to (batch_size, 16)
        x = torch.sigmoid(self.fc1(x))  # shape: (batch_size, 1)
        return x

class Simple1DCNN_v2(nn.Module):
    """
    A balanced CNN architecture between Simple1DCNN and Improved1DCNN
    with moderate capacity and regularization for ECG stress classification.
    """
    def __init__(self, dropout_rate=0.2):
        super(Simple1DCNN_v2, self).__init__()
        self.bn_input = nn.BatchNorm1d(1)
        
        # Block 1
        self.conv1 = nn.Conv1d(1, 16, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(16)
        self.pool1 = nn.MaxPool1d(kernel_size=4)  # More aggressive pooling
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # Block 2
        self.conv2 = nn.Conv1d(16, 32, kernel_size=7, padding=3)
        self.bn2 = nn.BatchNorm1d(32)
        self.pool2 = nn.MaxPool1d(kernel_size=4)  # More aggressive pooling
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # Global pooling
        self.gap = nn.AdaptiveAvgPool1d(1)
        
        # Classifier
        self.fc1 = nn.Linear(32, 16)
        self.dropout3 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(16, 1)
    
    def forward(self, x):
        # Input normalization
        x = self.bn_input(x)
        
        # Block 1
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # Block 2
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # Global pooling
        x = self.gap(x)
        x = x.view(x.size(0), -1)  # Flatten
        
        # Classifier
        x = F.relu(self.fc1(x))
        x = self.dropout3(x)
        x = torch.sigmoid(self.fc2(x))  
        
        return x
    
class Improved1DCNN(nn.Module):
    """
    A more complex 1D CNN model with 3 convolutional blocks with 
    3-layer classification head with dropout.
    """
    def __init__(self):
        super(Improved1DCNN, self).__init__()
        self.bn_input = nn.BatchNorm1d(1)
        # Block 1
        self.conv1_1 = nn.Conv1d(1, 32, kernel_size=5, padding=2, bias=False)
        self.conv1_2 = nn.Conv1d(32, 32, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.dropout1 = nn.Dropout(0.2)
        # Block 2
        self.conv2_1 = nn.Conv1d(32, 64, kernel_size=5, padding=2, bias=False)
        self.conv2_2 = nn.Conv1d(64, 64, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(0.2)
        # Block 3
        self.conv3_1 = nn.Conv1d(64, 128, kernel_size=5, padding=2, bias=False)
        self.conv3_2 = nn.Conv1d(128, 128, kernel_size=5, padding=2)
        self.gap = nn.AdaptiveAvgPool1d(1)
        # Dense layers
        self.fc1 = nn.Linear(128, 64)
        self.dropout3 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(64, 32)
        self.dropout4 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(32, 1)
    
    def forward(self, x):
        # Input x: (batch, channels, length)
        x = self.bn_input(x)
        # Block 1
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = self.pool1(x)
        x = self.bn1(x)
        x = self.dropout1(x)
        # Block 2
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.pool2(x)
        x = self.bn2(x)
        x = self.dropout2(x)
        # Block 3
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = self.gap(x)  # shape: (batch, 128, 1)
        x = x.view(x.size(0), -1)  # flatten to (batch, 128)
        # Dense layers
        x = F.relu(self.fc1(x))
        x = self.dropout3(x)
        x = F.relu(self.fc2(x))
        x = self.dropout4(x)
        x = torch.sigmoid(self.fc3(x))
        return x


# from: Sarkar, P., Etemad, A.: Self-Supervised ECG Representation Learning for Emotion Recognition. 
# IEEE Transactions on Affective Computing \textbf{13}(3), 1541--1554 (2020). \doi{10.1109/TAFFC.2020.3014842}
class EmotionRecognitionCNN(nn.Module):
    def __init__(self):
        super(EmotionRecognitionCNN, self).__init__()
        self.bn_input = nn.BatchNorm1d(1)
        # Conv block 1
        self.conv1_1 = nn.Conv1d(1, 32, kernel_size=32, padding='same')
        self.conv1_2 = nn.Conv1d(32, 32, kernel_size=32, padding='same')
        self.pool1 = nn.MaxPool1d(kernel_size=8, stride=2)

        # Conv block 2
        self.conv2_1 = nn.Conv1d(32, 64, kernel_size=16, padding='same')
        self.conv2_2 = nn.Conv1d(64, 64, kernel_size=16, padding='same')
        self.pool2 = nn.MaxPool1d(kernel_size=8, stride=2)

        # Conv block 3
        self.conv3_1 = nn.Conv1d(64, 128, kernel_size=8, padding='same')
        self.conv3_2 = nn.Conv1d(128, 128, kernel_size=8, padding='same')
        self.global_pool = nn.AdaptiveMaxPool1d(1)

        # Dense layers
        self.fc1 = nn.Linear(128, 512)
        self.dropout = nn.Dropout(0.6)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):
        # Conv block 1
        x = self.bn_input(x)
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = self.pool1(x)

        # Conv block 2
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.pool2(x)

        # Conv block 3
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = self.global_pool(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))

        return x

# Adapted from: Sarkar, P., Etemad, A.: Self-Supervised ECG Representation Learning for Emotion Recognition. 
# IEEE Transactions on Affective Computing \textbf{13}(3), 1541--1554 (2020). \doi{10.1109/TAFFC.2020.3014842}
class Improved1DCNN_v2(nn.Module):
    """
    A more complex 1D CNN model with 3 convolutional blocks with 
    3-layer classification head with dropout.
    """
    def __init__(self):
        super(Improved1DCNN_v2, self).__init__()
        self.bn_input = nn.BatchNorm1d(1)
        # Block 1
        self.conv1_1 = nn.Conv1d(1, 32, kernel_size=5, padding=2, bias=False)
        self.conv1_2 = nn.Conv1d(32, 32, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.dropout1 = nn.Dropout(0.1)
        # Block 2
        self.conv2_1 = nn.Conv1d(32, 64, kernel_size=11, padding=5, bias=False)
        self.conv2_2 = nn.Conv1d(64, 64, kernel_size=11, padding=5)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(0.1)
        # Block 3
        self.conv3_1 = nn.Conv1d(64, 128, kernel_size=17, padding=8, bias=False)
        self.conv3_2 = nn.Conv1d(128, 128, kernel_size=17, padding=8)
        self.gap = nn.AdaptiveAvgPool1d(1)
        # Dense layers
        self.fc1 = nn.Linear(128, 128)
        self.dropout3 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 64)
        self.dropout4 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(64, 1)
    
    def forward(self, x):
        # Input x: (batch, channels, length)
        x = self.bn_input(x)
        # Block 1
        x = F.gelu(self.conv1_1(x))
        x = F.gelu(self.conv1_2(x))
        x = self.pool1(x)
        x = self.bn1(x)
        x = self.dropout1(x)
        # Block 2
        x = F.gelu(self.conv2_1(x))
        x = F.gelu(self.conv2_2(x))
        x = self.pool2(x)
        x = self.bn2(x)
        x = self.dropout2(x)
        # Block 3
        x = F.gelu(self.conv3_1(x))
        x = F.gelu(self.conv3_2(x))
        x = self.gap(x)  # shape: (batch, 128, 1)
        x = x.view(x.size(0), -1)  # flatten to (batch, 128)
        # Dense layers
        x = F.gelu(self.fc1(x))
        x = self.dropout3(x)
        x = F.gelu(self.fc2(x))
        x = self.dropout4(x)
        x = self.fc3(x)
        #x = torch.sigmoid(self.fc3(x))
        return x
    
# ----------------------
# Transformer Model Class
# ----------------------
# Positional Encoding module (Vaswani et al., 2017)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        """
        Args:
            d_model: embedding dimension.
            dropout: dropout rate.
            max_len: maximum length of the input sequences.
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create constant 'pe' matrix with values dependent on position and dimension
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)   # even indices
        pe[:, 1::2] = torch.cos(position * div_term)   # odd indices
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch, seq_len, d_model)
        Returns:
            x after adding positional encodings and applying dropout.
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

# inspired from: Behinaein, B., Bhatti, A., Rodenburg, D., Hungler, P., Etemad, A.: 
# A Transformer Architecture for Stress Detection from ECG. 
# In: International Symposium on Wearable Computers, pp. 1--6 (2021). \doi{10.1145/3460421.3480427}
# Transformer-based classifier for ECG stress detection
class TransformerECGClassifier(nn.Module):
    def __init__(self, input_length=10000):
        """
        Args:
            input_length: Length of the input ECG signal.
                         For example, a 10 second window at 1000Hz gives 10000 samples.
        """
        super(TransformerECGClassifier, self).__init__()
        # Convolutional front-end subnetwork
        # For conv1: kernel_size=64, stride=8.
        # Use asymmetric padding: left=31, right=32.
        self.pad_conv1 = nn.ConstantPad1d((31, 32), 0)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=64, stride=8, padding=0)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # For conv2: kernel_size=32, stride=4.
        # Use asymmetric padding: left=15, right=16.
        self.pad_conv2 = nn.ConstantPad1d((15, 16), 0)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=32, stride=4, padding=0)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # For input_length=10000:
        # After conv1: output length = ceil(10000/8) = 1250
        # After pool1: output length = floor(1250/2) = 625
        # After conv2: output length = ceil(625/4) = 157
        # After pool2: output length = floor(157/2) = 78
        self.T = 74  # Final sequence length
        
        # Linear projection from conv output (128 channels) to transformer model dimension (1024)
        self.fc_embed = nn.Linear(128, 1024)
        
        # Positional encoder to inject order information into the embeddings
        self.pos_encoder = PositionalEncoding(d_model=1024, dropout=0.1, max_len=self.T)
        
        # Transformer encoder: 4 layers, with model dimension 1024, 4 attention heads, feed-forward dim 512, dropout 0.5
        encoder_layer = nn.TransformerEncoderLayer(d_model=1024, nhead=4, dim_feedforward=512, dropout=0.4)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)
        
        # Fully connected (FC) subnetwork for classification
        # Flattened transformer output has dimension T * 1024 (74 * 1024)
        self.fc1 = nn.Linear(self.T * 1024, 512)
        self.dropout_fc1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.dropout_fc2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, 1)
        
    def forward(self, x):
        """
        Args:
            x: Input ECG signal tensor of shape (batch, 1, input_length)
        Returns:
            Output probabilities of shape (batch, 1) via sigmoid activation.
        """
        # Convolutional front-end
        x = self.conv1(x)      # -> (batch, 64, L1)
        x = self.relu1(x)
        x = self.pool1(x)      # -> (batch, 64, L1/2)
        
        x = self.conv2(x)      # -> (batch, 128, L2)
        x = self.relu2(x)
        x = self.pool2(x)      # -> (batch, 128, T) where T ≈ self.T
        
        # Permute to (batch, T, channels)
        x = x.permute(0, 2, 1)  # -> (batch, T, 128)
        
        # Project to transformer model dimension (1024)
        x = self.fc_embed(x)    # -> (batch, T, 1024)
        
        # Add positional encoding
        x = self.pos_encoder(x)  # -> (batch, T, 1024)
        
        # Transformer expects input shape (seq_len, batch, d_model)
        x = x.permute(1, 0, 2)   # -> (T, batch, 1024)
        x = self.transformer_encoder(x)  # -> (T, batch, 1024)
        x = x.permute(1, 0, 2)   # -> (batch, T, 1024)
        
        # Flatten transformer output
        x = x.flatten(1)       # -> (batch, T * 1024)
        
        # Fully connected layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout_fc1(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout_fc2(x)
        x = self.fc3(x)
        return x
        # Sigmoid activation for binary classification
        #return torch.sigmoid(x)


# adapted from: Ingolfsson, T.M., Wang, X., Hersche, M., Burrello, A., Cavigelli, L., Benini, L.: 
# ECG-TCN: Wearable Cardiac Arrhythmia Detection with a Temporal Convolutional Network. 
# arXiv preprint arXiv:2103.13740 (2021). \url{https://arxiv.org/abs/2103.13740}
# ----------------------
# TCN Model Class
# ----------------------
class TCNClassifier(nn.Module):
    def __init__(self, input_length=10000, n_inputs=1, Kt=11, pt=0.3, Ft=11):
        super(TCNClassifier, self).__init__()
        # Initial 1x1 convolution to expand channels
        self.pad0 = nn.ConstantPad1d(padding=(Kt-1, 0), value=0)
        self.conv0 = nn.Conv1d(in_channels=n_inputs, out_channels=n_inputs + 1, kernel_size=Kt, bias=False)
        self.batchnorm0 = nn.BatchNorm1d(num_features=n_inputs + 1)
        self.act0 = nn.ReLU()
        
        # First residual block (dilation = 1)
        dilation = 1
        self.upsample = nn.Conv1d(in_channels=n_inputs + 1, out_channels=Ft, kernel_size=1, bias=False)
        self.upsamplebn = nn.BatchNorm1d(num_features=Ft)
        self.upsamplerelu = nn.ReLU()
        self.pad1 = nn.ConstantPad1d(padding=((Kt-1) * dilation, 0), value=0)
        self.conv1 = nn.Conv1d(in_channels=n_inputs + 1, out_channels=Ft, kernel_size=Kt, dilation=dilation, bias=False)
        self.batchnorm1 = nn.BatchNorm1d(num_features=Ft)
        self.act1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=pt)
        self.pad2 = nn.ConstantPad1d(padding=((Kt-1) * dilation, 0), value=0)
        self.conv2 = nn.Conv1d(in_channels=Ft, out_channels=Ft, kernel_size=Kt, dilation=dilation, bias=False)
        self.batchnorm2 = nn.BatchNorm1d(num_features=Ft)
        self.act2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=pt)
        self.reluadd1 = nn.ReLU()
        
        # Second residual block (dilation = 2)
        dilation = 2
        self.pad3 = nn.ConstantPad1d(padding=((Kt-1) * dilation, 0), value=0)
        self.conv3 = nn.Conv1d(in_channels=Ft, out_channels=Ft, kernel_size=Kt, dilation=dilation, bias=False)
        self.batchnorm3 = nn.BatchNorm1d(num_features=Ft)
        self.act3 = nn.ReLU()
        self.dropout3 = nn.Dropout(p=pt)
        self.pad4 = nn.ConstantPad1d(padding=((Kt-1) * dilation, 0), value=0)
        self.conv4 = nn.Conv1d(in_channels=Ft, out_channels=Ft, kernel_size=Kt, dilation=dilation, bias=False)
        self.batchnorm4 = nn.BatchNorm1d(num_features=Ft)
        self.act4 = nn.ReLU()
        self.dropout4 = nn.Dropout(p=pt)
        self.reluadd2 = nn.ReLU()
        
        # Third residual block (dilation = 4)
        dilation = 4
        self.pad5 = nn.ConstantPad1d(padding=((Kt-1) * dilation, 0), value=0)
        self.conv5 = nn.Conv1d(in_channels=Ft, out_channels=Ft, kernel_size=Kt, dilation=dilation, bias=False)
        self.batchnorm5 = nn.BatchNorm1d(num_features=Ft)
        self.act5 = nn.ReLU()
        self.dropout5 = nn.Dropout(p=pt)
        self.pad6 = nn.ConstantPad1d(padding=((Kt-1) * dilation, 0), value=0)
        self.conv6 = nn.Conv1d(in_channels=Ft, out_channels=Ft, kernel_size=Kt, dilation=dilation, bias=False)
        self.batchnorm6 = nn.BatchNorm1d(num_features=Ft)
        self.act6 = nn.ReLU()
        self.dropout6 = nn.Dropout(p=pt)
        self.reluadd3 = nn.ReLU()
        
        # Final linear layer: flattened feature map has shape Ft * input_length
        flattened_size = Ft * input_length  
        self.linear = nn.Linear(in_features=flattened_size, out_features=1, bias=False)
        
    def forward(self, x):
        # Input shape: (batch, channels, sequence_length)
        x = self.pad0(x)
        x = self.conv0(x)
        x = self.batchnorm0(x)
        x = self.act0(x)
        
        # First residual block
        res = self.pad1(x)
        res = self.conv1(res)
        res = self.batchnorm1(res)
        res = self.act1(res)
        res = self.dropout1(res)
        res = self.pad2(res)
        res = self.conv2(res)
        res = self.batchnorm2(res)
        res = self.act2(res)
        res = self.dropout2(res)
        
        x = self.upsample(x)
        x = self.upsamplebn(x)
        x = self.upsamplerelu(x)
        x = x + res
        x = self.reluadd1(x)
        
        # Second residual block
        res = self.pad3(x)
        res = self.conv3(res)
        res = self.batchnorm3(res)
        res = self.act3(res)
        res = self.dropout3(res)
        res = self.pad4(res)
        res = self.conv4(res)
        res = self.batchnorm4(res)
        res = self.act4(res)
        res = self.dropout4(res)
        x = x + res
        x = self.reluadd2(x)
        
        # Third residual block
        res = self.pad5(x)
        res = self.conv5(res)
        res = self.batchnorm5(res)
        res = self.act5(res)
        res = self.dropout5(res)
        res = self.pad6(res)
        res = self.conv6(res)
        res = self.batchnorm6(res)
        res = self.act6(res)
        res = self.dropout6(res)
        x = x + res
        x = self.reluadd3(x)
        
        # Flatten and classify
        x = x.flatten(1)
        x = self.linear(x)
        #return torch.sigmoid(x)
        return x
    
# -----------------------------------------------
# Helper: threshold sweep
# -----------------------------------------------
def find_best_threshold(
    probs:  np.ndarray,
    labels: np.ndarray,
    num_classes: int = 2,
    average: str = "macro",
    grid: int = 101
):
    """
    Scan `grid` equally–spaced thresholds in (0,1) and return the one that
    maximises macro-F1.  Returns (best_threshold, best_f1).
    """
    ts = np.linspace(0.0, 1.0, grid, endpoint=False)[1:]   # skip 0
    best_t = 0.5
    best_f1 = -1.0
    labels_t = torch.from_numpy(labels.astype(np.int64))

    for t in ts:
        preds = (probs >= t).astype(np.int64)
        f1 = multiclass_f1_score(
                        torch.from_numpy(preds),
                        labels_t,
                        num_classes=num_classes,
                        average=average,
                    ).item()
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, best_f1

# -------------------------------------------
# TRAIN (one epoch) + VALIDATION
# -------------------------------------------
def train_one_epoch(
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_fn,
    device,
    epoch,
    best_threshold_so_far=0.5,
    best_f1_so_far=-1.0,
    log_interval=100,
):
    model.train()
    run_loss = 0.0
    correct = 0
    n_seen = 0

    auroc_tr = BinaryAUROC()
    pr_tr = BinaryAveragePrecision()

    for batch_idx, (x, y) in enumerate(train_loader):
        x = x.to(device).permute(0, 2, 1)     # (B,C,L)
        y= y.to(device).float()

        optimizer.zero_grad(set_to_none=True)
        out = model(x).view(-1)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()

        run_loss += loss.item() * x.size(0)
        n_seen += x.size(0)

        probs = torch.sigmoid(out).detach()
        preds = (probs > 0.5).float()
        correct += preds.eq(y).sum().item()

        auroc_tr.update(probs.cpu(), y.cpu().int())
        pr_tr.update(probs.cpu(), y.cpu().int())

        if batch_idx % log_interval == 0:
            print(f"[Train] Epoch {epoch}  batch {batch_idx}/{len(train_loader)}  "
                  f"loss={run_loss/n_seen:.4f}")

    # train epoch metrics
    train_loss = run_loss / n_seen
    train_acc  = correct  / n_seen
    train_auroc = auroc_tr.compute().item()
    train_pr_auc = pr_tr.compute().item()

    mlflow.log_metrics({
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "train_auc_roc": train_auroc,
        "train_pr_auc": train_pr_auc,
    }, step=epoch)

    # VALIDATION  
    model.eval()
    val_probs, val_labels = [], []
    auroc_val = BinaryAUROC()
    pr_val = BinaryAveragePrecision()

    val_loss_total = 0.0
    n_val = 0

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device).permute(0, 2, 1)
            y = y.to(device).float()
            out = model(x).view(-1)
            probs = torch.sigmoid(out)

            # compute and accumulate BCE loss
            val_loss_total += loss_fn(out, y).item() * x.size(0)
            n_val += x.size(0)

            val_probs.append(probs.cpu().numpy())
            val_labels.append(y.cpu().numpy())

            auroc_val.update(probs.cpu(), y.cpu().int())
            pr_val.update(probs.cpu(), y.cpu().int())

    val_probs = np.concatenate(val_probs)
    val_labels = np.concatenate(val_labels)

    val_loss = val_loss_total / n_val

    # threshold sweep on this epoch
    this_t, this_f1 = find_best_threshold(val_probs, val_labels)

    if this_f1 > best_f1_so_far:
        best_f1_so_far = this_f1
        best_threshold_so_far = this_t

    # metrics at the chosen threshold of THIS epoch
    val_preds = (val_probs >= this_t).astype(float)
    val_acc = (val_preds == val_labels).mean()
    val_f1 = this_f1
    val_auroc = auroc_val.compute().item()
    val_pr_auc= pr_val.compute().item()

    mlflow.log_metrics({
        "val_loss": val_loss,
        "val_accuracy": val_acc,
        "val_auc_roc": val_auroc,
        "val_pr_auc": val_pr_auc,
        "val_f1": val_f1,
        "val_best_threshold": this_t,
    }, step=epoch)

    print(f"[Val ] Epoch {epoch}  acc={val_acc:.4f}  auc={val_auroc:.4f}  "
        f"f1*={val_f1:.4f} @ t={this_t:.2f}  loss={val_loss:.4f}")

    return val_loss, best_threshold_so_far, best_f1_so_far

# ------------------------------------------
# TEST
# ------------------------------------------
def test(
    model,
    test_loader,
    device,
    threshold,
    loss_fn=None  
):
    model.eval()
    probs_all, labels_all = [], []
    auroc_te = BinaryAUROC()
    pr_te = BinaryAveragePrecision()
    total_loss = 0.0
    n_seen = 0

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device).permute(0, 2, 1)
            y = y.to(device).float()
            out = model(x).view(-1)
            probs = torch.sigmoid(out)

            probs_all.append(probs.cpu().numpy())
            labels_all.append(y.cpu().numpy())

            auroc_te.update(probs.cpu(), y.cpu().int())
            pr_te.update(probs.cpu(), y.cpu().int())

            if loss_fn is not None:
                total_loss += loss_fn(out, y).item() * x.size(0)
                n_seen += x.size(0)

    probs_all  = np.concatenate(probs_all)
    labels_all = np.concatenate(labels_all).astype(np.int64)
    preds_all  = (probs_all >= threshold).astype(np.int64)
    
    acc = (preds_all == labels_all).mean()
    f1 = multiclass_f1_score(
             torch.from_numpy(preds_all),
             torch.from_numpy(labels_all),
             num_classes=2,
             average="macro"
           ).item()
    auroc = auroc_te.compute().item()
    prauc = pr_te.compute().item()
    loss = (total_loss / n_seen) if loss_fn is not None else np.nan

    mlflow.log_metrics({
        "test_loss": loss,
        "test_accuracy": acc,
        "test_auc_roc": auroc,
        "test_pr_auc": prauc,
        "test_f1": f1,
        "test_threshold": threshold,
    })

    print(f"[Test]  acc={acc:.4f}  auc={auroc:.4f}  pr_auc={prauc:.4f}  "
          f"f1*={f1:.4f} @ t={threshold:.2f}")
    return loss, acc, auroc, prauc, f1

class EarlyStopping:
    """
    Early stopping to prevent overfitting.
    Stops training when validation loss doesn't improve for 'patience' epochs.
    """
    def __init__(self, patience=3):
        self.patience = patience
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        
def log_model_summary(model, input_size):
    # Write model summary to a file and log as an artifact
    with tempfile.TemporaryDirectory() as tmp_dir:
        summmary_path = Path(tmp_dir) / "model_summary.txt"
        with open(summmary_path, "w") as f:
            f.write(str(summary(model, input_size=input_size)))
        mlflow.log_artifact(summmary_path)
        
def prepare_model_signature(model, sample_input):
    """Prepare model signature for MLflow registration"""
    model.cpu().eval()
    with torch.no_grad():
        tensor_input = torch.tensor(
            sample_input, 
            dtype=torch.float32
        ).permute(0, 2, 1)
        sample_output = model(tensor_input).numpy()
    
    input_schema = Schema([
        TensorSpec(np.dtype(np.float32), (-1, sample_input.shape[1], 1))
    ])
    output_schema = Schema([
        TensorSpec(np.dtype(np.float32), (-1, 1))
    ])
    
    return ModelSignature(inputs=input_schema, outputs=output_schema)

# def set_seed(seed=42, deterministic=True):
#     """
#     Set seeds for reproducibility.
    
#     Args:
#         seed (int): Seed number
#         deterministic (bool): If True, use deterministic algorithms
#     """
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
    
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed(seed)
#         torch.cuda.manual_seed_all(seed)  # for multi-GPU
        
#         if deterministic:
#             # Ensure deterministic behavior
#             torch.backends.cudnn.deterministic = True
#             torch.backends.cudnn.benchmark = False

def set_seed(seed=42, deterministic=True):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False
        
        if hasattr(torch, "use_deterministic_algorithms"):
            torch.use_deterministic_algorithms(True, warn_only=True)


class Bottleneck1D(nn.Module):
    """
    A 1D version of the ResNet bottleneck block.
    This block uses a 1x1 conv to reduce channels, a 3x3 conv for processing,
    and a final 1x1 conv to expand channels. If needed, a downsample layer is used.
    """
    expansion = 4

    def __init__(self, in_channels, planes, stride=1, downsample=None):
        super(Bottleneck1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = nn.Conv1d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)

        return out

class XResNet1D(nn.Module):
    """
    An XResNet1D architecture for 1D signals.
    It uses a modified stem (three 1D convolutional layers with batch norm and ReLU)
    followed by 4 stages of bottleneck blocks. For ResNet101, the layer configuration is [3, 4, 23, 3].
    """
    def __init__(self, block, layers, num_classes=1, in_channels=1):
        super(XResNet1D, self).__init__()
        self.inplanes = 64

        # Stem: Adapted from fastai's xresnet stem but for 1D input.
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )

        # Residual layers (each layer downsamples and increases feature channels)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # Global average pooling and final classifier
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.sigmoid = nn.Sigmoid()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, downsample=downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        # x is expected to be of shape (batch_size, channels, sequence_length)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)  # shape: (batch_size, features, 1)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.sigmoid(x)  # For binary classification
        return x

def xresnet1d101(num_classes=1, in_channels=1):
    """
    Constructs an xresnet1d101 model.
    
    For ResNet101, the block configuration is [3, 4, 23, 3] using Bottleneck1D blocks.
    """
    return XResNet1D(Bottleneck1D, [3, 4, 23, 3], num_classes=num_classes, in_channels=in_channels)
