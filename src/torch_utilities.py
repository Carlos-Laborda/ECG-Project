import tempfile
from pathlib import Path

import numpy as np
import h5py
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torcheval.metrics import BinaryAUROC
from torchinfo import summary
from sklearn.metrics import roc_curve, auc


# MLflow imports
import mlflow
import mlflow.pytorch
from mlflow.types import Schema, TensorSpec
from mlflow.models import ModelSignature

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
        label = torch.tensor(label, dtype=torch.long)
        return sample, label


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
        x = torch.sigmoid(self.fc3(x))
        return x

# CNN from SSL-ECG paper
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
    
# ----------------------
# Transformer Model Class
# ----------------------

# ----------------------
# TCN Model Class
# ----------------------

class TCNClassifier(nn.Module):
    def __init__(self, input_length=10000, n_inputs=1, Kt=11, pt=0.3, Ft=11):
        super(TCNClassifier, self).__init__()
        self.bn_input = nn.BatchNorm1d(n_inputs)
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
        x = self.bn_input(x)
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
        return torch.sigmoid(x)

# ----------------------
# Training and Evaluation Functions
# ----------------------
def train(model, train_loader, optimizer, loss_fn, device, epoch, log_interval=100):
    """Train for one epoch and log metrics"""
    model.train()
    running_loss = 0.0
    correct = 0
    total_samples = 0
    
    # Initialize metrics
    auroc_metric = BinaryAUROC()
    
    for batch_idx, (data, target) in enumerate(train_loader):
        # Data preprocessing
        data = data.to(device).permute(0, 2, 1)
        target = target.to(device).float()
        
        # Forward pass
        optimizer.zero_grad(set_to_none=True)
        output = model(data)
        output = output.view(-1)
        
        # Update AUROC metric
        auroc_metric.update(output.detach().cpu(), target.cpu().int())
        
        # Calculate loss and backpropagate
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        
        # Accumulate metrics
        running_loss += loss.item() * data.size(0)
        preds = (output > 0.5).float()
        correct += preds.eq(target).sum().item()
        total_samples += data.size(0)
        
        # Log intermediate training loss
        if batch_idx % log_interval == 0:
            batch_avg_loss = running_loss / total_samples
            print(f"Epoch {epoch} [{batch_idx}/{len(train_loader)}] "
                  f"Loss: {batch_avg_loss:.4f}")
    
    # Log epoch metrics
    epoch_loss = running_loss / total_samples
    epoch_acc = correct / total_samples
    auc_roc = auroc_metric.compute().item()
    mlflow.log_metrics({
        "train_loss": epoch_loss,
        "train_accuracy": epoch_acc,
        "train_auc_roc": auc_roc
    }, step=epoch)
    
    return epoch_loss, epoch_acc, auc_roc

def test(model, data_loader, loss_fn, device, phase='val', epoch=None):
    """Evaluate model and log metrics. Phase can be 'val' or 'test'"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total_samples = 0
    
    # Initialize metrics
    auroc_metric = BinaryAUROC()
    
    # Lists to store all predictions and labels for ROC curve
    all_preds = []
    all_labels = []
    is_test_phase = (phase == 'test')
    
    with torch.no_grad():
        for data, target in data_loader:
            # Data preprocessing
            data = data.to(device).permute(0, 2, 1)
            target = target.to(device).float()
            
            # Forward pass
            output = model(data)
            output = output.view(-1)
            
            # Update AUROC metric
            auroc_metric.update(output.cpu(), target.cpu().int())
            
            # Store predictions and labels for ROC curve
            if is_test_phase:
                all_preds.extend(output.cpu().numpy())
                all_labels.extend(target.cpu().numpy())
            
            # Calculate loss
            loss = loss_fn(output, target)
            
            # Accumulate metrics
            running_loss += loss.item() * data.size(0)
            preds = (output > 0.5).float()
            correct += preds.eq(target).sum().item()
            total_samples += data.size(0)
    
    # Calculate final metrics
    avg_loss = running_loss / total_samples
    accuracy = correct / total_samples
    auc_roc = auroc_metric.compute().item()
    
    # Convert lists to numpy arrays
    if is_test_phase:
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
    
        # Calculate ROC curve points
        fpr, tpr, thresholds = roc_curve(all_labels, all_preds)
        
        # Log ROC curve data as JSON artifact
        roc_data = {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": thresholds.tolist(),
            "auc": auc_roc
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json') as f:
            json.dump(roc_data, f)
            f.flush()
            
            # Log the file with a name based on phase and epoch
            if epoch is not None:
                mlflow.log_artifact(f.name, f"{phase}_roc_data_epoch_{epoch}.json")
            else:
                mlflow.log_artifact(f.name, f"{phase}_roc_data.json")
    
    # Log metrics with appropriate names based on phase
    metrics = {
        f"{phase}_loss": avg_loss,
        f"{phase}_accuracy": accuracy,
        f"{phase}_auc_roc": auc_roc
    }
    
    # For validation, include epoch for tracking
    if epoch is not None:
        mlflow.log_metrics(metrics, step=epoch)
    else:
        mlflow.log_metrics(metrics)
    
    print(f"{phase.capitalize()} set: Average loss: {avg_loss:.4f}, "
          f"Accuracy: {accuracy*100:.2f}%, AUC-ROC: {auc_roc:.4f}")
    
    return avg_loss, accuracy, auc_roc

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

def set_seed(seed=42, deterministic=True):
    """
    Set seeds for reproducibility.
    
    Args:
        seed (int): Seed number
        deterministic (bool): If True, use deterministic algorithms
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU
        
        if deterministic:
            # Ensure deterministic behavior
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

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
