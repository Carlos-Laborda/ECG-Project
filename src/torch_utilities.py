import tempfile
from pathlib import Path

import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchinfo import summary

# MLflow imports
import mlflow
import mlflow.pytorch
from mlflow.types import Schema, TensorSpec
from mlflow.models import ModelSignature

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
                windows = f[participant_key][cat][:]
                n_windows = windows.shape[0]
                groups_arr = np.array([participant_id] * n_windows, dtype=object)

                X_list.append(windows)
                y_list.append(np.full((n_windows,), label_map[cat], dtype=int))
                groups_list.append(groups_arr)

    if len(X_list) == 0:
        raise ValueError(f"No valid data found in {hdf5_path} with label_map {label_map}.")

    # Concatenate all data
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
        super(Improved1DCNN, self).__init__()
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

# ----------------------
# Training and Evaluation Functions
# ----------------------
def train(model, train_loader, optimizer, loss_fn, device, epoch, log_interval=100):
    """Train for one epoch and log metrics"""
    model.train()
    running_loss = 0.0
    correct = 0
    total_samples = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        # Data preprocessing
        data = data.to(device).permute(0, 2, 1)
        target = target.to(device).float()
        
        # Forward pass
        optimizer.zero_grad(set_to_none=True)
        output = model(data)
        output = output.view(-1)
        
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
    mlflow.log_metrics({
        "train_loss": epoch_loss,
        "train_accuracy": epoch_acc
    }, step=epoch)
    
    return epoch_loss, epoch_acc

def test(model, data_loader, loss_fn, device, phase='val', epoch=None):
    """Evaluate model and log metrics. Phase can be 'val' or 'test'"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for data, target in data_loader:
            # Data preprocessing
            data = data.to(device).permute(0, 2, 1)
            target = target.to(device).float()
            
            # Forward pass
            output = model(data)
            output = output.view(-1)
            loss = loss_fn(output, target)
            
            # Accumulate metrics
            running_loss += loss.item() * data.size(0)
            preds = (output > 0.5).float()
            correct += preds.eq(target).sum().item()
            total_samples += data.size(0)
    
    # Calculate final metrics
    avg_loss = running_loss / total_samples
    accuracy = correct / total_samples
    
    # Log metrics with appropriate names based on phase
    metrics = {
        f"{phase}_loss": avg_loss,
        f"{phase}_accuracy": accuracy
    }
    
    # For validation, include epoch for tracking
    if epoch is not None:
        mlflow.log_metrics(metrics, step=epoch)
    else:
        mlflow.log_metrics(metrics)
    
    print(f"{phase.capitalize()} set: Average loss: {avg_loss:.4f}, "
          f"Accuracy: {accuracy*100:.2f}%")
    
    return avg_loss, accuracy
        
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