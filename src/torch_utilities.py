import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchinfo import summary
import mlflow
import mlflow.pytorch

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

class Improved1DCNN(nn.Module):
    def __init__(self):
        super(Improved1DCNN, self).__init__()
        self.bn_input = nn.BatchNorm1d(1)
        # Block 1
        self.conv1_1 = nn.Conv1d(1, 32, kernel_size=5, padding=2)
        self.conv1_2 = nn.Conv1d(32, 32, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.dropout1 = nn.Dropout(0.1)
        # Block 2
        self.conv2_1 = nn.Conv1d(32, 64, kernel_size=11, padding=5)
        self.conv2_2 = nn.Conv1d(64, 64, kernel_size=11, padding=5)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(0.1)
        # Block 3
        self.conv3_1 = nn.Conv1d(64, 128, kernel_size=17, padding=8)
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
    model.train()
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        # Data comes in shape (batch, window_length, 1) -> permute to (batch, 1, window_length)
        data = data.to(device).permute(0, 2, 1)
        # For binary classification using BCELoss, convert target to float
        target = target.to(device).float()
        optimizer.zero_grad()
        output = model(data)
        # Flatten output from (batch, 1) to (batch,)
        output = output.view(-1)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        if batch_idx % log_interval == 0:
            avg_loss = running_loss / (batch_idx + 1)
            mlflow.log_metric("train_loss", avg_loss, step=epoch * len(train_loader) + batch_idx)
            print(f"Epoch {epoch} Batch {batch_idx}/{len(train_loader)} Loss: {avg_loss:.4f}")

def test(model, data_loader, loss_fn, device):
    model.eval()
    test_loss = 0.0
    correct = 0
    total_samples = 0
    with torch.no_grad():
        for data, target in data_loader:
            data = data.to(device).permute(0, 2, 1)
            target = target.to(device).float()
            output = model(data)
            output = output.view(-1)
            loss = loss_fn(output, target)
            test_loss += loss.item() * data.size(0)
            # Use 0.5 threshold for binary prediction
            preds = (output > 0.5).float()
            correct += preds.eq(target).sum().item()
            total_samples += data.size(0)
    
    avg_loss = test_loss / total_samples
    accuracy = correct / total_samples
    mlflow.log_metric("test_loss", avg_loss)
    mlflow.log_metric("test_accuracy", accuracy)
    print(f"Test set: Average loss: {avg_loss:.4f}, Accuracy: {accuracy*100:.2f}%")
    return accuracy

def log_model_summary(model, input_size):
    # Write model summary to a file and log as an artifact
    with open("model_summary.txt", "w") as f:
        f.write(str(summary(model, input_size=input_size)))
    mlflow.log_artifact("model_summary.txt")