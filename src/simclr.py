import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import cv2
from torch.utils.data import Dataset

# ----------------------------------------------------------------------
# Augmentations
# ----------------------------------------------------------------------
def add_noise(signal, noise_amount):
    noise = np.random.normal(0, noise_amount, size=signal.shape)
    return signal + noise


def add_noise_with_SNR(signal, snr_db):
    x_watts = signal ** 2
    sig_avg_watts = np.mean(x_watts)
    sig_avg_db = 10 * np.log10(sig_avg_watts)
    noise_avg_db = sig_avg_db - snr_db
    noise_avg_watts = 10 ** (noise_avg_db / 10)
    noise = np.random.normal(0, math.sqrt(noise_avg_watts), size=signal.shape)
    return signal + noise


def scaled(signal, factor):
    return signal * factor


def negate(signal):
    return -signal


def hor_flip(signal):
    return np.flip(signal)


def permute(signal, pieces):
    L = signal.shape[-1]
    piece_len = L // pieces
    segments = [signal[i*piece_len:(i+1)*piece_len] for i in range(pieces)]
    np.random.shuffle(segments)
    return np.concatenate(segments, axis=-1)


def time_warp(signal, pieces=4, stretch_factor=1.2, squeeze_factor=0.8):
    L = signal.shape[-1]
    seg_len = L // pieces
    output = []
    for i in range(pieces):
        seg = signal[i*seg_len:(i+1)*seg_len]
        if np.random.rand() < 0.5:
            new_len = int(np.ceil(len(seg) * stretch_factor))
        else:
            new_len = int(np.ceil(len(seg) * squeeze_factor))
        seg = seg.reshape(-1,1)
        warped = cv2.resize(seg, (1, new_len), interpolation=cv2.INTER_LINEAR).flatten()
        output.append(warped)
    return np.concatenate(output, axis=-1)


# List of available augmentations
AUGMENTATIONS = [
    lambda x: add_noise(x, noise_amount=0.01),
    lambda x: add_noise_with_SNR(x, snr_db=20),
    lambda x: scaled(x, factor=np.random.uniform(0.9,1.1)),
    lambda x: negate(x),
    lambda x: hor_flip(x),
    lambda x: permute(x, pieces=4),
    lambda x: time_warp(x, pieces=4, stretch_factor=1.2, squeeze_factor=0.8)
]


def DataTransform(signal):
    """Generate two augmented views of the same ECG signal."""
    view1 = signal.copy()
    view2 = signal.copy()
    # apply 2 random augmentations to each view
    for _ in range(2):
        aug = np.random.choice(AUGMENTATIONS)
        view1 = aug(view1)
        aug = np.random.choice(AUGMENTATIONS)
        view2 = aug(view2)
    return view1, view2

# ----------------------------------------------------------------------
# Encoder f(.)
# ----------------------------------------------------------------------
class ECGEncoder(nn.Module):
    """1D CNN encoder with 3 blocks, each block: Conv1d -> ReLU -> MaxPool -> Dropout"""
    def __init__(self, input_channels=1, dropout=0.3):
        super(ECGEncoder, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=32, stride=1, padding=16, bias=False),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(dropout)
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=16, stride=1, padding=8, bias=False),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(dropout)
        )
        self.block3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=8, stride=1, padding=4, bias=False),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(dropout)
        )
        # After conv blocks, flatten and FC to 80 units
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(128 *  (2560 // 8), 80)  # adjust 2560/window size accordingly
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
    """Non-linear projection to 256 dims with softmax activation"""
    def __init__(self, input_dim=80, proj_dim=256):
        super(ProjectionHead, self).__init__()
        self.fc = nn.Linear(input_dim, proj_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, h):
        z = self.fc(h)
        z = self.softmax(z)
        return z

# ----------------------------------------------------------------------
# SimCLR Model
# ----------------------------------------------------------------------
class SimCLR(nn.Module):
    def __init__(self, encoder, projector):
        super(SimCLR, self).__init__()
        self.encoder = encoder
        self.projector = projector

    def forward(self, x_i, x_j):
        # x_i, x_j: two views, shape (N,C,L)
        h_i = self.encoder(x_i)
        h_j = self.encoder(x_j)
        z_i = self.projector(h_i)
        z_j = self.projector(h_j)
        return h_i, h_j, z_i, z_j

# ----------------------------------------------------------------------
# NT-Xent Loss
# ----------------------------------------------------------------------
class NTXentLoss(nn.Module):
    def __init__(self, batch_size, temperature=0.5):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, z_i, z_j):
        N = 2 * self.batch_size
        z = torch.cat([z_i, z_j], dim=0)  # shape (2N, D)
        sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2) / self.temperature
        labels = torch.arange(self.batch_size).repeat(2)
        labels = labels.to(z_i.device)
        mask = (~torch.eye(N, N, dtype=bool)).float().to(z_i.device)
        logits = sim * mask - 1e9 * (1 - mask)
        loss = self.cross_entropy(logits, labels)
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