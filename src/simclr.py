import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import cv2
from torch.utils.data import Dataset, DataLoader

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
    def __init__(self, input_channels=1, dropout=0.3, window=10000):
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
        self.fc = nn.Linear(128 *  (window // 8), 80) 
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

def get_simclr_model(window:int=2560, device:str="cpu"):
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
        xb = xb.unsqueeze(1).to(device) if xb.ndim==2 else xb.to(device)
        h = F.normalize(model.f(xb), dim=1)
        reps.append(h.cpu().numpy())
    return np.concatenate(reps,0)