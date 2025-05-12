import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import random
import pandas as pd
import os
import sys
import logging
import mlflow
import tqdm

from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.metrics import classification_report, cohen_kappa_score, confusion_matrix, accuracy_score
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision, BinaryF1Score
from shutil import copy
from einops import rearrange, repeat
from typing import Union, Sequence
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from tslearn.metrics import dtw, dtw_path,gak

# ----------------------------------------------------------------------
# augmentations.py
# ----------------------------------------------------------------------
def DataTransform(sample, config):
    weak_aug = scaling(sample, config.augmentation.jitter_scale_ratio)
    strong_aug = jitter(permutation(sample, max_segments=config.augmentation.max_seg), config.augmentation.jitter_ratio)
    return weak_aug, strong_aug


def jitter(x, sigma=0.8):
    # https://arxiv.org/pdf/1706.00527.pdf
    return x + np.random.normal(loc=0., scale=sigma, size=x.shape)


def scaling(x, sigma=1.1):
    # https://arxiv.org/pdf/1706.00527.pdf
    factor = np.random.normal(loc=2., scale=sigma, size=(x.shape[0], x.shape[2]))
    ai = []
    for i in range(x.shape[1]):
        xi = x[:, i, :]
        ai.append(np.multiply(xi, factor[:, :])[:, np.newaxis, :])
    return np.concatenate((ai), axis=1)


def permutation(x, max_segments=5, seg_mode="random"):
    x_np = x if isinstance(x, np.ndarray) else x.numpy()
    N, C, L = x_np.shape
    orig_steps = np.arange(L)

    ret = np.empty_like(x_np)                 # (N, C, L)
    for i in range(N):
        # 1) choose how many segments to split into
        n_seg = np.random.randint(1, max_segments + 1)
        if n_seg > 1:
            # 2) split & permute indices
            if seg_mode == "random":
                split_points = np.random.choice(L - 2, n_seg - 1, replace=False) + 1
                split_points.sort()
                splits = np.split(orig_steps, split_points)
            else:
                splits = np.array_split(orig_steps, n_seg)
            order = np.random.permutation(len(splits))
            warp = np.concatenate([splits[o] for o in order])
        else:
            warp = orig_steps

        # 3) ***transpose*** to (C, L) before writing
        ret[i] = x_np[i, :, warp].T

    return ret

# ----------------------------------------------------------------------
# dataloader.py
# ----------------------------------------------------------------------
def get_DTW(UTS_tr):
    N = len(UTS_tr)
    dist_mat = np.zeros((N,N))
    for i in tqdm.tqdm(range(N)):
        for j in range(N):
            if i>j:
                dist = dtw(UTS_tr[i].reshape(-1,1), UTS_tr[j].reshape(-1,1))
                dist_mat[i,j] = dist
                dist_mat[j,i] = dist
            elif i==j:
                dist_mat[i,j] = 0
            else :
                pass
    return dist_mat

def get_TAM(UTS_tr):
    N = len(UTS_tr)
    dist_mat = np.zeros((N,N))
    for i in tqdm.tqdm(range(N)):
        for j in range(N):
            if i>j:
                k = dtw_path(UTS_tr[i].reshape(-1,1), 
                             UTS_tr[j].reshape(-1,1))[0]
                p = [np.array([i[0] for i in k]),
                     np.array([i[1] for i in k])]
                dist = tam(p)
                dist_mat[i,j] = dist
                dist_mat[j,i] = dist
            elif i==j:
                dist_mat[i,j] = 0
            else :
                pass
    return dist_mat

def get_GAK(UTS_tr):
    N = len(UTS_tr)
    dist_mat = np.zeros((N,N))
    for i in tqdm.tqdm(range(N)):
        for j in range(N):
            if i>j:
                dist = gak(UTS_tr[i].reshape(-1,1), 
                           UTS_tr[j].reshape(-1,1))
                dist_mat[i,j] = dist
                dist_mat[j,i] = dist
            elif i==j:
                dist_mat[i,j] = 0
            else :
                pass
    return dist_mat


def get_MDTW(MTS_tr):
    N = MTS_tr.shape[0]
    dist_mat = np.zeros((N,N))
    for i in tqdm.tqdm(range(N)):
        for j in range(N):
            if i>j:
                mdtw_dist = dtw(MTS_tr[i], MTS_tr[j])
                dist_mat[i,j] = mdtw_dist
                dist_mat[j,i] = mdtw_dist
            elif i==j:
                dist_mat[i,j] = 0
            else :
                pass
    return dist_mat

def get_COS(MTS_tr):
    cos_sim_matrix = -cosine_similarity(MTS_tr)
    return cos_sim_matrix

def get_EUC(MTS_tr):
    return euclidean_distances(MTS_tr)

def save_sim_mat(X_tr, min_ = 0, max_ = 1, multivariate=False, type_='DTW'):
    N = dist_mat.shape[0]
    if multivariate:
        assert type=='DTW'
        dist_mat = get_MDTW(X_tr)
    else:
        if type_=='DTW':
            dist_mat = get_DTW(X_tr)
        elif type_=='TAM':
            dist_mat = get_TAM(X_tr)
        elif type_=='COS':
            dist_mat = get_COS(X_tr)
        elif type_=='EUC':
            dist_mat = get_EUC(X_tr)
        elif type_=='GAK':
            dist_mat = get_GAK(X_tr)
        
    # (1) distance matrix
    diag_indices = np.diag_indices(N)
    mask = np.ones(dist_mat.shape, dtype=bool)
    mask[diag_indices] = False
    temp = dist_mat[mask].reshape(N, N-1)
    dist_mat[diag_indices] = temp.min()
    
    # (2) normalized distance matrix
    scaler = MinMaxScaler(feature_range=(min_, max_))
    dist_mat = scaler.fit_transform(dist_mat)
    
    # (3) normalized similarity matrix
    return 1 - dist_mat 

class Load_Dataset(Dataset):
    def __init__(self, dataset, config, training_mode):
        super().__init__()
        self.training_mode = training_mode

        X_train = dataset["samples"]
        y_train = dataset["labels"]

        # ─────────────────────────────────────────────────────
        # 1) ensure a channel-dim exists
        # ─────────────────────────────────────────────────────
        if len(X_train.shape) < 3:           # (N, L)  ->  (N, L, 1)
            if isinstance(X_train, np.ndarray):
                X_train = np.expand_dims(X_train, 2)
            else:  # torch.Tensor
                X_train = X_train.unsqueeze(2)

        # ─────────────────────────────────────────────────────
        # 2) make channel dimension second: (N, L, C) → (N, C, L)
        # ─────────────────────────────────────────────────────
        if X_train.shape.index(min(X_train.shape)) != 1:
            if isinstance(X_train, np.ndarray):
                X_train = np.transpose(X_train, (0, 2, 1))
            else:                             # torch.Tensor
                X_train = X_train.permute(0, 2, 1)

        # ─────────────────────────────────────────────────────
        # 3) final conversion to torch tensors
        # ─────────────────────────────────────────────────────
        if isinstance(X_train, np.ndarray):
            self.x_data = torch.from_numpy(X_train)
            self.y_data = torch.from_numpy(y_train).long()
        else:
            self.x_data = X_train
            self.y_data = y_train

        self.len = self.x_data.shape[0]

        # Augmentations only for self-supervised mode
        if training_mode == "self_supervised":
            aug1_np, aug2_np = DataTransform(self.x_data.numpy(), config)
            self.aug1 = torch.from_numpy(aug1_np).float()
            self.aug2 = torch.from_numpy(aug2_np).float()

    def __getitem__(self, index):
        if self.training_mode == "self_supervised":
            return self.x_data[index], self.y_data[index], self.aug1[index], self.aug2[index]
        else:
            return self.x_data[index], self.y_data[index], self.x_data[index], self.x_data[index]

    def __len__(self):
        return self.len

def data_generator_from_arrays(
    X_train, y_train, X_val, y_val, X_test, y_test,
    configs, training_mode
):
    
    train_dict = {"samples": X_train, "labels": y_train}
    val_dict   = {"samples": X_val,   "labels": y_val}
    test_dict  = {"samples": X_test,  "labels": y_test}

    # Create Dataset objects
    train_ds = Load_Dataset(train_dict, configs, training_mode)
    val_ds   = Load_Dataset(val_dict,   configs, training_mode)
    test_ds  = Load_Dataset(test_dict,  configs, training_mode)

    # Create DataLoaders
    train_loader = DataLoader(
        dataset=train_ds,
        batch_size=configs.batch_size,
        shuffle=True,
        drop_last=configs.drop_last,
        num_workers=0
    )
    valid_loader = DataLoader(
        dataset=val_ds,
        batch_size=configs.batch_size,
        shuffle=False,
        drop_last=configs.drop_last,
        num_workers=0
    )
    test_loader = DataLoader(
        dataset=test_ds,
        batch_size=configs.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0
    )

    return train_loader, valid_loader, test_loader

# ----------------------------------------------------------------------
# loss.py
# ----------------------------------------------------------------------
class NTXentLoss(torch.nn.Module):

    def __init__(self, device, batch_size, temperature, use_cosine_similarity):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, zis, zjs):
        representations = torch.cat([zjs, zis], dim=0)

        similarity_matrix = self.similarity_function(representations, representations)

        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)

        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        labels = torch.zeros(2 * self.batch_size).to(self.device).long()
        loss = self.criterion(logits, labels)

        return loss / (2 * self.batch_size)

# ----------------------------------------------------------------------
# hard_losses.py
# ----------------------------------------------------------------------
def inst_CL_hard(z1, z2):
    B, T = z1.size(0), z1.size(1)
    if B == 1:
        return z1.new_tensor(0.)
    z = torch.cat([z1, z2], dim=0)  # 2B x T x C
    z = z.transpose(0, 1)  # T x 2B x C
    sim = torch.matmul(z, z.transpose(1, 2))  # T x 2B x 2B
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]    # T x 2B x (2B-1)
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)
    
    i = torch.arange(B, device=z1.device)
    loss = (logits[:, i, B + i - 1].mean() + logits[:, B + i, i].mean()) / 2
    return loss

def temp_CL_hard(z1, z2):
    B, T = z1.size(0), z1.size(1)
    if T == 1:
        return z1.new_tensor(0.)
    z = torch.cat([z1, z2], dim=1)  # B x 2T x C
    sim = torch.matmul(z, z.transpose(1, 2))  # B x 2T x 2T
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]    # B x 2T x (2T-1)
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)
    
    t = torch.arange(T, device=z1.device)
    loss = (logits[:, t, T + t - 1].mean() + logits[:, T + t, t].mean()) / 2
    return loss

def hier_CL_hard(z1, z2, lambda_=0.5, temporal_unit=0):
    loss = torch.tensor(0., device=z1.device)
    d = 0
    while z1.size(1) > 1:
        if lambda_ != 0:
            loss += lambda_ * inst_CL_hard(z1, z2)
        if d >= temporal_unit:
            if 1 - lambda_ != 0:
                loss += (1 - lambda_) * temp_CL_hard(z1, z2)
        d += 1
        z1 = F.max_pool1d(z1.transpose(1, 2), kernel_size=2).transpose(1, 2)
        z2 = F.max_pool1d(z2.transpose(1, 2), kernel_size=2).transpose(1, 2)
        
    if z1.size(1) == 1:
        if lambda_ != 0:
            loss += lambda_ * inst_CL_hard(z1, z2)
        d += 1
    return loss / d

# ------------------------------------------------------------------------------------------#
# soft_losses.py
# ------------------------------------------------------------------------------------------#

# (1) Instance-wise CL
def inst_CL_soft(z1, z2, soft_labels_L, soft_labels_R):
    B, T = z1.size(0), z1.size(1)
    if B == 1:
        return z1.new_tensor(0.)
    z = torch.cat([z1, z2], dim=0)  # 2B x T x C
    z = z.transpose(0, 1)  # T x 2B x C
    sim = torch.matmul(z, z.transpose(1, 2))  # T x 2B x 2B
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]    # T x 2B x (2B-1)
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)
    i = torch.arange(B, device=z1.device)
    loss = torch.sum(logits[:,i]*soft_labels_L)
    loss += torch.sum(logits[:,B + i]*soft_labels_R)
    loss /= (2*B*T)
    return loss

# (2) Temporal CL
def temp_CL_soft(z1, z2, timelag_L, timelag_R):
    B, T = z1.size(0), z1.size(1)
    if T == 1:
        return z1.new_tensor(0.)
    z = torch.cat([z1, z2], dim=1)  # B x 2T x C
    sim = torch.matmul(z, z.transpose(1, 2))  # B x 2T x 2T
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]    # B x 2T x (2T-1)
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)
    t = torch.arange(T, device=z1.device)
    loss = torch.sum(logits[:,t]*timelag_L)
    loss += torch.sum(logits[:,T + t]*timelag_R)
    loss /= (2*B*T)
    return loss

#------------------------------------------------------------------------------------------#
# (3) Hierarchical CL = Instance CL + Temporal CL
# (The below differs by the way it generates timelag for temporal CL )
## 3-1) hier_CL_soft : sigmoid
## 3-2) hier_CL_soft_window : window
## 3-3) hier_CL_soft_thres : threshold
## 3-4) hier_CL_soft_gaussian : gaussian
## 3-5) hier_CL_soft_interval : same interval
## 3-6) hier_CL_soft_wo_inst : 3-1) w/o instance CL
#------------------------------------------------------------------------------------------#

def hier_CL_soft(z1, z2, soft_labels, tau_temp=2, lambda_=0.5, temporal_unit=0, 
                 soft_temporal=False, soft_instance=False, temporal_hierarchy=True):
    
    if soft_labels is not None:
        soft_labels = torch.tensor(soft_labels, device=z1.device)
        soft_labels_L, soft_labels_R = dup_matrix(soft_labels)
    loss = torch.tensor(0., device=z1.device)
    d = 0
    while z1.size(1) > 1:
        if lambda_ != 0:
            if soft_instance:
                loss += lambda_ * inst_CL_soft(z1, z2, soft_labels_L, soft_labels_R)
            else:
                loss += lambda_ * inst_CL_hard(z1, z2)
        if d >= temporal_unit:
            if 1 - lambda_ != 0:
                if soft_temporal:
                    if temporal_hierarchy:
                        timelag = timelag_sigmoid(z1.shape[1],tau_temp*(2**d))
                    else:
                        timelag = timelag_sigmoid(z1.shape[1],tau_temp)
                    timelag = torch.tensor(timelag, device=z1.device)
                    timelag_L, timelag_R = dup_matrix(timelag)
                    loss += (1 - lambda_) * temp_CL_soft(z1, z2, timelag_L, timelag_R)
                else:
                    loss += (1 - lambda_) * temp_CL_hard(z1, z2)
        d += 1
        z1 = F.max_pool1d(z1.transpose(1, 2), kernel_size=2).transpose(1, 2)
        z2 = F.max_pool1d(z2.transpose(1, 2), kernel_size=2).transpose(1, 2)

    if z1.size(1) == 1:
        if lambda_ != 0:
            if soft_instance:
                loss += lambda_ * inst_CL_soft(z1, z2, soft_labels_L, soft_labels_R)
            else:
                loss += lambda_ * inst_CL_hard(z1, z2)
        d += 1

    return loss / d


def hier_CL_soft_window(z1, z2, soft_labels, window_ratio, tau_temp=2, lambda_=0.5,
                        temporal_unit=0, soft_temporal=False, soft_instance=False):
    soft_labels = torch.tensor(soft_labels, device=z1.device)
    soft_labels_L, soft_labels_R = dup_matrix(soft_labels)
    loss = torch.tensor(0., device=z1.device)
    d = 0
    while z1.size(1) > 1:
        if lambda_ != 0:
            if soft_instance:
                loss += lambda_ * inst_CL_soft(z1, z2, soft_labels_L, soft_labels_R)
            else:
                loss += lambda_ * inst_CL_hard(z1, z2)
        if d >= temporal_unit:
            if 1 - lambda_ != 0:
                if soft_temporal:
                    timelag = timelag_sigmoid_window(z1.shape[1],tau_temp*(2**d),window_ratio)
                    timelag = torch.tensor(timelag, device=z1.device)
                    timelag_L, timelag_R = dup_matrix(timelag)
                    loss += (1 - lambda_) * temp_CL_soft(z1, z2, timelag_L, timelag_R)
                else:
                    loss += (1 - lambda_) * temp_CL_hard(z1, z2)
        d += 1
        z1 = F.max_pool1d(z1.transpose(1, 2), kernel_size=2).transpose(1, 2)
        z2 = F.max_pool1d(z2.transpose(1, 2), kernel_size=2).transpose(1, 2)

    if z1.size(1) == 1:
        if lambda_ != 0:
            if soft_instance:
                loss += lambda_ * inst_CL_soft(z1, z2, soft_labels_L, soft_labels_R)
            else:
                loss += lambda_ * inst_CL_hard(z1, z2)
        d += 1

    return loss / d

def hier_CL_soft_thres(z1, z2, soft_labels, threshold, lambda_=0.5, temporal_unit=0, soft_temporal=False, soft_instance=False):
    soft_labels = torch.tensor(soft_labels, device=z1.device)
    soft_labels_L, soft_labels_R = dup_matrix(soft_labels)
    loss = torch.tensor(0., device=z1.device)
    d = 0
    while z1.size(1) > 1:
        if lambda_ != 0:
            if soft_instance:
                loss += lambda_ * inst_CL_soft(z1, z2, soft_labels_L, soft_labels_R)
            else:
                loss += lambda_ * inst_CL_hard(z1, z2)
        if d >= temporal_unit:
            if 1 - lambda_ != 0:
                if soft_temporal:
                    timelag = timelag_sigmoid_threshold(z1.shape[1], threshold)
                    timelag = torch.tensor(timelag, device=z1.device)
                    timelag_L, timelag_R = dup_matrix(timelag)
                    loss += (1 - lambda_) * temp_CL_soft(z1, z2, timelag_L, timelag_R)
                else:
                    loss += (1 - lambda_) * temp_CL_hard(z1, z2)
        d += 1
        z1 = F.max_pool1d(z1.transpose(1, 2), kernel_size=2).transpose(1, 2)
        z2 = F.max_pool1d(z2.transpose(1, 2), kernel_size=2).transpose(1, 2)

    if z1.size(1) == 1:
        if lambda_ != 0:
            if soft_instance:
                loss += lambda_ * inst_CL_soft(z1, z2, soft_labels_L, soft_labels_R)
            else:
                loss += lambda_ * inst_CL_hard(z1, z2)
        d += 1

    return loss / d


def hier_CL_soft_gaussian(z1, z2, soft_labels, tau_temp=2, lambda_=0.5, temporal_unit=0, soft_temporal=False, soft_instance=False, temporal_hierarchy=True):
    soft_labels = torch.tensor(soft_labels, device=z1.device)
    soft_labels_L, soft_labels_R = dup_matrix(soft_labels)
    loss = torch.tensor(0., device=z1.device)
    d = 0
    while z1.size(1) > 1:
        if lambda_ != 0:
            if soft_instance:
                loss += lambda_ * inst_CL_soft(z1, z2, soft_labels_L, soft_labels_R)
            else:
                loss += lambda_ * inst_CL_hard(z1, z2)
        if d >= temporal_unit:
            if 1 - lambda_ != 0:
                if soft_temporal:
                    if temporal_hierarchy:
                        timelag = timelag_gaussian(z1.shape[1],tau_temp/(2**d))
                    else:
                        timelag = timelag_gaussian(z1.shape[1],tau_temp)
                    timelag = torch.tensor(timelag, device=z1.device)
                    timelag_L, timelag_R = dup_matrix(timelag)
                    loss += (1 - lambda_) * temp_CL_soft(z1, z2, timelag_L, timelag_R)
                else:
                    loss += (1 - lambda_) * temp_CL_hard(z1, z2)
        d += 1
        z1 = F.max_pool1d(z1.transpose(1, 2), kernel_size=2).transpose(1, 2)
        z2 = F.max_pool1d(z2.transpose(1, 2), kernel_size=2).transpose(1, 2)

    if z1.size(1) == 1:
        if lambda_ != 0:
            if soft_instance:
                loss += lambda_ * inst_CL_soft(z1, z2, soft_labels_L, soft_labels_R)
            else:
                loss += lambda_ * inst_CL_hard(z1, z2)
        d += 1

    return loss / d


def hier_CL_soft_interval(z1, z2, soft_labels, tau_temp=2, lambda_=0.5, temporal_unit=0, soft_temporal=False, soft_instance=False):
    soft_labels = torch.tensor(soft_labels, device=z1.device)
    soft_labels_L, soft_labels_R = dup_matrix(soft_labels)
    loss = torch.tensor(0., device=z1.device)
    d = 0
    while z1.size(1) > 1:
        if lambda_ != 0:
            if soft_instance:
                loss += lambda_ * inst_CL_soft(z1, z2, soft_labels_L, soft_labels_R)
            else:
                loss += lambda_ * inst_CL_hard(z1, z2)
        if d >= temporal_unit:
            if 1 - lambda_ != 0:
                if soft_temporal:
                    timelag = timelag_same_interval(z1.shape[1],tau_temp/(2**d))
                    timelag = torch.tensor(timelag, device=z1.device)
                    timelag_L, timelag_R = dup_matrix(timelag)
                    loss += (1 - lambda_) * temp_CL_soft(z1, z2, timelag_L, timelag_R)
                else:
                    loss += (1 - lambda_) * temp_CL_hard(z1, z2)
        d += 1
        z1 = F.max_pool1d(z1.transpose(1, 2), kernel_size=2).transpose(1, 2)
        z2 = F.max_pool1d(z2.transpose(1, 2), kernel_size=2).transpose(1, 2)

    if z1.size(1) == 1:
        if lambda_ != 0:
            if soft_instance:
                loss += lambda_ * inst_CL_soft(z1, z2, soft_labels_L, soft_labels_R)
            else:
                loss += lambda_ * inst_CL_hard(z1, z2)
        d += 1

    return loss / d

def hier_CL_soft_wo_inst(z1, z2, soft_labels, tau_temp=2, lambda_=0.5, temporal_unit=0, soft_temporal=False, soft_instance=False):
    soft_labels = torch.tensor(soft_labels, device=z1.device)
    soft_labels_L, soft_labels_R = dup_matrix(soft_labels)
    loss = torch.tensor(0., device=z1.device)
    d = 0
    while z1.size(1) > 1:
        if d >= temporal_unit:
            if 1 - lambda_ != 0:
                if soft_temporal:
                    timelag = timelag_sigmoid(z1.shape[1],tau_temp*(2**d))
                    timelag = torch.tensor(timelag, device=z1.device)
                    timelag_L, timelag_R = dup_matrix(timelag)
                    loss += (1 - lambda_) * temp_CL_soft(z1, z2, timelag_L, timelag_R)
                else:
                    loss += (1 - lambda_) * temp_CL_hard(z1, z2)
        d += 1
        z1 = F.max_pool1d(z1.transpose(1, 2), kernel_size=2).transpose(1, 2)
        z2 = F.max_pool1d(z2.transpose(1, 2), kernel_size=2).transpose(1, 2)

    if z1.size(1) == 1:
        d += 1

    return loss / d

# ----------------------------------------------------------------------
# attention.py
# ----------------------------------------------------------------------
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, float('-inf'))
            del mask

        attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads=heads, dropout=dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)))
            ]))

    def forward(self, x, mask=None):
        for attn, ff in self.layers:
            x = attn(x, mask=mask)
            x = ff(x)
        return x


class Seq_Transformer(nn.Module):
    def __init__(self, *, patch_size, dim, depth, heads, mlp_dim, channels=1, dropout=0.1):
        super().__init__()
        patch_dim = channels * patch_size
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.c_token = nn.Parameter(torch.randn(1, 1, dim))
        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout)
        self.to_c_token = nn.Identity()


    def forward(self, forward_seq):
        x = self.patch_to_embedding(forward_seq)
        b, n, _ = x.shape
        c_tokens = repeat(self.c_token, '() n d -> b n d', b=b)
        x = torch.cat((c_tokens, x), dim=1)
        x = self.transformer(x)
        c_t = self.to_c_token(x[:, 0])
        return c_t

# ----------------------------------------------------------------------
# tc.py
# ----------------------------------------------------------------------
class TC(nn.Module):
    def __init__(self, configs, device):
        super(TC, self).__init__()
        self.num_channels = configs.final_out_channels
        self.timestep = configs.TC.timesteps
        self.Wk = nn.ModuleList([nn.Linear(configs.TC.hidden_dim, self.num_channels) for i in range(self.timestep)])
        self.lsoftmax = nn.LogSoftmax()
        self.device = device
        
        self.projection_head = nn.Sequential(
            nn.Linear(configs.TC.hidden_dim, configs.final_out_channels // 2),
            nn.BatchNorm1d(configs.final_out_channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(configs.final_out_channels // 2, configs.final_out_channels // 4),
        )

        self.seq_transformer = Seq_Transformer(patch_size=self.num_channels, dim=configs.TC.hidden_dim, depth=4, heads=4, mlp_dim=64)

    def forward(self, features_aug1, features_aug2):
        show_shape("TC IN feats1/2", (features_aug1, features_aug2))
        z_aug1 = features_aug1  # features are (batch_size, #channels, seq_len)
        seq_len = z_aug1.shape[2]
        z_aug1 = z_aug1.transpose(1, 2)

        z_aug2 = features_aug2
        z_aug2 = z_aug2.transpose(1, 2)

        batch = z_aug1.shape[0]
        t_samples = torch.randint(seq_len - self.timestep, size=(1,)).long().to(self.device)  # randomly pick time stamps

        nce = 0  # average over timestep and batch
        encode_samples = torch.empty((self.timestep, batch, self.num_channels)).float().to(self.device)

        for i in np.arange(1, self.timestep + 1):
            encode_samples[i - 1] = z_aug2[:, t_samples + i, :].view(batch, self.num_channels)
        forward_seq = z_aug1[:, :t_samples + 1, :]

        c_t = self.seq_transformer(forward_seq)
        show_shape("TC context token", c_t)

        pred = torch.empty((self.timestep, batch, self.num_channels)).float().to(self.device)
        for i in np.arange(0, self.timestep):
            linear = self.Wk[i]
            pred[i] = linear(c_t)
        for i in np.arange(0, self.timestep):
            total = torch.mm(encode_samples[i], torch.transpose(pred[i], 0, 1))
            nce += torch.sum(torch.diag(self.lsoftmax(total)))
        nce /= -1. * batch * self.timestep
        return nce, self.projection_head(c_t)
    
# ----------------------------------------------------------------------
# model.py
# ----------------------------------------------------------------------
class tstcc_soft(nn.Module):
    def __init__(self, configs):
        super(tstcc_soft, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(configs.input_channels, 32, kernel_size=configs.kernel_size,
                      stride=configs.stride, bias=False, padding=(configs.kernel_size//2)),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(configs.dropout)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(64, configs.final_out_channels, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(configs.final_out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )

        model_output_dim = configs.features_len
        self.logits = nn.Linear(model_output_dim * configs.final_out_channels, configs.num_classes)
        self.lambda_ = configs.lambda_ # 0.5
        self.soft_temporal = configs.soft_temporal #True
        self.soft_instance = configs.soft_instance #True
        self.tau_temp = configs.tau_temp

    def forward(self, aug1, aug2, soft_labels, train=True):
        if train:
            if self.soft_instance:
                soft_labels_L, soft_labels_R = dup_matrix(soft_labels)
                del soft_labels
            
            temporal_loss = torch.tensor(0., device=aug1.device)
            instance_loss = torch.tensor(0., device=aug1.device)
            
            #-------------------------------------------------#
            # DEPTH = 1
            #-------------------------------------------------#
            aug1 = self.conv_block1(aug1)
            aug2 = self.conv_block1(aug2)
            z1, z2 = _btc(aug1), _btc(aug2) # transpose
            T      = z1.size(1)
            if self.soft_temporal:
                d=0
                timelag = timelag_sigmoid(T, self.tau_temp*(2**d))
                timelag = torch.tensor(timelag, device=z1.device)
                timelag_L, timelag_R = dup_matrix(timelag)
                temporal_loss += (1-self.lambda_) * temp_CL_soft(z1, z2, timelag_L, timelag_R)
            else:
                temporal_loss += (1-self.lambda_) * temp_CL_hard(z1, z2)
            if self.soft_instance:
                instance_loss += self.lambda_ * inst_CL_soft(z1, z2, soft_labels_L, soft_labels_R)
            else:
                instance_loss += self.lambda_ * inst_CL_hard(z1, z2)

            #-------------------------------------------------#
            # DEPTH = 2
            #-------------------------------------------------#
            aug1 = self.conv_block2(aug1)
            aug2 = self.conv_block2(aug2)
            z1, z2 = _btc(aug1), _btc(aug2) # transpose
            T      = z1.size(1)

            if self.soft_temporal:
                d=1
                timelag = timelag_sigmoid(T,self.tau_temp*(2**d))
                timelag = torch.tensor(timelag, device=z1.device)
                timelag_L, timelag_R = dup_matrix(timelag)
                temporal_loss += (1-self.lambda_) * temp_CL_soft(z1, z2, timelag_L, timelag_R)
            else:
                temporal_loss += (1-self.lambda_) * temp_CL_hard(z1, z2)
                
            if self.soft_instance:
                instance_loss += self.lambda_ * inst_CL_soft(z1, z2, soft_labels_L, soft_labels_R)
            else:
                instance_loss += self.lambda_ * inst_CL_hard(z1, z2)
            
            #-------------------------------------------------#
            # DEPTH = 3
            #-------------------------------------------------#
            aug1 = self.conv_block3(aug1)
            aug2 = self.conv_block3(aug2)
            z1, z2 = _btc(aug1), _btc(aug2) # transpose
            T      = z1.size(1)
        
            if self.soft_temporal:
                d=2
                timelag = timelag_sigmoid(T,self.tau_temp*(2**d))
                timelag = torch.tensor(timelag, device=z1.device)
                timelag_L, timelag_R = dup_matrix(timelag)
                temporal_loss += (1-self.lambda_) * temp_CL_soft(z1, z2, timelag_L, timelag_R)
            else:
                temporal_loss += (1-self.lambda_) * temp_CL_hard(z1, z2)
           
            if self.soft_instance:
                instance_loss += self.lambda_ * inst_CL_soft(z1, z2, soft_labels_L, soft_labels_R)
                del soft_labels_L, soft_labels_R
            else:
                instance_loss += self.lambda_ * inst_CL_hard(z1, z2)
        
        else:
            aug = self.conv_block1(aug1)
            aug = self.conv_block2(aug)
            aug = self.conv_block3(aug)
        
        ############################################################################
        ############################################################################
        
        if train:
            aug1_flat = aug1.reshape(aug1.shape[0], -1)
            aug2_flat = aug2.reshape(aug2.shape[0], -1)
            aug1_logits = self.logits(aug1_flat)
            aug2_logits = self.logits(aug2_flat)
            final_loss = temporal_loss + instance_loss
            return aug1_logits, aug2_logits, aug1, aug2, final_loss

        else:
            aug_flat = aug.reshape(aug.shape[0], -1)
            aug_logits = self.logits(aug_flat)
            return aug_logits, aug

# ----------------------------------------------------------------------
# timelags.py
# ----------------------------------------------------------------------    
def dup_matrix(mat):
    mat0 = torch.tril(mat, diagonal=-1)[:, :-1]   
    mat0 += torch.triu(mat, diagonal=1)[:, 1:]
    mat1 = torch.cat([mat0,mat],dim=1)
    mat2 = torch.cat([mat,mat0],dim=1)
    return mat1, mat2

##############################################################################
## 6 Different ways of generating time lags
##############################################################################
def timelag_sigmoid(T,sigma=1):
    dist = np.arange(T)
    dist = np.abs(dist - dist[:, np.newaxis])
    matrix = 2 / (1 +np.exp(dist*sigma))
    matrix = np.where(matrix < 1e-6, 0, matrix)  # set very small values to 0         
    return matrix

def timelag_gaussian(T,sigma):
    dist = np.arange(T)
    dist = np.abs(dist - dist[:, np.newaxis])
    matrix = np.exp(-(dist**2)/(2 * sigma ** 2))
    matrix = np.where(matrix < 1e-6, 0, matrix) 
    return matrix

def timelag_same_interval(T):
    d = np.arange(T)
    X, Y = np.meshgrid(d, d)
    matrix = 1 - np.abs(X - Y) / T
    return matrix

def timelag_sigmoid_window(T, sigma=1, window_ratio=1.0):
    dist = np.arange(T)
    dist = np.abs(dist - dist[:, np.newaxis])
    matrix = 2 / (1 +np.exp(dist*sigma))
    matrix = np.where(matrix < 1e-6, 0, matrix)          
    dist_from_diag = np.abs(np.subtract.outer(np.arange(dist.shape[0]), np.arange(dist.shape[1])))
    matrix[dist_from_diag > T*window_ratio] = 0
    return matrix

def timelag_sigmoid_threshold(T, threshold=1.0):
    dist = np.ones((T,T))
    dist_from_diag = np.abs(np.subtract.outer(np.arange(dist.shape[0]), np.arange(dist.shape[1])))
    dist[dist_from_diag > T*threshold] = 0
    return dist

# ----------------------------------------------------------------------
# utils.py
# ----------------------------------------------------------------------
def _btc(x):
    # (B, C, L)  ->  (B, L, C)
    return x.transpose(1, 2).contiguous()

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

def set_requires_grad(model, dict_, requires_grad=True):
    for param in model.named_parameters():
        if param[0] in dict_:
            param[1].requires_grad = requires_grad

def loop_iterable(iterable):
    while True:
        yield from iterable
        
def fix_randomness(SEED):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)
        # if name=='weight':
        #     nn.init.kaiming_uniform_(param.data)
        # else:
        #     torch.nn.init.zeros_(param.data)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def _calc_metrics(pred_labels, true_labels, log_dir, home_path):
    pred_labels = np.array(pred_labels).astype(int)
    true_labels = np.array(true_labels).astype(int)

    # save targets
    labels_save_path = os.path.join(log_dir, "labels")
    os.makedirs(labels_save_path, exist_ok=True)
    np.save(os.path.join(labels_save_path, "predicted_labels.npy"), pred_labels)
    np.save(os.path.join(labels_save_path, "true_labels.npy"), true_labels)

    r = classification_report(true_labels, pred_labels, digits=6, output_dict=True)
    cm = confusion_matrix(true_labels, pred_labels)
    df = pd.DataFrame(r)
    df["cohen"] = cohen_kappa_score(true_labels, pred_labels)
    df["accuracy"] = accuracy_score(true_labels, pred_labels)
    df = df * 100

    # save classification report
    exp_name = os.path.split(os.path.dirname(log_dir))[-1]
    training_mode = os.path.basename(log_dir)
    file_name = f"{exp_name}_{training_mode}_classification_report.xlsx"
    report_Save_path = os.path.join(home_path, log_dir, file_name)
    df.to_excel(report_Save_path)

    # save confusion matrix
    cm_file_name = f"{exp_name}_{training_mode}_confusion_matrix.torch"
    cm_Save_path = os.path.join(home_path, log_dir, cm_file_name)
    torch.save(cm, cm_Save_path)


def _logger(logger_name, level=logging.DEBUG):
    """
    Method to return a custom logger with the given name and level
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    # format_string = ("%(asctime)s — %(name)s — %(levelname)s — %(funcName)s:"
    #                 "%(lineno)d — %(message)s")
    format_string = "%(message)s"
    log_format = logging.Formatter(format_string)
    # Creating and adding the console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)
    # Creating and adding the file handler
    file_handler = logging.FileHandler(logger_name, mode='a')
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    return logger

def copy_Files(destination, data_type):
    destination_dir = os.path.join(destination, "model_files")
    os.makedirs(destination_dir, exist_ok=True)
    copy("main.py", os.path.join(destination_dir, "main.py"))
    copy("trainer/trainer.py", os.path.join(destination_dir, "trainer.py"))
    copy(f"config_files/{data_type}_Configs.py", os.path.join(destination_dir, f"{data_type}_Configs.py"))
    copy("dataloader/augmentations.py", os.path.join(destination_dir, "augmentations.py"))
    copy("dataloader/dataloader.py", os.path.join(destination_dir, "dataloader.py"))
    copy(f"models/model.py", os.path.join(destination_dir, f"model.py"))
    copy("models/loss.py", os.path.join(destination_dir, "loss.py"))
    copy("models/TC.py", os.path.join(destination_dir, "TC.py"))

# ----------------------------------------------------------------------
# trainer_utils.py
# ----------------------------------------------------------------------
def densify(x, tau, alpha=0.5):
    return ((2*alpha) / (1 + np.exp(-tau*x))) + (1-alpha)*np.eye(x.shape[0])

def compute_ssl_loss(feat1, feat2, tc_model, nt_xent_loss):
    feat1 = F.normalize(feat1, dim=1)
    feat2 = F.normalize(feat2, dim=1)
    loss1, proj1 = tc_model(feat1, feat2)
    loss2, proj2 = tc_model(feat2, feat1)
    proj1 = F.normalize(proj1, dim=1)
    proj2 = F.normalize(proj2, dim=1)
    ssl_loss = loss1 + loss2 + 0.7 * nt_xent_loss(proj1, proj2)
    return ssl_loss

def model_train(soft_labels, model, temporal_contr_model,
                model_opt, tc_opt,
                criterion, train_loader,
                config, device, training_mode, lambda_aux):

    model.train()
    temporal_contr_model.train()
    soft_labels = torch.tensor(soft_labels, device=device)

    nt_xent = NTXentLoss(device,
                         config.batch_size,
                         config.Context_Cont.temperature,
                         config.Context_Cont.use_cosine_similarity)

    batch_losses = []

    for batch_idx, (data, labels, aug1, aug2) in enumerate(train_loader):
        if batch_idx == 0:
            show_shape("train-loop INPUT   aug1/aug2", (aug1, aug2))

        # move to device 
        aug1, aug2 = aug1.float().to(device), aug2.float().to(device)
        data, labels = data.float().to(device), labels.long().to(device)
        aug1 = aug1*1 # careful with this
        aug2 = aug2*1

        model_opt.zero_grad()
        tc_opt.zero_grad()

        # SSL vs supervised branch
        if training_mode == "self_supervised":
            
            soft_labels_batch = soft_labels[batch_idx][:,batch_idx]
            
            _, _, feat1, feat2, final_loss = model(aug1, aug2, soft_labels_batch)
            del soft_labels_batch
            
            loss = compute_ssl_loss(feat1, feat2, temporal_contr_model, nt_xent)
            
            loss += lambda_aux * final_loss
        else:                                 
            preds, _ = model(data)
            loss = criterion(preds, labels)

        # backward & step
        loss.backward()
        model_opt.step()
        tc_opt.step()

        batch_losses.append(loss.item())

    return torch.tensor(batch_losses).mean(), torch.nan              

def model_train_wo_DTW(dist_func, dist_type, tau_inst, model, 
                       temporal_contr_model, model_opt, tc_opt,
                       criterion, train_loader, config, device, 
                       training_mode, lambda_aux):
    model.train()
    temporal_contr_model.train()

    nt_xent = NTXentLoss(device,
                         config.batch_size,
                         config.Context_Cont.temperature,
                         config.Context_Cont.use_cosine_similarity)

    batch_losses = []

    #for data, labels, aug1, aug2 in train_loader:
    for batch_idx, (data, labels, aug1, aug2) in enumerate(train_loader):
        if batch_idx == 0:
            show_shape("train-loop INPUT   aug1/aug2", (aug1, aug2))

        # move to device
        aug1, aug2 = aug1.float().to(device), aug2.float().to(device)
        data, labels = data.float().to(device), labels.long().to(device)
        aug1 = aug1*100
        aug2 = aug2*100

        model_opt.zero_grad()
        tc_opt.zero_grad()

        # SSL vs supervised branch
        if training_mode == "self_supervised":
            temp = data.view(data.shape[0], -1).detach().cpu().numpy()
            dist_mat_batch = dist_func(temp)
            if dist_type=='euc':
                dist_mat_batch = (dist_mat_batch - np.min(dist_mat_batch)) / (np.max(dist_mat_batch) - np.min(dist_mat_batch))
                dist_mat_batch = - dist_mat_batch
            dist_mat_batch = densify(dist_mat_batch, tau_inst, alpha=0.5)
            dist_mat_batch = torch.tensor(dist_mat_batch, device=device)
            
            _, _, feat1, feat2, final_loss = model(aug1, aug2, dist_mat_batch)
            del dist_mat_batch
            
            loss = compute_ssl_loss(feat1, feat2, temporal_contr_model, nt_xent)
            
            loss += lambda_aux * final_loss
        else:                                 
            preds, _ = model(data)
            loss = criterion(preds, labels)

        # backward & step
        loss.backward()
        model_opt.step()
        tc_opt.step()

        batch_losses.append(loss.item())

    return torch.tensor(batch_losses).mean(), torch.nan

def model_evaluate(model,
                   temporal_contr_model,
                   dl,
                   device,
                   training_mode):

    if training_mode == "self_supervised":
        return 0.0, 0.0, [], []

    model.eval()
    temporal_contr_model.eval()

    criterion = nn.CrossEntropyLoss()
    total_loss = []
    total_acc = []
    outs = np.array([])
    trgs = np.array([])

    with torch.no_grad():
        for data, labels, _, _ in dl:
            data, labels = data.float().to(device), labels.long().to(device)
            predictions, _ = model(data, 0, 0, train=False)
            loss = criterion(predictions, labels)
            total_loss.append(loss.item())
            total_acc.append(labels.eq(predictions.argmax(dim=1)).float().mean())
            outs = np.append(outs, predictions.argmax(dim=1).cpu().numpy())
            trgs = np.append(trgs, labels.cpu().numpy())

    return torch.tensor(total_loss).mean(), torch.tensor(total_acc).mean(), outs, trgs

def gen_pseudo_labels(model, dataloader, device, experiment_log_dir, pc):
    model.eval()
    softmax = nn.Softmax(dim=1)

    all_pseudo_labels = np.array([])
    all_labels = np.array([])
    all_data = []

    with torch.no_grad():
        for _, data, labels, _, _ in dataloader:
            data = data.float().to(device)
            labels = labels.view((-1)).long().to(device)

            output = model(data, 0, 0, train=False)
            predictions, features = output

            normalized_preds = softmax(predictions)
            pseudo_labels = normalized_preds.max(1, keepdim=True)[1].squeeze()
            all_pseudo_labels = np.append(all_pseudo_labels, pseudo_labels.cpu().numpy())

            all_labels = np.append(all_labels, labels.cpu().numpy())
            all_data.append(data)

    all_data = torch.cat(all_data, dim=0)

    data_save = dict()
    data_save["samples"] = all_data
    data_save["labels"] = torch.LongTensor(torch.from_numpy(all_pseudo_labels).long())
    file_name = f"pseudo_train_data_{str(pc)}perc.pt"
    torch.save(data_save, os.path.join(experiment_log_dir, file_name))
    print("Pseudo labels generated ...")
    
# ----------------------------------------------------------------------
# trainer.py
# ----------------------------------------------------------------------
def Trainer(DTW, model, temporal_contr_model, model_opt, temp_cont_optimizer, train_dl, valid_dl, test_dl, device,
            config, experiment_log_dir, training_mode):
    print("Training started ....")
    lambda_aux = config.lambda_aux

    # (1) Loss Function & LR Scheduler & Epochs
    criterion = nn.CrossEntropyLoss()

    # (2) Train
    for epoch in range(1, config.num_epoch + 1):
        train_loss, train_acc = model_train(DTW, model, temporal_contr_model, model_opt, temp_cont_optimizer,
                                            criterion, train_dl, config, device, training_mode, lambda_aux)
        val_loss, valid_acc, _, _ = model_evaluate(model, temporal_contr_model, valid_dl, device, training_mode)
        
        if (training_mode == "self_supervised"):
            print(f"Epoch {epoch:02d} | ssl_train_loss: {train_loss:.4f} | ssl_val_loss: {val_loss:.4f}")
            mlflow.log_metrics({
                "ssl_train_loss": train_loss,
                "ssl_val_loss": val_loss
            }, step=epoch)

    os.makedirs(os.path.join(experiment_log_dir, "saved_models"), exist_ok=True)
    chkpoint = {'model_state_dict': model.state_dict(), 'temporal_contr_model_state_dict': temporal_contr_model.state_dict()}
    torch.save(chkpoint, os.path.join(experiment_log_dir, "saved_models", f'ckp_last.pt'))
    print("\n################## Training is Done! #########################")
    
def Trainer_wo_DTW(model, temporal_contr_model, model_optimizer, temp_cont_optimizer, train_dl, valid_dl, test_dl, device,
            config, experiment_log_dir, training_mode):
    print("Training started ....")
    dist = config.dist_type
    tau_inst = config.tau_inst
    lambda_aux = config.lambda_aux

    # (1) Loss Function & LR Scheduler & Epochs
    criterion = nn.CrossEntropyLoss()
    
    ########################################################
    if dist == 'cos':
        dist_func = cosine_similarity
    elif dist == 'euc':
        dist_func = euclidean_distances
    ########################################################
    
    # (2) Train
    for epoch in range(1, config.num_epoch + 1):
        train_loss, train_acc = model_train_wo_DTW(dist_func, dist, tau_inst, model, temporal_contr_model, model_optimizer, temp_cont_optimizer,
                                            criterion, train_dl, config, device, training_mode, lambda_aux)
        
        val_loss, valid_acc, _, _ = model_evaluate(model, temporal_contr_model, valid_dl, device, training_mode)
        
        if (training_mode == "self_supervised"):
            print(f"Epoch {epoch:02d} | ssl_train_loss: {train_loss:.4f} | ssl_val_loss: {val_loss:.4f}")
            mlflow.log_metrics({
                "ssl_train_loss": train_loss,
                "ssl_val_loss": val_loss
            }, step=epoch)
    
    os.makedirs(os.path.join(experiment_log_dir, "saved_models"), exist_ok=True)
    chkpoint = {'model_state_dict': model.state_dict(), 'temporal_contr_model_state_dict': temporal_contr_model.state_dict()}
    torch.save(chkpoint, os.path.join(experiment_log_dir, "saved_models", f'ckp_last.pt'))
    print("\n################## Training is Done! #########################")
    
def Trainer_wo_val(DTW, model, temporal_contr_model, model_optimizer, temp_cont_optimizer, train_dl, test_dl, device,
            config, experiment_log_dir, training_mode):
    print("Training started ....")
    lambda_aux = config.lambda_aux

    # (1) Loss Function & LR Scheduler & Epochs
    criterion = nn.CrossEntropyLoss()
    
    # (2) Train
    for epoch in range(1, config.num_epoch + 1):
        train_loss, train_acc = model_train(DTW, model, temporal_contr_model, model_optimizer, temp_cont_optimizer,
                                            criterion, train_dl, config, device, training_mode, lambda_aux)
                
        if (training_mode == "self_supervised"):
            print(f"Epoch {epoch:02d} | ssl_train_loss: {train_loss:.4f}")
            mlflow.log_metrics({
                "ssl_train_loss": train_loss,
            }, step=epoch)
    
    os.makedirs(os.path.join(experiment_log_dir, "saved_models"), exist_ok=True)
    chkpoint = {'model_state_dict': model.state_dict(), 'temporal_contr_model_state_dict': temporal_contr_model.state_dict()}
    torch.save(chkpoint, os.path.join(experiment_log_dir, "saved_models", f'ckp_last.pt'))
    print("\n################## Training is Done! #########################")

def Trainer_wo_DTW_wo_val(model, temporal_contr_model, model_optimizer, temp_cont_optimizer, train_dl, valid_dl, test_dl, device,
            config, experiment_log_dir, training_mode):
    print("Training started ....")
    dist = config.dist_type
    tau_inst = config.tau_inst
    lambda_aux = config.lambda_aux

    # (1) Loss Function & LR Scheduler & Epochs
    criterion = nn.CrossEntropyLoss()
    
    ########################################################
    if dist == 'cos':
        dist_func = cosine_similarity
    elif dist == 'euc':
        dist_func = euclidean_distances
    ########################################################
    
    # (2) Train
    for epoch in range(1, config.num_epoch + 1):
        train_loss, train_acc = model_train_wo_DTW(dist_func, dist, tau_inst, model, temporal_contr_model, model_optimizer, temp_cont_optimizer,
                                            criterion, train_dl, config, device, training_mode, lambda_aux)
        
        if (training_mode == "self_supervised"):
            print(f"Epoch {epoch:02d} | ssl_train_loss: {train_loss:.4f}")
            mlflow.log_metrics({
                "ssl_train_loss": train_loss,
            }, step=epoch)
    
    os.makedirs(os.path.join(experiment_log_dir, "saved_models"), exist_ok=True)
    chkpoint = {'model_state_dict': model.state_dict(), 'temporal_contr_model_state_dict': temporal_contr_model.state_dict()}
    torch.save(chkpoint, os.path.join(experiment_log_dir, "saved_models", f'ckp_last.pt'))
    print("\n################## Training is Done! #########################")

# ----------------------------------------------------------------------
# config.py
# ----------------------------------------------------------------------
class Config(object):
    """
    Hyper-parameters tuned for windowed ECG segments
    (shape = [N, 10 000, 1])
    """
    def __init__(self):

        # ─────────────────── CNN encoder ────────────────────
        self.input_channels      = 1         # ECG is univariate
        self.kernel_size         = 32        # same as pFD → good for long series
        self.stride              = 4
        self.final_out_channels  = 128       # feature maps after last conv
        self.dropout             = 0.35

        # Length of the sequence that reaches the projection head
        # 10 000 → 315 after the conv + pool stack
        self.features_len        = 315
        self.num_classes         = 2

        # ─────────────────── Training ───────────────────────
        self.num_epoch           = 40

        # Optimiser
        self.beta1, self.beta2   = 0.9, 0.99
        self.lr                  = 3e-4

        # Data loader
        self.batch_size          = 64        # 10000-sample windows use more mem
        self.drop_last           = True      # match original repo
        
        # soft parameters
        self.lambda_        = 0.5 # Balance instance vs temporal contrast in soft loss
        self.lambda_aux    = 0.5  # Weight for soft loss contribution wrt the original TS-TCC loss
        self.soft_instance  = True
        self.soft_temporal  = True
        self.tau_temp       = 2.5
        self.tau_inst       = 10.0
        self.dist_type      = "euc"

        # ─────────────────── SSL blocks ─────────────────────
        # Contextual contrastive loss
        self.Context_Cont        = Context_Cont_configs()
        # Temporal contrasting
        self.TC                  = TCConfig()      # hidden=100, timesteps=50
        # Data augmentations
        self.augmentation        = augmentations()

class augmentations(object):
    """
    ECG is quite sensitive to amplitude changes; we therefore keep
    small jitter noise and allow permutation across up to 8 segments.
    """
    def __init__(self):
        self.jitter_scale_ratio = 0.001   # very mild scaling
        self.jitter_ratio       = 0.001    # mild additive jitter
        self.max_seg            = 8       # split into ≤8 segments for strong aug


class Context_Cont_configs(object):
    def __init__(self):
        self.temperature           = 0.2
        self.use_cosine_similarity = True

class TCConfig(object):
    def __init__(self):
        self.hidden_dim = 100      # same as original paper
        self.timesteps  = 50       # 15 % – 30 % of seq_len (here 45 – 95).

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

# ----------------------------------------------------------------------
# Encoding function to textract TS-TCC representations
# ----------------------------------------------------------------------
def encode_representations(X, y, model, temporal_contr_model,
                           batch_size, device):
    show_shape("encode_repr - raw X", X)

    loader = DataLoader(
        TensorDataset(torch.from_numpy(X).float(),
                      torch.from_numpy(y).long()),
        batch_size=batch_size,
        shuffle=False,
    )

    reprs, labs = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            # make sure channel dim is second
            if xb.shape.index(min(xb.shape)) != 1:
                xb = xb.permute(0, 2, 1)

            # forward through the encoder only
            # pass xb twice + a dummy soft‑label (None)
            _, feats = model(xb, xb, None, train=False)

            feats = F.normalize(feats, dim=1)

            # projection via TC head
            _, c_proj = temporal_contr_model(feats, feats)

            reprs.append(c_proj.cpu().numpy())
            labs.append(yb.numpy())

    return np.concatenate(reprs, axis=0), np.concatenate(labs, axis=0)
