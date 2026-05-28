"""PyTorch ``Dataset`` wrappers and train/validation splitting helpers.

Four dataset variants are provided: one per single feature (MFCC, WST, MEL) for
training the unimodal baselines, and :class:`CetaceanDataset` which yields all
three features together for the multimodal fusion models.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


class CetaceanDataset(Dataset):
    """Yields ``(mfcc, wst, mel, label)`` tuples for multimodal models."""

    def __init__(self, mfcc, wst, mel, labels):
        self.mfcc = mfcc
        self.wst = wst
        self.mel = mel
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx):
        return self.mfcc[idx], self.wst[idx], self.mel[idx], self.labels[idx]


class _SingleFeatureDataset(Dataset):
    """Base class for single-feature datasets yielding ``(feature, label)``."""

    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class MFCCDataset(_SingleFeatureDataset):
    """Single-feature dataset over MFCC tensors."""


class WSTDataset(_SingleFeatureDataset):
    """Single-feature dataset over WST tensors."""


class MELDataset(_SingleFeatureDataset):
    """Single-feature dataset over MEL tensors."""


def stratified_split(labels: torch.Tensor, test_size: float = 0.2, seed: int = 42):
    """Return ``(train_idx, val_idx)`` with class-stratified sampling."""
    indices = np.random.permutation(len(labels))
    return train_test_split(
        indices,
        test_size=test_size,
        stratify=labels.numpy(),
        random_state=seed,
    )


def build_dataloaders(
    mfcc_features,
    wst_features,
    mel_features,
    labels,
    train_idx,
    val_idx,
    batch_size: int = 32,
):
    """Build train/val ``DataLoader``s for every model variant.

    Returns a dict keyed by ``"mfcc"``, ``"wst"``, ``"mel"`` and ``"fusion"``;
    each value is a ``(train_loader, val_loader)`` tuple.
    """

    def loaders(dataset_cls, *feature_args):
        train_ds = dataset_cls(*(f[train_idx] for f in feature_args), labels[train_idx])
        val_ds = dataset_cls(*(f[val_idx] for f in feature_args), labels[val_idx])
        return (
            DataLoader(train_ds, batch_size=batch_size, shuffle=True),
            DataLoader(val_ds, batch_size=batch_size, shuffle=False),
        )

    return {
        "mfcc": loaders(MFCCDataset, mfcc_features),
        "wst": loaders(WSTDataset, wst_features),
        "mel": loaders(MELDataset, mel_features),
        "fusion": loaders(CetaceanDataset, mfcc_features, wst_features, mel_features),
    }
