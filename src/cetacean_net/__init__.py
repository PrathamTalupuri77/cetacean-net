"""CetaceanNet: multimodal attention-based classification of cetacean acoustics.

Public API re-exports the most commonly used components so callers can simply::

    from cetacean_net import AudioPreprocessor, CetaceanNet, train_model
"""

from .datasets import (
    CetaceanDataset,
    MELDataset,
    MFCCDataset,
    WSTDataset,
    build_dataloaders,
    stratified_split,
)
from .models import (
    CetaceanNet,
    CetaceanNetLight,
    MobileNetV2,
    MobileNetV2Combined,
)
from .preprocessing import AudioPreprocessor
from .training import train_model
from .visualization import visualize_samples

__all__ = [
    "AudioPreprocessor",
    "CetaceanDataset",
    "MFCCDataset",
    "WSTDataset",
    "MELDataset",
    "stratified_split",
    "build_dataloaders",
    "CetaceanNet",
    "CetaceanNetLight",
    "MobileNetV2",
    "MobileNetV2Combined",
    "train_model",
    "visualize_samples",
]

__version__ = "0.1.0"
