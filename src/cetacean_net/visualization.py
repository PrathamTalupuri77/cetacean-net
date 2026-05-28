"""Quick-look plotting helpers for inspecting extracted features."""

from __future__ import annotations

import random

import matplotlib.pyplot as plt
import torch


def visualize_samples(
    feature_tensor: torch.Tensor,
    feature_name: str = "Feature",
    cmap: str = "gray",
    num_samples: int = 3,
) -> None:
    """Plot a few random feature maps from a ``[B, 1, H, W]`` tensor.

    Useful as a sanity check that MEL / WST / MFCC extraction produced sensible
    time-frequency images before training.
    """
    indices = random.sample(range(feature_tensor.shape[0]), num_samples)
    _fig, axes = plt.subplots(1, num_samples, figsize=(4 * num_samples, 4))

    for i, idx in enumerate(indices):
        data = feature_tensor[idx][0].cpu().numpy()
        axes[i].imshow(data, cmap=cmap, aspect="auto")
        axes[i].set_title(f"{feature_name} - Random Sample {i + 1}")
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()
