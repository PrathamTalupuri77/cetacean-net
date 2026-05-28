"""Training loop shared by every model variant.

The loop transparently supports both single-input models (which receive one
feature tensor) and the multimodal fusion models (which receive the
``(mfcc, wst, mel)`` triple), inferring the call signature from the batch shape.
Metrics are optionally logged to Weights & Biases.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader


def _run_batch(model: nn.Module, batch, device: torch.device):
    """Move a batch to ``device``, run the model, and return ``(outputs, labels)``."""
    if isinstance(batch, (list, tuple)) and len(batch) == 4:
        x1, x2, x3, labels = (t.to(device) for t in batch)
        outputs = model(x1.float(), x2.float(), x3.float())
    else:
        inputs, labels = (t.to(device) for t in batch)
        outputs = model(inputs.float())
    return outputs, labels


def _evaluate(model, loader, criterion, device):
    """Return ``(avg_loss, accuracy)`` over ``loader`` without gradient tracking."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for batch in loader:
            outputs, labels = _run_batch(model, batch, device)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)
            correct += (torch.max(outputs, 1)[1] == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    y,
    label2id: dict,
    num_classes: int,
    project_name: str = "cetacean-net",
    architecture: str = "Default",
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    optimizer_name: str = "AdamW",
    weight_decay: float = 1e-2,
    use_wandb: bool = True,
):
    """Train ``model`` and report per-epoch train/validation loss and accuracy.

    Class imbalance is handled with balanced class weights in the loss. Set
    ``use_wandb=False`` to disable Weights & Biases logging (e.g. for local runs).
    """
    run = None
    if use_wandb:
        import wandb

        run = wandb.init(
            project=project_name,
            config={
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "optimizer": optimizer_name,
                "architecture": architecture,
            },
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # Balanced class weights to counteract dataset imbalance.
    class_np = np.array(range(num_classes))
    labels = torch.tensor([label2id[label] for label in y])
    class_weights = compute_class_weight("balanced", classes=class_np, y=labels.cpu().numpy())
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    if optimizer_name.lower() == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        train_loss, correct_train, total_train = 0.0, 0, 0

        for batch in train_loader:
            outputs, batch_labels = _run_batch(model, batch, device)

            optimizer.zero_grad()
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_labels.size(0)
            correct_train += (torch.max(outputs, 1)[1] == batch_labels).sum().item()
            total_train += batch_labels.size(0)

        avg_train_loss = train_loss / total_train
        train_accuracy = correct_train / total_train
        avg_val_loss, val_accuracy = _evaluate(model, val_loader, criterion, device)

        if run is not None:
            run.log(
                {
                    "epoch": epoch + 1,
                    "train_loss": avg_train_loss,
                    "train_accuracy": train_accuracy,
                    "val_loss": avg_val_loss,
                    "val_accuracy": val_accuracy,
                }
            )

        print(
            f"Epoch {epoch + 1} | Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.4f}"
        )

    if run is not None:
        run.finish()

    return model
