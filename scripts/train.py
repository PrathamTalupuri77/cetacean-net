"""Command-line entry point for training a CetaceanNet model variant.

Example
-------
    python scripts/train.py --data-path ./data --model cetacean_net --epochs 60

The dataset directory must be laid out as ``<data-path>/<class-name>/<clip>.wav``.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from cetacean_net import (
    AudioPreprocessor,
    CetaceanNet,
    CetaceanNetLight,
    MobileNetV2,
    MobileNetV2Combined,
    build_dataloaders,
    stratified_split,
    train_model,
)

# Which loader each model consumes, and how to construct it.
SINGLE_FEATURE_MODELS = {
    "mobilenet_mel": ("mel", MobileNetV2),
    "mobilenet_mfcc": ("mfcc", MobileNetV2),
    "mobilenet_wst": ("wst", MobileNetV2),
}
FUSION_MODELS = {
    "cetacean_net": CetaceanNet,
    "cetacean_net_light": CetaceanNetLight,
    "mobilenet_combined": MobileNetV2Combined,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a CetaceanNet model variant.")
    parser.add_argument("--data-path", required=True, help="Root folder of <label>/<clip>.wav files.")
    parser.add_argument(
        "--model",
        default="cetacean_net",
        choices=list(FUSION_MODELS) + list(SINGLE_FEATURE_MODELS),
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-wandb", action="store_true", help="Disable Weights & Biases logging.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 1. Preprocess audio and extract all three feature representations.
    preprocessor = AudioPreprocessor()
    X, y = preprocessor.basic_preprocess(args.data_path)
    mel = preprocessor.compute_mel(X)
    wst = preprocessor.compute_wst(X)
    mfcc = preprocessor.compute_mfcc(X)

    # 2. Build the label mapping.
    classes = np.unique(y)
    num_classes = len(classes)
    id2label = {i: c.item() for i, c in enumerate(classes)}
    label2id = {v: k for k, v in id2label.items()}
    labels = torch.tensor([label2id[label] for label in y])

    # 3. Stratified split and dataloaders.
    train_idx, val_idx = stratified_split(labels, seed=args.seed)
    loaders = build_dataloaders(
        mfcc, wst, mel, labels, train_idx, val_idx, batch_size=args.batch_size
    )

    # 4. Pick the model and the loader it consumes.
    if args.model in FUSION_MODELS:
        model = FUSION_MODELS[args.model](num_classes=num_classes)
        train_loader, val_loader = loaders["fusion"]
    else:
        feature_key, model_cls = SINGLE_FEATURE_MODELS[args.model]
        model = model_cls(num_classes=num_classes)
        train_loader, val_loader = loaders[feature_key]

    # 5. Train.
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        y=y,
        label2id=label2id,
        num_classes=num_classes,
        architecture=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        use_wandb=not args.no_wandb,
    )


if __name__ == "__main__":
    main()
