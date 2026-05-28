# CetaceanNet

**Multi-model, attention-based classification of cetacean species from acoustic signals.**

CetaceanNet identifies whale and dolphin species from short underwater audio
clips. Each clip is turned into three complementary time–frequency
representations — a **MEL spectrogram**, a **Wavelet Scattering Transform (WST)**,
and **MFCCs** — which are then fused by a custom attention network. Several
MobileNetV2 baselines are included for comparison.

---

## Why three features?

Bio-acoustic events (clicks, whistles, burst-pulses) vary widely in their
time–frequency structure, and no single representation captures all of them:

| Feature | Captures | Role in the model |
|---------|----------|-------------------|
| **MEL spectrogram** | Perceptual magnitude envelope over time | Primary stream / attention *query* |
| **WST** | Translation-invariant, deformation-stable structure of transients | Context stream |
| **MFCC** | Compact spectral-envelope descriptor | Context stream |

The flagship model, **`CetaceanNet`**, lets the MEL stream *cross-attend* to the
WST and MFCC streams, then combines the result with a learned modal-fusion gate.

---

## Architecture overview

```
        MEL ─► CNN ─► self-attention ─┐
                                      ├─► cross-attention (MEL ← MFCC) ─┐
       MFCC ─► CNN ─► self-attention ─┘                                ├─► modal-fusion gate ─► MLP ─► logits
                                      ┌─► cross-attention (MEL ← WST)  ─┘
        WST ─► CNN ─► self-attention ─┘
```

**Models provided** (`cetacean_net.models`):

| Model | Description |
|-------|-------------|
| `CetaceanNet` | Multimodal attention CNN (standard convolutions) |
| `CetaceanNetLight` | Same design with depthwise-separable convolutions — far fewer parameters |
| `MobileNetV2Combined` | MobileNetV2 with MFCC/WST/MEL stacked as 3 input channels (ImageNet-pretrained) |
| `MobileNetV2` | MobileNetV2 single-feature baseline (1-channel input) |

---

## Project layout

```
cetacean-net/
├── src/cetacean_net/
│   ├── preprocessing.py     # AudioPreprocessor: load, clean, extract MEL/WST/MFCC
│   ├── datasets.py          # Dataset classes + stratified split + dataloader builder
│   ├── training.py          # Unified training loop (single- and multi-input models)
│   ├── visualization.py     # Feature-map plotting helper
│   └── models/
│       ├── attention.py     # Self / cross / modal-fusion attention blocks
│       ├── blocks.py        # Residual & depthwise-separable conv blocks
│       ├── cetacean_net.py  # CetaceanNet + CetaceanNetLight
│       └── mobilenet.py     # MobileNetV2 baselines
├── notebooks/
│   └── init-pipeline.ipynb  # End-to-end walkthrough (the original Colab pipeline, refactored)
├── scripts/
│   └── train.py             # CLI training entry point
├── requirements.txt
└── pyproject.toml
```

---

## Installation

```bash
git clone https://github.com/Ramanuja-Chanduri/cetacean-net.git
cd cetacean-net
pip install -r requirements.txt
# or, to install the package itself (editable):
pip install -e .
```

---

## Dataset format

Organise audio as one sub-folder per class; the folder name is the label:

```
data/
├── humpback_whale/
│   ├── clip_0001.wav
│   └── ...
├── bottlenose_dolphin/
│   └── ...
└── orca/
    └── ...
```

Clips are resampled to 47.6 kHz, validated by duration and bit depth, then
centre-cropped or reflect-padded to a fixed length.

---

## Usage

### Command line

```bash
python scripts/train.py --data-path ./data --model cetacean_net --epochs 60
# disable Weights & Biases logging for a quick local run:
python scripts/train.py --data-path ./data --model cetacean_net_light --no-wandb
```

Available `--model` values: `cetacean_net`, `cetacean_net_light`,
`mobilenet_combined`, `mobilenet_mel`, `mobilenet_mfcc`, `mobilenet_wst`.

### Python API

```python
import numpy as np, torch
from cetacean_net import (
    AudioPreprocessor, CetaceanNet,
    build_dataloaders, stratified_split, train_model,
)

pre = AudioPreprocessor()
X, y = pre.basic_preprocess("./data")
mel, wst, mfcc = pre.compute_mel(X), pre.compute_wst(X), pre.compute_mfcc(X)

classes = np.unique(y)
label2id = {c.item(): i for i, c in enumerate(classes)}
labels = torch.tensor([label2id[v] for v in y])

train_idx, val_idx = stratified_split(labels)
loaders = build_dataloaders(mfcc, wst, mel, labels, train_idx, val_idx)
train_loader, val_loader = loaders["fusion"]

model = CetaceanNet(num_classes=len(classes))
train_model(model, train_loader, val_loader, y, label2id,
            num_classes=len(classes), epochs=60, use_wandb=False)
```

### Notebook

`notebooks/init-pipeline.ipynb` reproduces the entire pipeline end-to-end and
runs on Google Colab (mount Drive, point `DATA_PATH` at your dataset).

---

## Training details

- **Loss:** cross-entropy with **balanced class weights** to handle imbalance.
- **Optimizer:** AdamW (`lr=1e-4`, `weight_decay=1e-2`) by default.
- **Logging:** per-epoch train/val loss and accuracy, optionally to
  [Weights & Biases](https://wandb.ai/).

---

## License

Released under the MIT License.
