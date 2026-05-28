"""Audio loading, validation, and acoustic feature extraction.

This module defines :class:`AudioPreprocessor`, which walks a directory tree of
``.wav`` files (one sub-folder per class), loads and normalises each waveform to
a fixed length, and converts the raw signal into three complementary
time-frequency representations:

* **MEL spectrogram** - perceptually-scaled magnitude spectrogram.
* **WST** - Wavelet Scattering Transform, a translation-invariant, stable
  representation well suited to short transient bio-acoustic events.
* **MFCC** - Mel-frequency cepstral coefficients, a compact classic descriptor.
"""

from __future__ import annotations

import os

import numpy as np
import torch
import torchaudio
from kymatio import Scattering1D
from torchaudio.transforms import MelSpectrogram, Resample
from tqdm import tqdm


class AudioPreprocessor:
    """Load raw audio and turn it into model-ready feature tensors.

    Parameters
    ----------
    sample_rate:
        Target sample rate (Hz). Every clip is resampled to this rate.
    target_samples:
        Fixed length, in samples, that every waveform is cropped or padded to.
    valid_bit_depths:
        Clips whose bit depth is not in this list are skipped.
    min_duration_sec, max_duration_sec:
        Clips shorter than ``min_duration_sec`` are skipped; clips longer than
        ``max_duration_sec`` are truncated before the centre crop.
    device:
        Torch device used for feature computation. Defaults to CUDA if available.
    """

    def __init__(
        self,
        sample_rate: int = 47600,
        target_samples: int = 8000,
        valid_bit_depths: list[int] | None = None,
        min_duration_sec: float = 0.1,
        max_duration_sec: float = 10.0,
        device: torch.device | None = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.target_samples = target_samples
        self.valid_bit_depths = valid_bit_depths or [16, 24, 32]
        self.min_duration_sec = min_duration_sec
        self.max_duration_sec = max_duration_sec
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.resampler: Resample | None = None

    # ------------------------------------------------------------------ #
    # Loading & cleaning
    # ------------------------------------------------------------------ #
    def load_audio(self, filepath: str):
        """Load a single file and resample it to ``self.sample_rate``.

        Returns ``(None, None)`` if the bit depth is unsupported.
        """
        info = torchaudio.info(filepath)
        if info.bits_per_sample not in self.valid_bit_depths:
            return None, None

        waveform, sr = torchaudio.load(filepath)

        if sr != self.sample_rate:
            if self.resampler is None or self.resampler.orig_freq != sr:
                self.resampler = Resample(orig_freq=sr, new_freq=self.sample_rate)
            waveform = self.resampler(waveform)

        return waveform, self.sample_rate

    def valid_duration(self, waveform: torch.Tensor) -> int:
        """Classify a clip's duration: ``-1`` too short, ``1`` too long, ``0`` ok."""
        duration = waveform.shape[1] / self.sample_rate
        if duration < self.min_duration_sec:
            return -1
        if duration > self.max_duration_sec:
            return 1
        return 0

    def center_cut_or_pad(self, waveform: torch.Tensor) -> torch.Tensor:
        """Centre-crop or reflect-pad a waveform to exactly ``target_samples``."""
        length = waveform.shape[1]

        if length == self.target_samples:
            return waveform
        if length > self.target_samples:
            center = length // 2
            left = center - (self.target_samples // 2)
            return waveform[:, left : left + self.target_samples]

        pad_total = self.target_samples - length
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        return torch.nn.functional.pad(waveform, (pad_left, pad_right), mode="reflect")

    def basic_preprocess(self, root_folder: str):
        """Walk ``root_folder`` and build a ``(X, y)`` pair.

        The directory layout is expected to be ``root_folder/<label>/<clip>.wav``;
        the immediate parent folder name is used as the class label.

        Returns
        -------
        X : torch.Tensor
            Stacked waveforms of shape ``[N, target_samples]`` on ``self.device``.
        y : numpy.ndarray
            Array of string labels of length ``N``.
        """
        X_list: list[torch.Tensor] = []
        y_list: list[str] = []

        for root, _dirs, files in os.walk(root_folder):
            for file in tqdm(files, desc=f"Processing {root}"):
                if not file.endswith(".wav"):
                    continue

                filepath = os.path.join(root, file)
                label = os.path.basename(os.path.dirname(filepath))

                waveform, _sr = self.load_audio(filepath)
                if waveform is None:
                    continue

                check = self.valid_duration(waveform)
                if check == -1:
                    continue
                if check == 1:
                    max_samples = int(self.max_duration_sec * self.sample_rate)
                    waveform = waveform[:, :max_samples]

                waveform = self.center_cut_or_pad(waveform)

                X_list.append(waveform)
                y_list.append(label)

        X = torch.cat(X_list, dim=0).to(self.device)
        y = np.array(y_list)
        return X, y

    # ------------------------------------------------------------------ #
    # Feature extraction
    # ------------------------------------------------------------------ #
    def compute_mel(self, X: torch.Tensor, n_mels: int = 32) -> torch.Tensor:
        """Compute normalised MEL spectrograms. Output: ``[B, 1, n_mels, time]``."""
        spectr = MelSpectrogram(
            sample_rate=self.sample_rate,
            n_mels=n_mels,
            normalized=True,
            window_fn=torch.hann_window,
        ).to(self.device)

        MX = spectr(X.to(self.device))
        return MX.unsqueeze(1)

    def compute_wst(self, X: torch.Tensor, J: int = 8, Q: int = 14) -> torch.Tensor:
        """Compute the Wavelet Scattering Transform. Output: ``[B, 1, paths, time]``.

        Each scattering order (0, 1, 2) is independently median-normalised to
        keep the very different per-order magnitudes on a comparable scale.
        """
        T = X.shape[1]
        scattering = Scattering1D(J=J, shape=T, Q=Q)

        SX = torch.from_numpy(scattering(X.detach().cpu().numpy())).to(self.device)

        meta = scattering.meta()
        order0 = np.where(meta["order"] == 0)[0]
        order1 = np.where(meta["order"] == 1)[0]
        order2 = np.where(meta["order"] == 2)[0]

        def median_norm(t: torch.Tensor) -> torch.Tensor:
            md = torch.median(t)
            sn = torch.std(t)
            return (t - md) / sn if sn > 1e-6 else t - md

        SX_med = SX.clone()
        for i in range(SX.shape[0]):
            SX_med[i, order0] = median_norm(SX[i, order0])
            SX_med[i, order1] = median_norm(SX[i, order1])
            SX_med[i, order2] = median_norm(SX[i, order2])

        return SX_med.unsqueeze(1)

    def compute_mfcc(self, X: torch.Tensor, n_mfcc: int = 13) -> torch.Tensor:
        """Compute MFCC features. Output: ``[B, 1, n_mfcc, time]``."""
        mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=self.sample_rate,
            n_mfcc=n_mfcc,
            melkwargs={
                "n_fft": 1024,
                "n_mels": 32,
                "hop_length": 512,
                "mel_scale": "htk",
            },
        ).to(self.device)

        MFCC_features = mfcc_transform(X.to(self.device))
        return MFCC_features.unsqueeze(1)
