from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .io import discover_images, normalize_pair, normalize_reflectance, pair_images, read_image


def augment(x: torch.Tensor) -> torch.Tensor:
    """Satellite-safe spatial and spectral augmentation for one C x H x W patch."""
    if random.random() < 0.5:
        x = x.flip(-1)
    if random.random() < 0.5:
        x = x.flip(-2)
    x = torch.rot90(x, random.randrange(4), (-2, -1))
    gain = torch.empty(x.shape[0], 1, 1).uniform_(0.85, 1.15)
    bias = torch.empty(x.shape[0], 1, 1).uniform_(-0.05, 0.05)
    x = x * gain + bias
    if random.random() < 0.5:
        x = x + torch.randn_like(x) * 0.015
    if random.random() < 0.25:
        x = F.avg_pool2d(x[None], 3, 1, 1)[0]
    return x.clamp(0, 1)


class ContrastivePatchDataset(Dataset):
    """Yields two augmented views of a random patch from a random image.

    Masks are excluded from discovery, so only imagery is sampled.
    """

    def __init__(self, root: str | Path, patch_size: int = 64, samples_per_epoch: int = 4096, bands: int = 4):
        self.paths = discover_images(root)
        if not self.paths:
            raise ValueError(f"No .npy/.tif imagery found in {root} (masks are excluded)")
        self.patch_size = patch_size
        self.samples_per_epoch = samples_per_epoch
        self.bands = bands
        self._cache: dict[Path, np.ndarray] = {}  # normalised images, so each file is read and stretched once

    def __len__(self) -> int:
        return self.samples_per_epoch

    def _load(self, path: Path) -> np.ndarray:
        if path not in self._cache:
            arr, _ = read_image(path)
            if arr.shape[0] < self.bands:
                raise ValueError(f"{path} has {arr.shape[0]} band(s); the model expects {self.bands}")
            arr = normalize_reflectance(arr[: self.bands])
            _, h, w = arr.shape
            if min(h, w) < self.patch_size:
                pad_h, pad_w = max(0, self.patch_size - h), max(0, self.patch_size - w)
                arr = np.pad(arr, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")
            self._cache[path] = arr
        return self._cache[path]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        arr = self._load(self.paths[index % len(self.paths)])
        _, h, w = arr.shape
        y = random.randint(0, h - self.patch_size)
        x = random.randint(0, w - self.patch_size)
        patch = torch.from_numpy(arr[:, y:y + self.patch_size, x:x + self.patch_size].copy())
        return augment(patch.clone()), augment(patch.clone())


def photometric(x: torch.Tensor) -> torch.Tensor:
    """The brightness, noise and blur part of ``augment``, with no flips or rotations."""
    gain = torch.empty(x.shape[0], 1, 1).uniform_(0.85, 1.15)
    bias = torch.empty(x.shape[0], 1, 1).uniform_(-0.05, 0.05)
    x = x * gain + bias
    if random.random() < 0.5:
        x = x + torch.randn_like(x) * 0.015
    if random.random() < 0.25:
        x = F.avg_pool2d(x[None], 3, 1, 1)[0]
    return x.clamp(0, 1)


class TemporalPairDataset(Dataset):
    """Yields the same random patch at two dates of one scene: ``(before, after)``.

    Both dates get one shared brightness stretch (as at scoring time) and the same
    flip and rotation, so every pixel of ``before`` shows the same ground as the
    pixel at the same position in ``after``. Brightness jitter is drawn separately
    for each date. No change mask is read: masks are excluded from discovery, and
    the benchmark's train split has none on disk.
    """

    def __init__(self, root: str | Path, patch_size: int = 64, samples_per_epoch: int = 4096, bands: int = 4):
        self.pairs = pair_images(root)
        if not self.pairs:
            raise ValueError(f"No before/after image pairs found in {root}")
        self.patch_size = patch_size
        self.samples_per_epoch = samples_per_epoch
        self.bands = bands
        self._cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def __len__(self) -> int:
        return self.samples_per_epoch

    def _load(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        if index not in self._cache:
            before, _ = read_image(self.pairs[index][0])
            after, _ = read_image(self.pairs[index][1])
            if before.shape != after.shape:
                raise ValueError(f"{self.pairs[index][0]} and {self.pairs[index][1]} have different shapes")
            if before.shape[0] < self.bands:
                raise ValueError(f"{self.pairs[index][0]} has {before.shape[0]} band(s); the model expects {self.bands}")
            before, after = normalize_pair(before[: self.bands], after[: self.bands])
            _, h, w = before.shape
            if min(h, w) < self.patch_size:
                pad = ((0, 0), (0, max(0, self.patch_size - h)), (0, max(0, self.patch_size - w)))
                before, after = np.pad(before, pad, mode="reflect"), np.pad(after, pad, mode="reflect")
            self._cache[index] = (before, after)
        return self._cache[index]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        before, after = self._load(index % len(self.pairs))
        _, h, w = before.shape
        y = random.randint(0, h - self.patch_size)
        x = random.randint(0, w - self.patch_size)
        window = (slice(None), slice(y, y + self.patch_size), slice(x, x + self.patch_size))
        b, a = torch.from_numpy(before[window].copy()), torch.from_numpy(after[window].copy())
        if random.random() < 0.5:
            b, a = b.flip(-1), a.flip(-1)
        if random.random() < 0.5:
            b, a = b.flip(-2), a.flip(-2)
        turns = random.randrange(4)
        b, a = torch.rot90(b, turns, (-2, -1)), torch.rot90(a, turns, (-2, -1))
        return photometric(b), photometric(a)
