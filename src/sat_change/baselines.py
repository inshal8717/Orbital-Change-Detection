"""Change scores that use no network: plain per-pixel colour differences."""

from __future__ import annotations

import numpy as np

from .io import normalize_pair, normalize_reflectance


def shared_difference(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    """Colour distance after stretching both dates with one shared stretch."""
    xb, xa = normalize_pair(before, after)
    return np.sqrt(((xb - xa) ** 2).sum(0))


def per_photo_difference(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    """Colour distance after stretching each date on its own (the flawed scaling)."""
    xb, xa = normalize_reflectance(before), normalize_reflectance(after)
    return np.sqrt(((xb - xa) ** 2).sum(0))


def raw_difference(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    """Per-pixel Euclidean distance between the raw values, with no normalisation."""
    return np.sqrt(((before.astype(np.float32) - after.astype(np.float32)) ** 2).sum(0))
