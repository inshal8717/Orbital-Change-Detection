from __future__ import annotations

import numpy as np


def otsu_threshold(values: np.ndarray, bins: int = 256) -> float:
    x = values[np.isfinite(values)].astype(np.float64)
    if x.size == 0 or np.ptp(x) == 0:
        return float(x[0]) if x.size else 0.0
    hist, edges = np.histogram(x, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    weight1 = np.cumsum(hist)
    weight2 = np.cumsum(hist[::-1])[::-1]
    mean1 = np.cumsum(hist * centers) / np.maximum(weight1, 1)
    mean2 = (np.cumsum((hist * centers)[::-1]) / np.maximum(weight2[::-1], 1))[::-1]
    variance = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
    return float(centers[int(np.argmax(variance))])


def binary_mask(score: np.ndarray, threshold: float | None = None) -> tuple[np.ndarray, float]:
    threshold = otsu_threshold(score) if threshold is None else threshold
    return (score >= threshold).astype(np.uint8), float(threshold)



def remove_small_components(mask: np.ndarray, min_size: int) -> np.ndarray:
    """Drop 4-connected groups of changed pixels smaller than ``min_size`` pixels."""
    mask = mask.astype(bool)
    if min_size <= 1 or not mask.any():
        return mask
    h, w = mask.shape
    labels = np.zeros((h, w), np.int32)
    keep = np.zeros_like(mask)
    current = 0
    for y, x in zip(*np.nonzero(mask)):
        if labels[y, x]:
            continue
        current += 1
        labels[y, x] = current
        stack, members = [(y, x)], [(y, x)]
        while stack:
            cy, cx = stack.pop()
            for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not labels[ny, nx]:
                    labels[ny, nx] = current
                    stack.append((ny, nx))
                    members.append((ny, nx))
        if len(members) >= min_size:
            ys, xs = zip(*members)
            keep[list(ys), list(xs)] = True
    return keep


def make_postprocess(config: dict | None):
    """Build the mask post-processing step named in a v2 config, or ``None`` when there is none."""
    if not config:
        return None
    min_size = int(config.get("min_size", 0))
    unknown = set(config) - {"min_size"}
    if unknown:
        raise ValueError(f"unknown post-processing option(s): {sorted(unknown)}")
    if min_size <= 1:
        return None
    return lambda mask: remove_small_components(mask, min_size)
