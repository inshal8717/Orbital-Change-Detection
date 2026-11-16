from __future__ import annotations

import numpy as np


def segmentation_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    """IoU, F1, precision, recall and accuracy for a binary prediction."""
    p = pred.astype(bool).ravel()
    y = target.astype(bool).ravel()
    tp = int(np.sum(p & y))
    fp = int(np.sum(p & ~y))
    fn = int(np.sum(~p & y))
    tn = int(np.sum(~p & ~y))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "iou": tp / max(tp + fp + fn, 1),
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "precision": precision,
        "recall": recall,
        "accuracy": (tp + tn) / max(tp + fp + fn + tn, 1),
    }


def rank_auc(score: np.ndarray, target: np.ndarray) -> float:
    """AUROC via the Mann-Whitney U statistic with average ranks for ties.

    Returns NaN when only one class is present.
    """
    s = score.ravel().astype(np.float64)
    y = target.astype(bool).ravel()
    n_pos = int(y.sum())
    n_neg = y.size - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _average_ranks(s)
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """1-based ranks where tied values share the mean of their positions."""
    order = np.argsort(values, kind="stable")
    sorted_vals = values[order]
    # Start index of each run of equal values in the sorted array.
    run_starts = np.flatnonzero(np.r_[True, sorted_vals[1:] != sorted_vals[:-1]])
    run_ends = np.r_[run_starts[1:], values.size]
    mean_ranks = (run_starts + run_ends + 1) / 2.0  # mean of 1-based positions in each run
    run_index = np.repeat(np.arange(run_starts.size), run_ends - run_starts)
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = mean_ranks[run_index]
    return ranks


def _ranked_counts(score: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Distinct thresholds (descending) with the true and false positives when predicting ``score >= threshold``."""
    s = score.ravel().astype(np.float64)
    y = target.astype(bool).ravel()
    order = np.argsort(-s, kind="stable")
    s, y = s[order], y[order]
    last_of_run = np.flatnonzero(np.r_[s[1:] != s[:-1], True])  # tied scores enter together
    tp = np.cumsum(y)[last_of_run]
    fp = (last_of_run + 1) - tp
    return s[last_of_run], tp, fp, int(y.sum())


def average_precision(score: np.ndarray, target: np.ndarray) -> float:
    """Area under the precision-recall curve (step-wise sum, as in scikit-learn).

    Returns NaN when there is no positive pixel.
    """
    _, tp, fp, n_pos = _ranked_counts(score, target)
    if n_pos == 0:
        return float("nan")
    precision = tp / (tp + fp)
    recall = tp / n_pos
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def pr_curve(score: np.ndarray, target: np.ndarray, points: int = 101) -> dict[str, list[float]]:
    """Precision at ``points`` evenly spaced recall levels, plus the thresholds that reach them."""
    thresholds, tp, fp, n_pos = _ranked_counts(score, target)
    if n_pos == 0:
        return {"recall": [], "precision": [], "threshold": []}
    recall = tp / n_pos
    levels = np.linspace(0, 1, points)[1:]
    index = np.minimum(np.searchsorted(recall, levels, side="left"), recall.size - 1)
    return {
        "recall": [float(v) for v in recall[index]],
        "precision": [float(v) for v in tp[index] / (tp[index] + fp[index])],
        "threshold": [float(v) for v in thresholds[index]],
    }


def best_f1_threshold(score: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    """The threshold ``t`` that maximises F1 of ``score >= t``, and that F1.

    Pass the pooled pixels of several scenes to fit one global threshold.
    """
    thresholds, tp, fp, n_pos = _ranked_counts(score, target)
    if n_pos == 0:
        raise ValueError("cannot fit a threshold without any changed pixel")
    f1 = 2 * tp / (2 * tp + fp + (n_pos - tp))
    best = int(np.argmax(f1))
    return float(thresholds[best]), float(f1[best])
