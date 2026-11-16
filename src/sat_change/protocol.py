"""The v2 evaluation protocol: thresholds fitted on validation scenes, test scored with them.

Vocabulary used in the result files:

- ``pooled``: all pixels of all scenes in the split counted together (the usual way OSCD results are reported).
- ``mean``: the metric computed per scene, then averaged over scenes (the convention of this project's v1 table).
- ``in_sample``: one global threshold fitted on every validation scene and applied to the same scenes. Optimistic.
- ``loso``: leave one scene out. Each validation scene is thresholded with a value fitted on the *other*
  validation scenes. This is the honest estimate of how a global threshold transfers to an unseen scene, and
  its pooled F1 is the primary metric of the model-improvement loop.
- ``otsu``: v1's rule, a per-scene Otsu threshold that uses no labels.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from .metrics import average_precision, best_f1_threshold, pr_curve, rank_auc
from .postprocess import otsu_threshold

PostProcess = Callable[[np.ndarray], np.ndarray]


def _counts(pred: np.ndarray, target: np.ndarray) -> dict[str, int]:
    p, y = pred.astype(bool), target.astype(bool)
    return {"tp": int((p & y).sum()), "fp": int((p & ~y).sum()), "fn": int((~p & y).sum()), "tn": int((~p & ~y).sum())}


def _scores_from_counts(tp: float, fp: float, fn: float) -> dict[str, float]:
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "f1": 2 * tp / max(2 * tp + fp + fn, 1),
        "iou": tp / max(tp + fp + fn, 1),
        "precision": precision,
        "recall": recall,
    }


def pooled(scores: list[np.ndarray], masks: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    return np.concatenate([s.ravel() for s in scores]), np.concatenate([m.ravel() for m in masks])


def fit_threshold(scores: list[np.ndarray], masks: list[np.ndarray]) -> float:
    """One global threshold that maximises pooled-pixel F1 over the given scenes."""
    return best_f1_threshold(*pooled(scores, masks))[0]


def loso_thresholds(scores: list[np.ndarray], masks: list[np.ndarray]) -> list[float]:
    """For each scene, the global threshold fitted on all the other scenes."""
    if len(scores) < 2:
        raise ValueError("leave-one-scene-out needs at least two scenes")
    return [
        fit_threshold(scores[:i] + scores[i + 1:], masks[:i] + masks[i + 1:])
        for i in range(len(scores))
    ]


def summarize_masks(
    scores: list[np.ndarray],
    masks: list[np.ndarray],
    thresholds: list[float],
    post: PostProcess | None = None,
) -> dict:
    """Binary-map metrics when scene ``i`` is cut at ``thresholds[i]`` (then optionally post-processed)."""
    per_scene = []
    for score, mask, threshold in zip(scores, masks, thresholds):
        pred = score >= threshold
        if post is not None:
            pred = post(pred)
        counts = _counts(pred, mask)
        per_scene.append({"threshold": float(threshold), **counts,
                          **_scores_from_counts(counts["tp"], counts["fp"], counts["fn"])})
    total = {k: sum(row[k] for row in per_scene) for k in ("tp", "fp", "fn")}
    keys = ("f1", "iou", "precision", "recall")
    return {
        "pooled": _scores_from_counts(total["tp"], total["fp"], total["fn"]),
        "mean": {k: float(np.mean([row[k] for row in per_scene])) for k in keys},
        "std": {k: float(np.std([row[k] for row in per_scene])) for k in keys},
        "per_scene": per_scene,
    }


def summarize_ranking(scores: list[np.ndarray], masks: list[np.ndarray]) -> dict:
    """Threshold-free quality of the score maps: AUROC and average precision (PR-AUC)."""
    per_scene = [{"auroc": rank_auc(s, m), "ap": average_precision(s, m)} for s, m in zip(scores, masks)]
    all_scores, all_masks = pooled(scores, masks)
    return {
        "pooled": {"auroc": rank_auc(all_scores, all_masks), "ap": average_precision(all_scores, all_masks)},
        "mean": {k: float(np.mean([row[k] for row in per_scene])) for k in ("auroc", "ap")},
        "std": {k: float(np.std([row[k] for row in per_scene])) for k in ("auroc", "ap")},
        "per_scene": per_scene,
    }


def evaluate_validation(scores: list[np.ndarray], masks: list[np.ndarray], post: PostProcess | None = None) -> dict:
    """Everything measured on the validation scenes, including the threshold that test will use."""
    threshold = fit_threshold(scores, masks)
    return {
        "threshold": threshold,
        "in_sample": summarize_masks(scores, masks, [threshold] * len(scores), post),
        "loso": summarize_masks(scores, masks, loso_thresholds(scores, masks), post),
        "otsu": summarize_masks(scores, masks, [otsu_threshold(s) for s in scores], post),
        "ranking": summarize_ranking(scores, masks),
    }


def evaluate_test(
    scores: list[np.ndarray],
    masks: list[np.ndarray],
    val_threshold: float,
    post: PostProcess | None = None,
    curve_points: int = 101,
) -> dict:
    """Score held-out scenes with a threshold fitted elsewhere. Nothing here is fitted on these scenes
    except v1's per-scene Otsu, which uses no labels."""
    return {
        "threshold": val_threshold,
        "val_threshold": summarize_masks(scores, masks, [val_threshold] * len(scores), post),
        "otsu": summarize_masks(scores, masks, [otsu_threshold(s) for s in scores], post),
        "ranking": summarize_ranking(scores, masks),
        "pr_curve": pr_curve(*pooled(scores, masks), points=curve_points),
    }


def compare(a: dict, b: dict, n_boot: int = 2000, seed: int = 0) -> dict:
    """Paired comparison of two ``summarize_masks`` results on the same scenes: ``a`` minus ``b``.

    Reports the per-scene F1 differences and a scene-level bootstrap interval for the pooled F1
    difference (scenes are resampled with replacement; pixels inside a scene stay together).
    """
    rows_a, rows_b = a["per_scene"], b["per_scene"]
    if len(rows_a) != len(rows_b):
        raise ValueError("both results must cover the same scenes")
    diffs = np.array([ra["f1"] - rb["f1"] for ra, rb in zip(rows_a, rows_b)])
    rng = np.random.default_rng(seed)
    n = len(rows_a)

    def pooled_f1(rows: list[dict], index: np.ndarray) -> float:
        tp = sum(rows[i]["tp"] for i in index)
        fp = sum(rows[i]["fp"] for i in index)
        fn = sum(rows[i]["fn"] for i in index)
        return 2 * tp / max(2 * tp + fp + fn, 1)

    boot = np.empty(n_boot)
    for k in range(n_boot):
        index = rng.integers(0, n, n)
        boot[k] = pooled_f1(rows_a, index) - pooled_f1(rows_b, index)
    return {
        "pooled_f1_diff": a["pooled"]["f1"] - b["pooled"]["f1"],
        "pooled_f1_diff_ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        "mean_scene_f1_diff": float(diffs.mean()),
        "std_scene_f1_diff": float(diffs.std()),
        "scenes_won": int((diffs > 0).sum()),
        "scenes_lost": int((diffs < 0).sum()),
        "scenes": n,
    }
