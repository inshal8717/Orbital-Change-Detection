import numpy as np
import pytest

from sat_change.metrics import rank_auc, segmentation_metrics
from sat_change.postprocess import binary_mask, otsu_threshold


def test_metrics_perfect():
    y = np.array([[0, 0], [1, 1]], dtype=np.uint8)
    metrics = segmentation_metrics(y, y)
    assert metrics["f1"] == 1 and metrics["iou"] == 1
    assert rank_auc(y.astype(float), y) == 1


def test_auc_with_ties_matches_pairwise_definition():
    """AUROC equals P(score_pos > score_neg) + 0.5 * P(tie), regardless of pixel order."""
    rng = np.random.default_rng(0)
    score = rng.integers(0, 4, size=200).astype(float)  # many ties
    target = rng.random(200) < 0.4
    pos, neg = score[target], score[~target]
    expected = ((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (pos.size * neg.size)
    assert np.isclose(rank_auc(score, target), expected)


def test_auc_is_nan_when_one_class_missing():
    assert np.isnan(rank_auc(np.array([0.1, 0.2]), np.array([0, 0])))


def test_otsu_separates_bimodal_values():
    score = np.r_[np.zeros(100), np.ones(100)].reshape(10, 20)
    threshold = otsu_threshold(score)
    mask, _ = binary_mask(score, threshold)
    assert mask.sum() == 100


def test_average_precision_matches_hand_computed_value():
    from sat_change.metrics import average_precision

    score = np.array([0.9, 0.8, 0.7, 0.6])
    target = np.array([1, 0, 1, 0])
    # Positives are reached at ranks 1 and 3: (1/1 + 2/3) / 2.
    assert average_precision(score, target) == pytest.approx((1 + 2 / 3) / 2)
    assert average_precision(np.array([0.1, 0.9]), np.array([0, 1])) == pytest.approx(1.0)
    assert np.isnan(average_precision(score, np.zeros(4)))


def test_average_precision_treats_ties_as_one_cut():
    from sat_change.metrics import average_precision

    # All four pixels share one score, so the only cut marks everything: precision 0.5 at recall 1.
    assert average_precision(np.ones(4), np.array([1, 0, 1, 0])) == pytest.approx(0.5)


def test_best_f1_threshold_finds_the_separating_cut():
    from sat_change.metrics import best_f1_threshold, segmentation_metrics

    score = np.array([0.1, 0.2, 0.3, 0.8, 0.9, 0.35])
    target = np.array([0, 0, 0, 1, 1, 0])
    threshold, f1 = best_f1_threshold(score, target)
    assert threshold == pytest.approx(0.8) and f1 == pytest.approx(1.0)
    assert segmentation_metrics(score >= threshold, target)["f1"] == pytest.approx(f1)
    with pytest.raises(ValueError):
        best_f1_threshold(score, np.zeros(6))


def test_best_f1_threshold_agrees_with_brute_force():
    from sat_change.metrics import best_f1_threshold, segmentation_metrics

    rng = np.random.default_rng(0)
    target = rng.random(500) < 0.1
    score = np.round(rng.random(500) + 0.5 * target, 2)  # rounding creates ties
    threshold, f1 = best_f1_threshold(score, target)
    brute = max(segmentation_metrics(score >= t, target)["f1"] for t in np.unique(score))
    assert f1 == pytest.approx(brute)
    assert segmentation_metrics(score >= threshold, target)["f1"] == pytest.approx(brute)


def test_pr_curve_is_consistent_with_its_thresholds():
    from sat_change.metrics import pr_curve, segmentation_metrics

    rng = np.random.default_rng(1)
    target = rng.random(400) < 0.2
    score = rng.random(400) + 0.4 * target
    curve = pr_curve(score, target, points=11)
    assert len(curve["recall"]) == 10 and curve["recall"][-1] == pytest.approx(1.0)
    for recall, precision, threshold in zip(curve["recall"], curve["precision"], curve["threshold"]):
        m = segmentation_metrics(score >= threshold, target)
        assert m["recall"] == pytest.approx(recall) and m["precision"] == pytest.approx(precision)
