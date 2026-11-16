"""The v2 evaluation protocol: validation-fitted thresholds, leave-one-scene-out, paired comparison."""

import numpy as np
import pytest

from sat_change import protocol
from sat_change.postprocess import make_postprocess, remove_small_components


def scene(seed: int, shift: float = 0.0, size: int = 40):
    rng = np.random.default_rng(seed)
    mask = np.zeros((size, size), bool)
    mask[5:15, 5:15] = True
    score = rng.random((size, size)) * 0.5 + shift + 0.6 * mask
    return score, mask


def flat_scene(inside: float, outside: float):
    mask = np.zeros((6, 6), bool)
    mask[:3] = True
    return np.where(mask, inside, outside), mask


def test_loso_never_fits_on_the_scene_it_scores():
    (a, mask), (b, _), (c, _) = flat_scene(0.9, 0.1), flat_scene(0.5, 0.1), flat_scene(0.7, 0.3)
    masks = [mask, mask, mask]
    thresholds = protocol.loso_thresholds([a, b, c], masks)
    assert thresholds == [0.5, 0.7, 0.5]
    # Changing scene b moves the thresholds fitted on it, and leaves its own threshold alone.
    b_changed, _ = flat_scene(0.8, 0.6)
    changed = protocol.loso_thresholds([a, b_changed, c], masks)
    assert changed[1] == thresholds[1]
    assert changed[0] == 0.7 and changed[0] != thresholds[0]


def test_in_sample_is_at_least_as_good_as_loso():
    scores, masks = zip(*[scene(i, shift=0.1 * i) for i in range(4)])
    result = protocol.evaluate_validation(list(scores), list(masks))
    assert result["in_sample"]["pooled"]["f1"] >= result["loso"]["pooled"]["f1"] - 1e-12
    assert set(result) == {"threshold", "in_sample", "loso", "otsu", "ranking"}
    assert len(result["loso"]["per_scene"]) == 4


def test_test_scores_use_the_given_threshold_only():
    scores, masks = zip(*[scene(i) for i in range(2)])
    result = protocol.evaluate_test(list(scores), list(masks), val_threshold=0.55)
    assert all(row["threshold"] == 0.55 for row in result["val_threshold"]["per_scene"])
    assert 0 < result["ranking"]["pooled"]["ap"] <= 1
    assert len(result["pr_curve"]["recall"]) == 100


def test_pooled_metrics_count_pixels_not_scenes():
    big = (np.ones((10, 10)), np.ones((10, 10), bool))        # 100 true positives
    small = (np.ones((2, 2)), np.zeros((2, 2), bool))          # 4 false positives
    result = protocol.summarize_masks([big[0], small[0]], [big[1], small[1]], [0.5, 0.5])
    assert result["pooled"]["precision"] == pytest.approx(100 / 104)
    assert result["mean"]["precision"] == pytest.approx(0.5)


def test_compare_is_paired_and_signed():
    scores, masks = zip(*[scene(i) for i in range(3)])
    good = protocol.summarize_masks(list(scores), list(masks), [0.55] * 3)
    bad = protocol.summarize_masks([np.zeros_like(s) for s in scores], list(masks), [0.55] * 3)
    result = protocol.compare(good, bad, n_boot=200)
    assert result["pooled_f1_diff"] > 0 and result["scenes_won"] == 3 and result["scenes_lost"] == 0
    assert result["pooled_f1_diff_ci95"][0] > 0
    assert protocol.compare(bad, good, n_boot=200)["pooled_f1_diff"] == pytest.approx(-result["pooled_f1_diff"])


def test_small_components_are_removed():
    mask = np.zeros((8, 8), bool)
    mask[0, 0] = True            # 1 pixel
    mask[3:5, 3:6] = True        # 6 pixels
    mask[7, 6:8] = True          # 2 pixels
    assert remove_small_components(mask, 3).sum() == 6
    assert remove_small_components(mask, 2).sum() == 8
    assert remove_small_components(mask, 1).sum() == 9
    assert make_postprocess(None) is None and make_postprocess({"min_size": 0}) is None
    assert make_postprocess({"min_size": 3})(mask).sum() == 6
    with pytest.raises(ValueError):
        make_postprocess({"blur": 2})
