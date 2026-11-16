"""Pair-based pretraining: co-located patches and the dense temporal loss."""

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sat_change import train  # noqa: E402
from sat_change.data import TemporalPairDataset  # noqa: E402
from sat_change.model import SatelliteEncoder, dense_temporal_loss  # noqa: E402


def write_pair(root: Path, scene: str, size: int = 40, with_mask: bool = False) -> None:
    rng = np.random.default_rng(len(scene))
    images = root / scene / "images"
    images.mkdir(parents=True)
    base = rng.random((4, size, size)).astype(np.float32)
    np.save(images / f"{scene}_20160101.npy", base)
    np.save(images / f"{scene}_20180101.npy", base)  # identical dates: views differ only by brightness jitter
    if with_mask:
        (root / scene / "masks").mkdir()
        np.save(root / scene / "masks" / f"{scene}.npy", np.ones((size, size), np.float32))


def test_pair_patches_show_the_same_ground(tmp_path: Path, monkeypatch):
    write_pair(tmp_path, "alpha")
    dataset = TemporalPairDataset(tmp_path, patch_size=16, samples_per_epoch=4)
    # Switch brightness jitter off: with identical dates the two views must then be identical,
    # which proves the crop, flips and rotation are shared.
    monkeypatch.setattr("sat_change.data.photometric", lambda x: x)
    for i in range(4):
        before, after = dataset[i]
        assert before.shape == (4, 16, 16)
        assert torch.equal(before, after)


def test_pair_dataset_ignores_masks_and_needs_pairs(tmp_path: Path):
    write_pair(tmp_path, "alpha", with_mask=True)
    dataset = TemporalPairDataset(tmp_path, patch_size=16, samples_per_epoch=2)
    assert len(dataset.pairs) == 1
    assert all("masks" not in path.parts for pair in dataset.pairs for path in pair)
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="pairs"):
        TemporalPairDataset(empty)


def test_dense_loss_is_low_for_matching_features_and_high_for_shuffled():
    torch.manual_seed(0)
    feats = [torch.randn(4, 8, 8, 8), torch.randn(4, 16, 4, 4)]
    torch.manual_seed(1)
    matched = dense_temporal_loss(feats, [f.clone() for f in feats], points=8, temperature=0.1)
    torch.manual_seed(1)
    shuffled = dense_temporal_loss(feats, [f.flip(-1).roll(1, 0) for f in feats], points=8, temperature=0.1)
    assert matched.isfinite() and shuffled.isfinite()
    assert matched < shuffled


def test_dense_loss_backpropagates_through_the_encoder():
    torch.manual_seed(0)
    model = SatelliteEncoder(4, 8, 16)
    a, b = torch.rand(2, 4, 32, 32), torch.rand(2, 4, 32, 32)
    loss = dense_temporal_loss(model.features(a), model.features(b), points=4)
    loss.backward()
    assert model.stem[0].weight.grad is not None and model.stem[0].weight.grad.abs().sum() > 0


def test_training_with_the_pair_loss_runs_and_logs_it(tmp_path: Path):
    write_pair(tmp_path / "data", "alpha")
    write_pair(tmp_path / "data", "beta")
    out = tmp_path / "out"
    train.main(["--data", str(tmp_path / "data"), "--output", str(out), "--epochs", "1", "--samples-per-epoch", "8",
                "--batch-size", "4", "--patch-size", "16", "--width", "8", "--projection-dim", "16",
                "--pair-weight", "1.0", "--pair-points", "4", "--device", "cpu"])
    history = json.loads((out / "history.json").read_text())
    assert np.isfinite(history[0]["loss"]) and np.isfinite(history[0]["pair_loss"])
    assert (out / "best.pt").exists()
