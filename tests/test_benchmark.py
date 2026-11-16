"""Splits and scene discovery for the v2 benchmark."""

import json
from pathlib import Path

import numpy as np
import pytest

from sat_change.benchmark import load_splits, make_splits, split_scenes, training_images, validate_splits

ROOT = Path(__file__).resolve().parent.parent
SPLITS_FILE = ROOT / "docs" / "data" / "splits_v2.json"


def write_scene(root: Path, split: str, scene: str, with_mask: bool) -> None:
    images = root / split / scene / "images"
    images.mkdir(parents=True)
    np.save(images / f"{scene}_20160101.npy", np.zeros((4, 8, 8), np.float32))
    np.save(images / f"{scene}_20180101.npy", np.ones((4, 8, 8), np.float32))
    if with_mask:
        masks = root / split / scene / "masks"
        masks.mkdir()
        np.save(masks / f"{scene}.npy", np.ones((8, 8), np.float32))


def test_committed_splits_are_disjoint_and_complete():
    splits = load_splits(SPLITS_FILE)
    scenes = [s for name in ("train", "val", "test") for s in splits[name]]
    assert len(scenes) == len(set(scenes)) == 24
    # The test split is OSCD's official test list.
    assert splits["test"] == sorted(
        ["brasilia", "montpellier", "norcia", "rio", "saclay_w", "valencia", "dubai", "lasvegas", "milano", "chongqing"]
    )


def test_committed_splits_follow_the_recorded_rule():
    payload = json.loads(SPLITS_FILE.read_text())
    splits = payload["splits"]
    rebuilt = make_splits(splits["train"] + splits["val"], splits["test"], payload["n_val"], payload["seed"])
    assert rebuilt == splits


def test_a_scene_in_two_splits_is_rejected():
    with pytest.raises(ValueError, match="both"):
        validate_splits({"train": ["a", "b"], "val": ["b"], "test": ["c"]})
    with pytest.raises(ValueError, match="empty"):
        validate_splits({"train": ["a"], "val": [], "test": ["c"]})


def test_make_splits_is_deterministic_and_disjoint():
    train = [f"t{i}" for i in range(14)]
    test = [f"x{i}" for i in range(10)]
    first = make_splits(train, test, 5, 42)
    assert first == make_splits(list(reversed(train)), test, 5, 42)
    assert len(first["val"]) == 5 and len(first["train"]) == 9
    assert not set(first["val"]) & set(first["train"])


def test_masks_are_never_read_as_imagery(tmp_path: Path):
    write_scene(tmp_path, "train", "alpha", with_mask=True)  # a stray train mask must still be ignored
    write_scene(tmp_path, "val", "beta", with_mask=True)
    images = training_images(tmp_path)
    assert [p.name for p in images] == ["alpha_20160101.npy", "alpha_20180101.npy"]
    assert all("masks" not in p.parts for p in images)

    scenes = split_scenes(tmp_path, "val")
    assert [s.key for s in scenes] == ["beta"]
    assert "masks" not in scenes[0].before.parts and "masks" not in scenes[0].after.parts
    assert scenes[0].mask is not None and scenes[0].mask.parent.name == "masks"


def test_training_never_sees_other_splits(tmp_path: Path):
    write_scene(tmp_path, "train", "alpha", with_mask=False)
    write_scene(tmp_path, "val", "beta", with_mask=True)
    write_scene(tmp_path, "test", "gamma", with_mask=True)
    assert {p.parent.parent.name for p in training_images(tmp_path)} == {"alpha"}


def test_a_scene_in_the_wrong_split_directory_is_an_error(tmp_path: Path):
    write_scene(tmp_path, "val", "beta", with_mask=True)
    write_scene(tmp_path, "val", "gamma", with_mask=True)
    splits = {"train": ["alpha"], "val": ["beta"], "test": ["gamma"]}
    with pytest.raises(ValueError, match="unexpected"):
        split_scenes(tmp_path, "val", splits)
