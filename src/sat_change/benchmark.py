"""Scene splits and scene discovery for the labelled real-data benchmark (v2).

The benchmark lives on disk as ``<root>/<split>/<scene>/images/<scene>_<date>.npy``
with ``<root>/<split>/<scene>/masks/<scene>.npy`` for the ``val`` and ``test``
splits. Train scenes have no masks on disk at all, so pretraining cannot read
a label even by accident.

Splits are by scene, never by patch, and are stored in ``docs/data/splits_v2.json``.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

from .io import discover_images, discover_masks, pair_images, scene_key

SPLIT_NAMES = ("train", "val", "test")


@dataclass(frozen=True)
class Scene:
    key: str
    before: Path
    after: Path
    mask: Path | None


def make_splits(official_train: list[str], official_test: list[str], n_val: int, seed: int) -> dict[str, list[str]]:
    """Hold the official test scenes out, and carve ``n_val`` validation scenes from the official training scenes.

    The validation scenes are drawn with ``random.Random(seed)`` from the sorted
    training list, so the split depends only on the scene names and the seed.
    """
    if not 0 < n_val < len(official_train):
        raise ValueError("n_val must leave at least one training scene")
    val = sorted(random.Random(seed).sample(sorted(official_train), n_val))
    splits = {
        "train": sorted(s for s in official_train if s not in val),
        "val": val,
        "test": sorted(official_test),
    }
    validate_splits(splits)
    return splits


def validate_splits(splits: dict[str, list[str]]) -> None:
    """Raise if a split is missing or empty, or if a scene sits in more than one split."""
    for name in SPLIT_NAMES:
        if not splits.get(name):
            raise ValueError(f"split {name!r} is missing or empty")
    seen: dict[str, str] = {}
    for name in SPLIT_NAMES:
        for scene in splits[name]:
            if scene in seen:
                raise ValueError(f"scene {scene!r} is in both {seen[scene]!r} and {name!r}")
            seen[scene] = name


def load_splits(path: str | Path) -> dict[str, list[str]]:
    payload = json.loads(Path(path).read_text())
    splits = {name: list(payload["splits"][name]) for name in SPLIT_NAMES}
    validate_splits(splits)
    return splits


def split_scenes(root: str | Path, split: str, splits: dict[str, list[str]] | None = None) -> list[Scene]:
    """The before/after pairs (and masks, when present) of one split, sorted by scene key.

    When ``splits`` is given, the scenes found on disk must match that split
    exactly. A scene from another split in this directory is an error, not a warning.
    """
    if split not in SPLIT_NAMES:
        raise ValueError(f"split must be one of {SPLIT_NAMES}, not {split!r}")
    split_root = Path(root) / split
    masks = {scene_key(p): p for p in discover_masks(split_root)}
    scenes = [Scene(scene_key(b), b, a, masks.get(scene_key(b))) for b, a in pair_images(split_root)]
    scenes.sort(key=lambda s: s.key)
    if splits is not None:
        found, expected = {s.key for s in scenes}, set(splits[split])
        if found != expected:
            raise ValueError(
                f"{split_root} does not match the {split!r} split: "
                f"missing {sorted(expected - found)}, unexpected {sorted(found - expected)}"
            )
    return scenes


def training_images(root: str | Path) -> list[Path]:
    """Imagery of the train split only. Masks are never returned."""
    return discover_images(Path(root) / "train")
