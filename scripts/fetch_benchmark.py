"""Download the OSCD benchmark and convert it to this project's data layout.

OSCD (Onera Satellite Change Detection) is 24 registered Sentinel-2 L1C pairs
with pixel-level urban change masks. See docs/DATA_SOURCES.md for the source,
licence and citation.

This script:

1. downloads the three official archives (about 513 MB) and checks their SHA-256,
2. extracts only the four 10 m bands this project uses (B02, B03, B04, B08) from
   the registered ``imgs_*_rect`` folders, plus the labels and text files,
3. writes each scene as ``<output>/<split>/<scene>/images/<scene>_<YYYYMMDD>.npy``
   (float32, 4 x H x W, original digital numbers) and, for val and test scenes
   only, ``<output>/<split>/<scene>/masks/<scene>.npy`` (0 = unchanged, 1 = changed).

Train scenes get no mask on disk, so pretraining cannot read a label.

The split comes from docs/data/splits_v2.json. With ``--write-splits`` the file
is (re)created from OSCD's own train.txt / test.txt: the official test scenes
are the test split, and ``--n-val`` scenes are drawn from the official training
scenes with ``--seed``.

Usage (from the repo root; needs the ``geo`` extra for rasterio):
    python scripts/fetch_benchmark.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

from sat_change.benchmark import SPLIT_NAMES, load_splits, make_splits
from sat_change.io import write_image

ROOT = Path(__file__).resolve().parent.parent
BASE_URL = "https://huggingface.co/datasets/hkristen/oscd/resolve/main/"
PREFIX = "Onera Satellite Change Detection dataset - "
ARCHIVES = {
    "Images": "940b87887511058a933e67cd6d0e43e2eb825a55d8e79a50983dee7f23003656",
    "Train Labels": "89fb54cd12ad0dbea6c447528139dec305b865294215434bf6dd170fb8fd3ca5",
    "Test Labels": "2e195eaa1b788b99fa93ea8073e3780bc0b763000b0c49dbf70548acf1e5d67d",
}
BANDS = ("B02", "B03", "B04", "B08")  # blue, green, red, near-infrared; all native 10 m
WANTED_SUFFIXES = tuple(f"_rect/{band}.tif" for band in BANDS) + (".txt", "-cm.tif")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download(raw: Path) -> None:
    raw.mkdir(parents=True, exist_ok=True)
    for name, expected in ARCHIVES.items():
        target = raw / f"{PREFIX}{name}.zip"
        if not target.exists():
            url = BASE_URL + urllib.parse.quote(target.name)
            print(f"downloading {target.name}")
            with urllib.request.urlopen(url) as response, target.open("wb") as fh:
                shutil.copyfileobj(response, fh)
        actual = sha256(target)
        if actual != expected:
            raise SystemExit(f"{target.name}: SHA-256 is {actual}, expected {expected}. Delete the file and retry.")


def extract(raw: Path) -> None:
    """Extract only the members this project reads, refusing any path that leaves ``raw``."""
    for name in ARCHIVES:
        with zipfile.ZipFile(raw / f"{PREFIX}{name}.zip") as archive:
            for member in archive.namelist():
                if not member.endswith(WANTED_SUFFIXES):
                    continue
                destination = (raw / member).resolve()
                if raw.resolve() not in destination.parents:
                    raise SystemExit(f"refusing to extract {member!r} outside {raw}")
                if not destination.exists():
                    archive.extract(member, raw)


def read_band_stack(folder: Path) -> np.ndarray:
    import rasterio

    bands = []
    for band in BANDS:
        with rasterio.open(folder / f"{band}.tif") as src:
            bands.append(src.read(1).astype(np.float32))
    return np.stack(bands)


def read_dates(path: Path) -> tuple[str, str]:
    fields = dict(line.split(":") for line in path.read_text().split("\n") if ":" in line)
    return fields["date_1"].strip(), fields["date_2"].strip()


def read_mask(raw: Path, scene: str) -> np.ndarray:
    """OSCD label rasters store 1 for unchanged and 2 for changed (not 0/1 as its README says)."""
    import rasterio

    matches = sorted(raw.glob(f"{PREFIX}* Labels/{scene}/cm/{scene}-cm.tif"))
    if len(matches) != 1:
        raise SystemExit(f"expected one label raster for {scene}, found {len(matches)}")
    with rasterio.open(matches[0]) as src:
        label = src.read(1)
    values = set(np.unique(label).tolist())
    if not values <= {1, 2}:
        raise SystemExit(f"{matches[0]}: unexpected label values {sorted(values)}")
    return (label == 2).astype(np.float32)


def convert(raw: Path, output: Path, splits: dict[str, list[str]]) -> list[dict]:
    images = raw / f"{PREFIX}Images"
    rows = []
    for split in SPLIT_NAMES:
        shutil.rmtree(output / split, ignore_errors=True)
        for scene in splits[split]:
            date_1, date_2 = read_dates(images / scene / "dates.txt")
            before = read_band_stack(images / scene / "imgs_1_rect")
            after = read_band_stack(images / scene / "imgs_2_rect")
            if before.shape != after.shape:
                raise SystemExit(f"{scene}: before {before.shape} and after {after.shape} differ")
            scene_dir = output / split / scene
            write_image(scene_dir / "images" / f"{scene}_{date_1}.npy", before)
            write_image(scene_dir / "images" / f"{scene}_{date_2}.npy", after)
            row = {"scene": scene, "split": split, "before": date_1, "after": date_2,
                   "height": before.shape[1], "width": before.shape[2]}
            if split != "train":  # train labels are never opened
                mask = read_mask(raw, scene)
                if mask.shape != before.shape[1:]:
                    raise SystemExit(f"{scene}: mask {mask.shape} does not match imagery {before.shape[1:]}")
                row["changed_fraction"] = round(float(mask.mean()), 5)
                write_image(scene_dir / "masks" / f"{scene}.npy", mask)
            rows.append(row)
    return rows


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", default=str(ROOT / "data" / "benchmark"))
    p.add_argument("--splits", default=str(ROOT / "docs" / "data" / "splits_v2.json"))
    p.add_argument("--write-splits", action="store_true", help="(re)create the splits file from OSCD's train/test lists")
    p.add_argument("--n-val", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    output = Path(args.output)
    raw = output / "raw"
    download(raw)
    extract(raw)

    splits_path = Path(args.splits)
    if args.write_splits or not splits_path.exists():
        images = raw / f"{PREFIX}Images"
        official_train = (images / "train.txt").read_text().strip().split(",")
        official_test = (images / "test.txt").read_text().strip().split(",")
        splits = make_splits(official_train, official_test, args.n_val, args.seed)
        splits_path.parent.mkdir(parents=True, exist_ok=True)
        splits_path.write_text(json.dumps({
            "dataset": "OSCD (Onera Satellite Change Detection)",
            "rule": "test = OSCD official test scenes; val = n_val scenes drawn from the official training scenes "
                    "with random.Random(seed).sample(sorted(train), n_val); train = the rest",
            "seed": args.seed,
            "n_val": args.n_val,
            "splits": splits,
        }, indent=2) + "\n")
        print(f"wrote {splits_path}")
    splits = load_splits(splits_path)

    rows = convert(raw, output, splits)
    (output / "scenes.json").write_text(json.dumps(rows, indent=2) + "\n")
    for split in SPLIT_NAMES:
        print(f"{split:5s} {len(splits[split]):2d} scenes: {', '.join(splits[split])}")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
