"""Tests for the Sentinel-2 composite logic in scripts/fetch_sentinel2.py.

These build tiny local GeoTIFFs and pass an identity ``sign`` function, so no
network access is needed.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")
from rasterio.transform import from_origin  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import fetch_sentinel2 as fetch  # noqa: E402

CRS = 32611
TRANSFORM = from_origin(500000, 4000000, 10, 10)
H = W = 8
CLEAR = 1000.0
BRIGHT_CLOUD = 9000.0


def _write(path: Path, arr: np.ndarray, nodata, dtype="float32") -> None:
    with rasterio.open(
        path, "w", driver="GTiff", height=H, width=W, count=1, dtype=dtype,
        crs=f"EPSG:{CRS}", transform=TRANSFORM, nodata=nodata,
    ) as dst:
        dst.write(arr, 1)


def _item(tmp: Path, name: str, *, cloud: bool = False, nodata: bool = False, baseline: str | None = None,
          value: float = CLEAR):
    """Fake STAC item. Pixel (0, 0) is cloud (bright, SCL=8) or nodata (0) when requested."""
    assets = {}
    for band in fetch.BANDS:
        arr = np.full((H, W), value, np.float32)
        if cloud:
            arr[0, 0] = BRIGHT_CLOUD
        if nodata:
            arr[0, 0] = 0
        path = tmp / f"{name}_{band}.tif"
        _write(path, arr, nodata=0)
        assets[band] = SimpleNamespace(href=str(path))
    scl = np.full((H, W), 4, np.uint8)  # vegetation
    if cloud:
        scl[0, 0] = 8  # cloud medium probability
    scl_path = tmp / f"{name}_SCL.tif"
    _write(scl_path, scl, nodata=0, dtype="uint8")
    assets["SCL"] = SimpleNamespace(href=str(scl_path))
    properties = {} if baseline is None else {"s2:processing_baseline": baseline}
    return SimpleNamespace(id=name, assets=assets, properties=properties)


def _identity(item):
    return item


def test_composite_ignores_cloud_pixels(tmp_path: Path):
    """A bright cloud in two of three acquisitions must not change the median."""
    items = [_item(tmp_path, "clear"), _item(tmp_path, "cloud1", cloud=True), _item(tmp_path, "cloud2", cloud=True)]
    comp = fetch._composite(items, CRS, TRANSFORM, H, W, use_scl=True, sign=_identity)
    assert comp.shape == (4, H, W)
    assert np.allclose(comp[:, 0, 0], CLEAR)
    assert np.allclose(comp[:, 4, 4], CLEAR)


def test_composite_without_scl_ignores_nodata(tmp_path: Path):
    items = [_item(tmp_path, "clear"), _item(tmp_path, "nd1", nodata=True), _item(tmp_path, "nd2", nodata=True)]
    comp = fetch._composite(items, CRS, TRANSFORM, H, W, use_scl=False, sign=_identity)
    assert np.allclose(comp[:, 0, 0], CLEAR)


def test_write_tiff_stores_nodata_as_zero(tmp_path: Path):
    arr = np.full((1, H, W), 500.0, np.float32)
    arr[0, 0, 0] = np.nan
    out = tmp_path / "out.tif"
    fetch._write_tiff(out, arr, CRS, TRANSFORM)
    with rasterio.open(out) as src:
        data = src.read(1)
    assert np.isfinite(data).all()
    assert data[0, 0] == 0
    assert data[1, 1] == 500


def test_boa_offset_follows_the_processing_baseline():
    def item(baseline):
        return SimpleNamespace(properties={"s2:processing_baseline": baseline})

    assert fetch._boa_offset(item("02.12")) == 0
    assert fetch._boa_offset(item("03.00")) == 0
    assert fetch._boa_offset(item("04.00")) == 1000
    assert fetch._boa_offset(item("05.10")) == 1000
    assert fetch._boa_offset(SimpleNamespace(properties={})) == 0
    assert fetch._boa_offset(SimpleNamespace()) == 0


def test_composites_before_and_after_2022_share_one_scale(tmp_path: Path):
    """The same ground must give the same value whichever processing baseline delivered it."""
    old = fetch._composite([_item(tmp_path, "old", baseline="02.12")], CRS, TRANSFORM, H, W, use_scl=True, sign=_identity)
    new = fetch._composite([_item(tmp_path, "new", baseline="05.10", value=CLEAR + 1000)], CRS, TRANSFORM, H, W,
                           use_scl=True, sign=_identity)
    assert np.allclose(old, CLEAR) and np.allclose(new, CLEAR)


def test_offset_removal_keeps_nodata_and_never_writes_zero_for_valid_pixels(tmp_path: Path):
    item = _item(tmp_path, "dark", baseline="04.00", value=900.0, nodata=True)  # darker than the offset
    comp = fetch._composite([item], CRS, TRANSFORM, H, W, use_scl=False, sign=_identity)
    assert np.isnan(comp[:, 0, 0]).all()
    assert np.allclose(comp[:, 4, 4], 1.0)
