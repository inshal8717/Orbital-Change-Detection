"""The case-study site list and the slider page."""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import build_case_studies_page  # noqa: E402

SPEC = json.loads((ROOT / "docs" / "data" / "case_studies.json").read_text())


def fake_report() -> dict:
    return {"config": "demo", "seed": 42, "sites": [
        {"key": site["key"], "before_date": "2016-01-20", "after_date": "2022-03-14", "width": 500, "height": 400,
         "model_marked_fraction": 0.25, "diff_marked_fraction": 0.2} for site in SPEC["sites"]]}


def test_every_site_documents_its_change_and_sources():
    assert len(SPEC["sites"]) >= 2
    for site in SPEC["sites"]:
        lon, lat = site["center"]
        assert 51 < lon < 57 and 22 < lat < 27, f"{site['key']} is not in the UAE"
        assert site["size_km"] <= 10
        assert len(site["change"]) > 40 and site["center_source"]
        assert len(site["sources"]) >= 2
        assert all(s["url"].startswith("https://") for s in site["sources"])
        for window in (site["before"], site["after"]):
            assert re.fullmatch(r"\d{4}-\d{2}-\d{2}/\d{4}-\d{2}-\d{2}", window)
        assert site["before"] < site["after"]
        # Both windows cover the same months, so season and sun angle are comparable.
        assert site["before"][5:7] == site["after"][5:7]


def test_page_has_a_slider_and_both_overlays_for_every_site():
    page = build_case_studies_page.build(SPEC, fake_report())
    for site in SPEC["sites"]:
        for suffix in ("before.jpg", "after.jpg", "model.png", "diff.png"):
            assert f"img/v2/case_{site['key']}_{suffix}" in page
        assert site["name"] in page
        for source in site["sources"]:
            assert source["url"] in page
    assert page.count('type="range"') == len(SPEC["sites"])
    assert page.count("<script>") == 1 and "src=\"http" not in page  # one inline script, nothing loaded from elsewhere


def test_page_escapes_text():
    spec = {"sites": [{**SPEC["sites"][0], "name": "A <b>site</b>"}]}
    page = build_case_studies_page.build(spec, fake_report())
    assert "A &lt;b&gt;site&lt;/b&gt;" in page and "<b>site</b>" not in page
