"""The v2 documents must agree with the result files they quote."""

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = [ROOT / "README.md", ROOT / "docs" / "V2_RESULTS.md", ROOT / "docs" / "CASE_STUDIES.md",
        ROOT / "docs" / "V2_EXPERIMENTS.md", ROOT / "docs" / "V2_IMPROVEMENT_SUMMARY.md", ROOT / "docs" / "DATA_SOURCES.md",
        ROOT / "docs" / "case-studies.html"]


def test_result_tables_are_generated_from_the_result_files():
    result = subprocess.run([sys.executable, str(ROOT / "scripts" / "render_v2_tables.py"), "--check"],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_local_links_and_images_resolve():
    missing = []
    for doc in DOCS:
        if not doc.exists():
            continue
        text = doc.read_text()
        targets = re.findall(r"\]\(([^)\s]+)\)", text) + re.findall(r'src="([^"]+)"', text)
        for target in targets:
            if target.startswith(("http://", "https://", "#", "mailto:")):
                continue
            path = (doc.parent / target.split("#")[0]).resolve()
            if not path.exists():
                missing.append(f"{doc.relative_to(ROOT)} -> {target}")
    assert not missing, "broken local links: " + ", ".join(missing)


def test_no_large_or_raw_data_files_are_tracked():
    tracked = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True).stdout.split("\n")
    too_big = [name for name in tracked if name and (ROOT / name).exists() and (ROOT / name).stat().st_size > 5_000_000]
    assert not too_big, f"files over 5 MB are tracked: {too_big}"
    raw = [name for name in tracked if name.startswith("data/") or name.endswith((".tif", ".tiff", ".npy", ".pt", ".zip"))]
    assert not raw, f"raw data or checkpoints are tracked: {raw}"
