"""Build docs/case-studies.html: a before/after slider for each UAE case-study site.

Reads the site definitions (docs/data/case_studies.json) and the run report
(docs/data/v2_case_studies.json) written by scripts/run_case_studies.py. The
page is self-contained apart from the images in docs/img/v2/. It uses a few
lines of inline JavaScript for the slider and the overlay switches.

Usage:
    python scripts/build_case_studies_page.py
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

STYLE = """
:root { color-scheme: light dark; --fg: #1c2330; --muted: #5b6577; --bg: #f7f8fa; --card: #ffffff; --line: #d9dee7;
        --model: #ff8c00; --diff: #00a8d6; }
@media (prefers-color-scheme: dark) { :root { --fg: #e8ecf3; --muted: #9aa5b8; --bg: #12161d; --card: #1a2029; --line: #2c3545; } }
* { box-sizing: border-box; }
body { margin: 0; font: 16px/1.55 system-ui, -apple-system, "Segoe UI", sans-serif; color: var(--fg); background: var(--bg); }
main { max-width: 860px; margin: 0 auto; padding: 24px 16px 64px; }
h1 { font-size: 1.6rem; line-height: 1.25; margin: 0 0 8px; }
h2 { font-size: 1.2rem; margin: 40px 0 6px; }
p { margin: 8px 0; }
.note, figcaption { color: var(--muted); font-size: 0.9rem; }
a { color: inherit; }
figure { margin: 14px 0; padding: 12px; background: var(--card); border: 1px solid var(--line); border-radius: 12px; }
.compare { position: relative; width: 100%; overflow: hidden; border-radius: 8px; line-height: 0; touch-action: pan-y; }
.compare img { width: 100%; height: auto; display: block; }
.compare img.layer { position: absolute; inset: 0; height: 100%; }
.compare .line { position: absolute; top: 0; bottom: 0; width: 2px; background: #fff; box-shadow: 0 0 0 1px rgba(0,0,0,.45); pointer-events: none; }
.compare .tag { position: absolute; top: 8px; padding: 2px 8px; border-radius: 999px; font: 600 12px/1.6 system-ui, sans-serif;
                color: #fff; background: rgba(0,0,0,.6); }
.compare .tag.left { left: 8px; } .compare .tag.right { right: 8px; }
.controls { display: flex; flex-wrap: wrap; gap: 6px 18px; align-items: center; margin-top: 10px; font-size: 0.92rem; }
.controls input[type=range] { flex: 1 1 100%; width: 100%; accent-color: var(--fg); }
.controls label { display: inline-flex; align-items: center; gap: 6px; }
.swatch { width: 12px; height: 12px; border-radius: 3px; display: inline-block; }
table { border-collapse: collapse; font-size: 0.92rem; margin: 8px 0; }
td, th { padding: 4px 12px 4px 0; text-align: left; vertical-align: top; }
th { color: var(--muted); font-weight: 600; }
ul { padding-left: 20px; margin: 6px 0; }
"""

SCRIPT = """
document.querySelectorAll('figure').forEach(function (figure) {
  var compare = figure.querySelector('.compare');
  var range = figure.querySelector('input[type=range]');
  var before = compare.querySelector('.before');
  var line = compare.querySelector('.line');
  function set() {
    before.style.clipPath = 'inset(0 ' + (100 - range.value) + '% 0 0)';
    line.style.left = 'calc(' + range.value + '% - 1px)';
  }
  range.addEventListener('input', set);
  set();
  figure.querySelectorAll('input[type=checkbox]').forEach(function (box) {
    var layer = compare.querySelector('.' + box.dataset.layer);
    function show() { layer.style.display = box.checked ? 'block' : 'none'; }
    box.addEventListener('change', show);
    show();
  });
});
"""


def esc(text: object) -> str:
    return html.escape(str(text))


def site_section(site: dict, run: dict, tag: str = "") -> str:
    key = tag + site["key"]
    sources = "".join(f'<li><a href="{esc(s["url"])}">{esc(s["title"])}</a></li>' for s in site["sources"])
    return f"""
<h2>{esc(site["name"])}</h2>
<p>{esc(site["change"])}</p>
<figure>
  <div class="compare">
    <img src="img/v2/case_{key}_after.jpg" alt="{esc(site["name"])}, after ({esc(run["after_date"])})" width="{run["width"]}" height="{run["height"]}">
    <img class="layer model" src="img/v2/case_{key}_model.png" alt="">
    <img class="layer diff" src="img/v2/case_{key}_diff.png" alt="">
    <img class="layer before" src="img/v2/case_{key}_before.jpg" alt="{esc(site["name"])}, before ({esc(run["before_date"])})">
    <div class="line"></div>
    <span class="tag left">Before {esc(run["before_date"][:4])}</span>
    <span class="tag right">After {esc(run["after_date"][:4])}</span>
  </div>
  <div class="controls">
    <input type="range" min="0" max="100" value="50" aria-label="Slide between the before and after image">
    <label><input type="checkbox" data-layer="model" checked><span class="swatch" style="background:var(--model)"></span>
      Trained network ({run["model_marked_fraction"]:.0%} of the scene)</label>
    <label><input type="checkbox" data-layer="diff"><span class="swatch" style="background:var(--diff)"></span>
      Colour difference ({run["diff_marked_fraction"]:.0%} of the scene)</label>
  </div>
  <figcaption>{site["size_km"]} km across, 10 m per pixel. Before: best of {esc(site["before"].replace("/", " to "))}.
  After: best of {esc(site["after"].replace("/", " to "))}. The marks are drawn on the after image, so they show to the
  right of the slider line. Contains modified Copernicus Sentinel data.</figcaption>
</figure>
<p class="note">Sources for the change and its dates:</p>
<ul class="note">{sources}</ul>
"""


def build(spec: dict, report: dict) -> str:
    runs = {run["key"]: run for run in report["sites"]}
    tag = report.get("tag", "")
    sections = "".join(site_section(site, runs[site["key"]], tag) for site in spec["sites"] if site["key"] in runs)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Case studies: change seen from space in the UAE</title>
<style>{STYLE}</style>
</head>
<body>
<main>
<h1>Case studies: three places in the UAE that changed</h1>
<p>Each picture below is two real Sentinel-2 photos of the same place. Drag the slider to move between them. The
coloured marks are what each method flagged as changed. There is no answer key for these sites, so the marks are
shown for inspection and are not scored.</p>
<p class="note">Model: <code>{esc(report["config"])}</code>, seed {esc(report["seed"])}. Both thresholds were fitted
on labelled validation scenes of the OSCD benchmark and applied here unchanged. Full write-up:
<a href="CASE_STUDIES.md">CASE_STUDIES.md</a>. Benchmark results: <a href="V2_RESULTS.md">V2_RESULTS.md</a>.
<a href="index.html">Back to the project page</a>.</p>
{sections}
</main>
<script>{SCRIPT}</script>
</body>
</html>
"""


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sites", default=str(ROOT / "docs" / "data" / "case_studies.json"))
    p.add_argument("--report", default=str(ROOT / "docs" / "data" / "v2_case_studies.json"))
    p.add_argument("--output", default=str(ROOT / "docs" / "case-studies.html"))
    args = p.parse_args(argv)
    page = build(json.loads(Path(args.sites).read_text()), json.loads(Path(args.report).read_text()))
    Path(args.output).write_text(page, encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
