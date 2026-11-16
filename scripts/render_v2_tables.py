"""Fill the result tables in the v2 documents from the result files.

Markdown files hold blocks like::

    <!-- v2:test_main file=v2_results.json -->
    (generated table)
    <!-- /v2 -->

This script regenerates every block from ``docs/data/<file>``, so no metric in
a table is typed by hand. ``--check`` exits with an error when a block is out
of date, which tests/test_docs_v2.py uses.

Usage:
    python scripts/render_v2_tables.py            # rewrite the blocks
    python scripts/render_v2_tables.py --check    # verify only
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCUMENTS = ["README.md", "docs/V2_RESULTS.md", "docs/CASE_STUDIES.md", "docs/V2_EXPERIMENTS.md",
             "docs/V2_IMPROVEMENT_SUMMARY.md"]
BLOCK = re.compile(r"(<!-- v2:(\w+)((?: \w+=[^ ]+)*) -->\n)(.*?)(<!-- /v2 -->)", re.S)
LABELS = {
    "diff_shared": "Colour difference, shared stretch",
    "diff_raw": "Colour difference, raw values",
    "trained": "Trained network",
    "untrained": "Same network, untrained (random weights)",
}
ORDER = ["trained", "untrained", "diff_shared", "diff_raw"]


def load(name: str) -> dict:
    return json.loads((ROOT / "docs" / "data" / name).read_text())


def cell(stat: dict, digits: int = 3) -> str:
    if stat["n"] > 1:
        return f"{stat['mean']:.{digits}f} ± {stat['std']:.{digits}f}"
    return f"{stat['mean']:.{digits}f}"


def table(header: list[str], rows: list[list[str]]) -> str:
    align = ["---"] + ["---:"] * (len(header) - 1)
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(align) + " |"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines) + "\n"


def metric_table(headline: dict, rule: str, scope: str) -> str:
    rows = []
    for method in ORDER:
        if method not in headline:
            continue
        h = headline[method]
        rows.append([LABELS[method], cell(h[f"auroc_{scope}"]), cell(h[f"ap_{scope}"])]
                    + [cell(h[f"{rule}_{scope}_{m}"]) for m in ("f1", "iou", "precision", "recall")])
    return table(["Method", "AUROC", "PR-AUC", "F1", "IoU", "Precision", "Recall"], rows)


def test_main(data: dict, **_) -> str:
    return metric_table(data["test"]["headline"], "val_threshold", "pooled")


def test_otsu(data: dict, **_) -> str:
    return metric_table(data["test"]["headline"], "otsu", "pooled")


def test_scene_mean(data: dict, **_) -> str:
    return metric_table(data["test"]["headline"], "val_threshold", "mean")


def val_main(data: dict, **_) -> str:
    return metric_table(data["val"]["headline"], "loso", "pooled")


def val_thresholds(data: dict, **_) -> str:
    rows = []
    for method in ORDER:
        for label, result in sorted(data["val"]["methods"].get(method, {}).items()):
            name = LABELS[method] + ("" if label == "-" else f" ({label.replace('seed', 'seed ')})")
            rows.append([name, f"{result['threshold']:.4g}", f"{result['in_sample']['pooled']['f1']:.3f}",
                         f"{result['loso']['pooled']['f1']:.3f}", f"{result['otsu']['pooled']['f1']:.3f}"])
    return table(["Method", "Threshold fitted on validation", "Validation F1 at that threshold",
                  "Validation F1, leave one scene out", "Validation F1, per-scene Otsu"], rows)


def paired(data: dict, split: str = "test", **_) -> str:
    block = data[split]["paired_val_threshold" if split == "test" else "paired_loso"]
    names = {"vs_diff_shared": "colour difference (shared stretch)", "vs_untrained": "untrained network (same seed)"}
    rows = []
    for seed, comparisons in sorted(block.items()):
        for key, c in comparisons.items():
            lo, hi = c["pooled_f1_diff_ci95"]
            rows.append([f"Trained ({seed.replace('seed', 'seed ')}) minus {names[key]}", f"{c['pooled_f1_diff']:+.3f}",
                         f"{lo:+.3f} to {hi:+.3f}", f"{c['mean_scene_f1_diff']:+.3f} ± {c['std_scene_f1_diff']:.3f}",
                         f"{c['scenes_won']} of {c['scenes']}"])
    return table(["Comparison", "Pooled F1 difference", "95% interval (scenes resampled)",
                  "Per-scene F1 difference, mean ± spread", "Scenes the trained network wins"], rows)


def per_scene(data: dict, split: str = "test", seed: str = "seed42", **_) -> str:
    block = data[split]
    rule = "val_threshold" if split == "test" else "loso"
    rows = []
    for i, scene in enumerate(block["scenes"]):
        row = [scene]
        for method in ORDER:
            variants = block["methods"].get(method)
            if not variants:
                continue
            result = variants["-"] if "-" in variants else variants[seed]
            row += [f"{result[rule]['per_scene'][i]['f1']:.3f}", f"{result['ranking']['per_scene'][i]['auroc']:.3f}"]
        rows.append(row)
    header = ["Scene"]
    short = {"trained": "Trained", "untrained": "Untrained", "diff_shared": "Diff shared", "diff_raw": "Diff raw"}
    for method in ORDER:
        if method in block["methods"]:
            header += [f"{short[method]} F1", f"{short[method]} AUROC"]
    return table(header, rows)


def case_studies(data: dict, **_) -> str:
    rows = [[s["name"], f"{s['before_date']} to {s['after_date']}", f"{s['height']} × {s['width']}",
             f"{s['model_marked_fraction']:.1%}", f"{s['diff_marked_fraction']:.1%}", f"{s['agreement_iou']:.2f}"]
            for s in data["sites"]]
    return table(["Site", "Dates of the best acquisition in each window", "Size (pixels)", "Marked by the trained network",
                  "Marked by the colour difference", "Overlap of the two masks (IoU)"], rows)


def experiments(data: dict, **_) -> str:
    rows = []
    for e in data["experiments"]:
        rows.append([e["id"], e["change"], e["kind"], e["seeds"], e["primary"], e["delta"], e["ap"], e["decision"]])
    return table(["#", "Single change", "Kind", "Seeds", "Validation F1 (leave one scene out)", "Change vs best so far",
                  "Validation PR-AUC", "Decision"], rows)


RENDERERS = {f.__name__: f for f in (test_main, test_otsu, test_scene_mean, val_main, val_thresholds, paired,
                                     per_scene, case_studies, experiments)}


def render(text: str) -> str:
    def replace(match: re.Match) -> str:
        options = dict(pair.split("=") for pair in match.group(3).split())
        data = load(options.pop("file"))
        return match.group(1) + RENDERERS[match.group(2)](data, **options) + match.group(5)

    return BLOCK.sub(replace, text)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--check", action="store_true")
    args = p.parse_args(argv)
    stale = []
    for name in DOCUMENTS:
        path = ROOT / name
        if not path.exists():
            continue
        text = path.read_text()
        rendered = render(text)
        if rendered != text:
            stale.append(name)
            if not args.check:
                path.write_text(rendered)
                print(f"updated {name}")
    if args.check and stale:
        sys.exit(f"out-of-date result tables in: {', '.join(stale)}. Run python scripts/render_v2_tables.py")
    if not stale:
        print("all result tables are up to date")


if __name__ == "__main__":
    main()
