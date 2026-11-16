"""Build the experiment summary table from the result files.

docs/data/v2_experiments/log.json lists the experiments in order: which config
each one is, which earlier config it is compared with, and the decision.
This script reads the numbers from docs/data/v2_experiments/<config>.json and
writes docs/data/v2_experiments/index.json, which the summary table in
docs/V2_EXPERIMENTS.md is rendered from. No metric is typed by hand.

Usage:
    python scripts/summarize_experiments.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FOLDER = ROOT / "docs" / "data" / "v2_experiments"


def stat(path: Path, method: str, rule: str) -> dict:
    data = json.loads(path.read_text())
    variants = data["val"]["methods"][method]
    if rule == "ap":
        values = [v["ranking"]["pooled"]["ap"] for v in variants.values()]
    else:
        values = [v[rule]["pooled"]["f1"] for v in variants.values()]
    mean = sum(values) / len(values)
    std = (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5
    return {"mean": mean, "std": std, "n": len(values), "seeds": sorted(variants)}


def fmt(s: dict) -> str:
    return f"{s['mean']:.3f} ± {s['std']:.3f}" if s["n"] > 1 else f"{s['mean']:.3f}"


def main() -> None:
    log = json.loads((FOLDER / "log.json").read_text())
    rows = []
    for entry in log["experiments"]:
        method = entry.get("method", "trained")
        rule = entry.get("rule", "loso")
        primary = stat(FOLDER / f"{entry['config']}.json", method, rule)
        ap = stat(FOLDER / f"{entry['config']}.json", method, "ap")
        delta = "–"
        if entry.get("compare_to"):
            ref = stat(FOLDER / f"{entry['compare_to']['config']}.json", entry["compare_to"].get("method", "trained"),
                       entry["compare_to"].get("rule", "loso"))
            if entry["compare_to"].get("same_seeds"):
                ref_path = FOLDER / f"{entry['compare_to']['config']}.json"
                data = json.loads(ref_path.read_text())["val"]["methods"][entry["compare_to"].get("method", "trained")]
                values = [data[s][entry["compare_to"].get("rule", "loso")]["pooled"]["f1"] for s in primary["seeds"]]
                ref = {"mean": sum(values) / len(values)}
            delta = f"{primary['mean'] - ref['mean']:+.3f}"
        seeds = ", ".join(s.replace("seed", "") for s in primary["seeds"]) if primary["seeds"] != ["-"] else "none"
        rows.append({"id": str(entry["id"]), "change": entry["change"], "kind": entry["kind"], "seeds": seeds,
                     "primary": fmt(primary), "delta": delta, "ap": fmt(ap), "decision": entry["decision"],
                     "primary_mean": primary["mean"], "primary_std": primary["std"], "config": entry["config"]})
    (FOLDER / "index.json").write_text(json.dumps({"experiments": rows}, indent=2) + "\n")
    for row in rows:
        print(f"{row['id']:>3}  {row['primary']:16s} {row['delta']:>7s}  {row['decision']:10s} {row['change']}")


if __name__ == "__main__":
    main()
