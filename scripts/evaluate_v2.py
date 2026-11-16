"""Score one v2 configuration on the real-data benchmark.

Four methods are scored on the same scenes with the same threshold rules:

    diff_shared   colour difference after one shared brightness stretch (no network)
    diff_raw      colour difference on the raw values (no network)
    trained       the encoder pretrained on the train split (one result per --seeds entry)
    untrained     the same encoder with random weights (one result per --seeds entry)

Thresholds are always fitted on the validation scenes. ``--split val`` (the
default) reports validation results only and never opens a test file.
``--split test`` applies the validation thresholds to the test scenes and
records the pass in a ledger; a second test pass for the same config name is
refused unless ``--rerun-reason`` explains it.

Usage:
    python scripts/train_v2.py --config configs/v2/baseline.json --seed 42
    python scripts/evaluate_v2.py --config configs/v2/baseline.json --seeds 42 43 44
    python scripts/evaluate_v2.py --config configs/v2/baseline.json --seeds 42 43 44 --split test
"""

from __future__ import annotations

import argparse
import datetime
import json
import subprocess
from pathlib import Path

import numpy as np
import torch

from sat_change import baselines, protocol
from sat_change.benchmark import Scene, load_splits, split_scenes
from sat_change.io import read_image
from sat_change.model import SatelliteEncoder
from sat_change.postprocess import make_postprocess
from sat_change.runtime import load_model, score_pair

ROOT = Path(__file__).resolve().parent.parent
NETWORK_METHODS = ("trained", "untrained")


def git_commit() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def load_mask(scene: Scene) -> np.ndarray:
    if scene.mask is None:
        raise SystemExit(f"scene {scene.key} has no mask")
    return read_image(scene.mask)[0][0] > 0.5


def build_scorers(config: dict, seeds: list[int], runs: Path, methods: list[str]) -> dict[str, dict[str, callable]]:
    """``{method: {variant label: scene -> score map}}``. Network methods have one variant per seed."""
    score_cfg = config.get("score", {})
    train_cfg = config["train"]

    def network(model):
        return lambda scene: score_pair(model, scene.before, scene.after, "cpu", **score_cfg)[0]

    def array_scorer(fn):
        return lambda scene: fn(read_image(scene.before)[0], read_image(scene.after)[0])

    scorers: dict[str, dict[str, callable]] = {}
    if "diff_shared" in methods:
        scorers["diff_shared"] = {"-": array_scorer(baselines.shared_difference)}
    if "diff_raw" in methods:
        scorers["diff_raw"] = {"-": array_scorer(baselines.raw_difference)}
    if "trained" in methods:
        scorers["trained"] = {}
        for seed in seeds:
            checkpoint = runs / config.get("checkpoint_from", config["name"]) / f"seed{seed}" / "best.pt"
            if not checkpoint.exists():
                raise SystemExit(f"missing {checkpoint}; run scripts/train_v2.py --config ... --seed {seed} first")
            scorers["trained"][f"seed{seed}"] = network(load_model(checkpoint))
    if "untrained" in methods:
        scorers["untrained"] = {}
        for seed in seeds:
            torch.manual_seed(seed)
            model = SatelliteEncoder(4, train_cfg.get("width", 32), train_cfg.get("projection_dim", 128)).eval()
            scorers["untrained"][f"seed{seed}"] = network(model)
    return scorers


def across_variants(results: dict[str, dict], path: list[str]) -> dict[str, float]:
    """Mean and standard deviation over variants (seeds) of one scalar, addressed by ``path``."""
    values = []
    for result in results.values():
        node = result
        for key in path:
            node = node[key]
        values.append(node)
    return {"mean": float(np.mean(values)), "std": float(np.std(values)), "n": len(values)}


def headline(results: dict[str, dict], threshold_rules: list[str]) -> dict:
    """Seed-averaged headline numbers for one method."""
    out = {"auroc_pooled": across_variants(results, ["ranking", "pooled", "auroc"]),
           "ap_pooled": across_variants(results, ["ranking", "pooled", "ap"]),
           "auroc_mean": across_variants(results, ["ranking", "mean", "auroc"]),
           "ap_mean": across_variants(results, ["ranking", "mean", "ap"])}
    for rule in threshold_rules:
        for scope in ("pooled", "mean"):
            for metric in ("f1", "iou", "precision", "recall"):
                out[f"{rule}_{scope}_{metric}"] = across_variants(results, [rule, scope, metric])
    return out


def paired(results: dict[str, dict[str, dict]], rule: str) -> dict:
    """Trained network against the colour difference and against random weights, seed by seed."""
    out = {}
    if "trained" not in results:
        return out
    for seed, trained in results["trained"].items():
        row = {}
        if "diff_shared" in results:
            row["vs_diff_shared"] = protocol.compare(trained[rule], results["diff_shared"]["-"][rule])
        if "untrained" in results and seed in results["untrained"]:
            row["vs_untrained"] = protocol.compare(trained[rule], results["untrained"][seed][rule])
        out[seed] = row
    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--split", choices=("val", "test"), default="val")
    p.add_argument("--seeds", type=int, nargs="+", default=[42])
    p.add_argument("--methods", nargs="+", default=["diff_shared", "diff_raw", "trained", "untrained"])
    p.add_argument("--data", default=str(ROOT / "data" / "benchmark"))
    p.add_argument("--splits", default=str(ROOT / "docs" / "data" / "splits_v2.json"))
    p.add_argument("--runs", default=str(ROOT / "outputs" / "v2" / "runs"))
    p.add_argument("--output", help="result file; default docs/data/v2_experiments/<name>.json (val) "
                                    "or docs/data/v2_results.json (test)")
    p.add_argument("--ledger", default=str(ROOT / "docs" / "data" / "v2_test_ledger.json"))
    p.add_argument("--rerun-reason", help="required to score the test split again for the same config name")
    args = p.parse_args(argv)

    config = json.loads(Path(args.config).read_text())
    splits = load_splits(args.splits)
    post = make_postprocess(config.get("post"))
    scorers = build_scorers(config, args.seeds, Path(args.runs), args.methods)

    val_scenes = split_scenes(args.data, "val", splits)
    val_masks = [load_mask(s) for s in val_scenes]
    val: dict[str, dict[str, dict]] = {}
    val_scores: dict[tuple[str, str], list[np.ndarray]] = {}
    for method, variants in scorers.items():
        val[method] = {}
        for label, score_fn in variants.items():
            val_scores[method, label] = [score_fn(s) for s in val_scenes]
            val[method][label] = protocol.evaluate_validation(val_scores[method, label], val_masks, post)
        print(f"val  {method:12s} LOSO pooled F1 = {across_variants(val[method], ['loso', 'pooled', 'f1'])['mean']:.4f}"
              f"  AP = {across_variants(val[method], ['ranking', 'pooled', 'ap'])['mean']:.4f}")

    report = {
        "config": config,
        "commit": git_commit(),
        "seeds": args.seeds,
        "splits": splits,
        "val": {
            "scenes": [s.key for s in val_scenes],
            "methods": val,
            "headline": {m: headline(r, ["loso", "in_sample", "otsu"]) for m, r in val.items()},
            "paired_loso": paired(val, "loso"),
        },
    }

    if args.split == "test":
        ledger_path = Path(args.ledger)
        ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else []
        earlier = [entry for entry in ledger if entry["config"] == config["name"]]
        if earlier and not args.rerun_reason:
            raise SystemExit(f"the test split was already scored for config {config['name']!r} "
                             f"({earlier[-1]['time']}). Pass --rerun-reason to record why it is scored again.")
        test_scenes = split_scenes(args.data, "test", splits)
        test_masks = [load_mask(s) for s in test_scenes]
        test: dict[str, dict[str, dict]] = {}
        for method, variants in scorers.items():
            test[method] = {}
            for label, score_fn in variants.items():
                scores = [score_fn(s) for s in test_scenes]
                test[method][label] = protocol.evaluate_test(scores, test_masks, val[method][label]["threshold"], post)
            print(f"test {method:12s} pooled F1 = "
                  f"{across_variants(test[method], ['val_threshold', 'pooled', 'f1'])['mean']:.4f}")
        report["test"] = {
            "scenes": [s.key for s in test_scenes],
            "methods": test,
            "headline": {m: headline(r, ["val_threshold", "otsu"]) for m, r in test.items()},
            "paired_val_threshold": paired(test, "val_threshold"),
        }
        ledger.append({
            "config": config["name"],
            "commit": report["commit"],
            "time": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "seeds": args.seeds,
            "rerun_reason": args.rerun_reason,
        })
        ledger_path.write_text(json.dumps(ledger, indent=2) + "\n")

    default = ROOT / "docs" / "data" / ("v2_results.json" if args.split == "test" else f"v2_experiments/{config['name']}.json")
    output = Path(args.output) if args.output else default
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=1) + "\n")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
