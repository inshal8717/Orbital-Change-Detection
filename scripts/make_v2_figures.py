"""Figures for the v2 real-data results.

Three kinds of figure, all written under docs/img/v2/:

``panels``  One figure per benchmark scene of a split: before, after, the true
            change mask, and the predictions of the trained network and of the
            colour difference. Predictions are cut at the validation-fitted
            thresholds stored in a result file and coloured by outcome
            (white = correctly marked, orange = wrongly marked, blue = missed).
``pr``      Pooled precision-recall curves of every method on the test split,
            read from docs/data/v2_results.json.
``ranking`` Per-scene F1 of the network against the colour difference, as a
            small JSON summary used to pick which panels to show.

Usage:
    python scripts/make_v2_figures.py panels --config configs/v2/baseline.json --seed 42 --split val
    python scripts/make_v2_figures.py pr
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from sat_change import baselines  # noqa: E402
from sat_change.benchmark import load_splits, split_scenes  # noqa: E402
from sat_change.io import normalize_pair, read_image  # noqa: E402
from sat_change.postprocess import make_postprocess  # noqa: E402
from sat_change.runtime import load_model, score_pair  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
BACKGROUND = np.array([38, 44, 53], np.uint8)      # dark slate: unchanged, not marked
HIT = np.array([255, 255, 255], np.uint8)          # white: changed and marked
FALSE_ALARM = np.array([255, 140, 0], np.uint8)    # orange: marked, but not changed
MISS = np.array([70, 140, 255], np.uint8)          # blue: changed, but not marked
METHOD_LABELS = {
    "diff_shared": "Colour difference, shared stretch",
    "diff_raw": "Colour difference, raw values",
    "trained": "Trained network",
    "untrained": "Untrained network",
}


def rgb(before: np.ndarray, after: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """True-colour views of both dates, stretched together so brightness is comparable."""
    nb, na = normalize_pair(before, after)
    return tuple(np.clip(np.moveaxis(x[[2, 1, 0]], 0, -1), 0, 1) for x in (nb, na))


def outcome_image(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    image = np.empty(pred.shape + (3,), np.uint8)
    image[:] = BACKGROUND
    image[pred & truth] = HIT
    image[pred & ~truth] = FALSE_ALARM
    image[~pred & truth] = MISS
    return image


def f1(pred: np.ndarray, truth: np.ndarray) -> float:
    tp = (pred & truth).sum()
    return float(2 * tp / max(2 * tp + (pred & ~truth).sum() + (~pred & truth).sum(), 1))


def panels(args: argparse.Namespace) -> None:
    config = json.loads(Path(args.config).read_text())
    results = json.loads(Path(args.results).read_text())
    seed = f"seed{args.seed}"
    thresholds = {
        "trained": results["val"]["methods"]["trained"][seed]["threshold"],
        "diff_shared": results["val"]["methods"]["diff_shared"]["-"]["threshold"],
    }
    post = make_postprocess(config.get("post"))
    model = load_model(Path(args.runs) / config.get("checkpoint_from", config["name"]) / seed / "best.pt")
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for scene in split_scenes(args.data, args.split, load_splits(args.splits)):
        before, after = read_image(scene.before)[0], read_image(scene.after)[0]
        truth = read_image(scene.mask)[0][0] > 0.5
        scores = {
            "trained": score_pair(model, scene.before, scene.after, "cpu", **config.get("score", {}))[0],
            "diff_shared": baselines.shared_difference(before, after),
        }
        preds = {name: score >= thresholds[name] for name, score in scores.items()}
        if post is not None:
            preds = {name: post(pred) for name, pred in preds.items()}
        scene_f1 = {name: f1(pred, truth) for name, pred in preds.items()}
        summary.append({"scene": scene.key, "changed_fraction": float(truth.mean()), **{f"f1_{k}": v for k, v in scene_f1.items()}})

        before_rgb, after_rgb = rgb(before, after)
        images = [before_rgb, after_rgb, outcome_image(truth, truth),
                  outcome_image(preds["trained"], truth), outcome_image(preds["diff_shared"], truth)]
        titles = [f"Before ({scene.before.stem.split('_')[-1]})", f"After ({scene.after.stem.split('_')[-1]})",
                  f"What really changed ({truth.mean():.1%} of pixels)",
                  f"Trained network, F1 {scene_f1['trained']:.2f}",
                  f"Colour difference, F1 {scene_f1['diff_shared']:.2f}"]
        h, w = truth.shape
        fig, axes = plt.subplots(1, 5, figsize=(15, 3.2 * h / w + 0.9), dpi=110)
        for ax, image, title in zip(axes, images, titles):
            ax.imshow(image, interpolation="nearest")
            ax.set_title(title, fontsize=9)
            ax.axis("off")
        fig.suptitle(f"{scene.key} ({args.split} split). White: correctly marked. Orange: wrongly marked. Blue: missed.",
                     fontsize=10)
        fig.tight_layout()
        fig.savefig(out_dir / f"{args.prefix}{args.split}_{scene.key}.jpg", pil_kwargs={"quality": 88})
        plt.close(fig)
    (out_dir / f"{args.prefix}{args.split}_panels.json").write_text(json.dumps({
        "config": config["name"], "seed": args.seed, "thresholds": thresholds, "scenes": summary}, indent=2) + "\n")
    for row in summary:
        print(f"{row['scene']:12s} network F1 {row['f1_trained']:.3f}   colour difference F1 {row['f1_diff_shared']:.3f}")


def pr(args: argparse.Namespace) -> None:
    results = json.loads(Path(args.results).read_text())
    test = results["test"]["methods"]
    fig, ax = plt.subplots(figsize=(6.4, 4.6), dpi=130)
    styles = {"diff_shared": ("#1f77b4", "-"), "diff_raw": ("#17becf", "--"), "trained": ("#d62728", "-"), "untrained": ("#7f7f7f", ":")}
    for method, variants in test.items():
        colour, line = styles[method]
        for i, (label, result) in enumerate(sorted(variants.items())):
            curve = result["pr_curve"]
            ax.plot(curve["recall"], curve["precision"], line, color=colour, lw=1.6, alpha=0.9 if i == 0 else 0.45,
                    label=METHOD_LABELS[method] if i == 0 else None)
    masks_total = sum(row["tp"] + row["fn"] for row in next(iter(test["diff_shared"].values()))["val_threshold"]["per_scene"])
    pixels_total = sum(row["tp"] + row["fn"] + row["fp"] + row["tn"]
                       for row in next(iter(test["diff_shared"].values()))["val_threshold"]["per_scene"])
    ax.axhline(masks_total / pixels_total, color="black", lw=0.8, ls=(0, (1, 3)))
    ax.text(0.99, masks_total / pixels_total + 0.01, "marking pixels at random", ha="right", fontsize=8)
    ax.set_xlabel("Recall (share of truly changed pixels found)")
    ax.set_ylabel("Precision (share of marked pixels that truly changed)")
    ax.set_title(f"OSCD test scenes, all pixels pooled ({results['config']['name']})", fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    out = Path(args.output) / args.name
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(f"wrote {out}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    a = sub.add_parser("panels")
    a.add_argument("--config", required=True)
    a.add_argument("--results", required=True, help="result file holding the validation thresholds for this config")
    a.add_argument("--seed", type=int, default=42)
    a.add_argument("--split", choices=("val", "test"), default="val")
    a.add_argument("--prefix", default="", help="filename prefix, e.g. 'final_'")
    a.add_argument("--data", default=str(ROOT / "data" / "benchmark"))
    a.add_argument("--splits", default=str(ROOT / "docs" / "data" / "splits_v2.json"))
    a.add_argument("--runs", default=str(ROOT / "outputs" / "v2" / "runs"))
    a.add_argument("--output", default=str(ROOT / "docs" / "img" / "v2"))
    a.set_defaults(func=panels)
    b = sub.add_parser("pr")
    b.add_argument("--results", default=str(ROOT / "docs" / "data" / "v2_results.json"))
    b.add_argument("--output", default=str(ROOT / "docs" / "img" / "v2"))
    b.add_argument("--name", default="pr_curves_test.png")
    b.set_defaults(func=pr)
    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
