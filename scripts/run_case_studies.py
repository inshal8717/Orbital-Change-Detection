"""Run the model and the colour-difference baseline on the UAE case-study sites.

Sites, date windows and sources come from docs/data/case_studies.json. For each
site this script:

1. (with ``--fetch``) downloads the Sentinel-2 pair with scripts/fetch_sentinel2.py,
2. runs ``sat_change.predict`` with the trained checkpoint and the model's
   validation-fitted threshold, writing the score map and mask to
   ``outputs/v2/<site>/``,
3. computes the shared-stretch colour difference and cuts it at its own
   validation-fitted threshold,
4. writes a four-panel figure and the separate images used by the slider page
   to ``docs/img/v2/``,
5. records what each method marked in ``docs/data/v2_case_studies.json``.

There is no ground-truth mask for these sites, so nothing here is a score. The
numbers are the share of the scene each method marked and how far they agree.

Usage:
    python scripts/run_case_studies.py --fetch          # download, then run
    python scripts/run_case_studies.py                  # run on data already in data/real
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from sat_change import baselines, predict  # noqa: E402
from sat_change.io import normalize_pair, pair_images, read_image, write_image  # noqa: E402
from sat_change.postprocess import make_postprocess, otsu_threshold  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MODEL_COLOUR = (255, 140, 0)     # orange
DIFF_COLOUR = (0, 200, 255)      # cyan


def fetch_command(site: dict, fetch: dict, output: Path) -> list[str]:
    lon, lat = site["center"]
    command = [
        sys.executable, str(ROOT / "scripts" / "fetch_sentinel2.py"),
        f"--center={lon},{lat}", "--size", str(site["size_km"]),
        "--before", site["before"], "--after", site["after"],
        "--scene", site["key"], "--output", str(output),
        "--cloud-max", str(fetch["cloud_max"]), "--limit", str(fetch["limit"]),
    ]
    if not fetch["scl_mask"]:
        command.append("--no-scl")
    return command


def overlay(photo: np.ndarray, mask: np.ndarray, colour: tuple[int, int, int], alpha: float = 0.6) -> np.ndarray:
    out = photo.copy()
    out[mask] = (1 - alpha) * out[mask] + alpha * np.array(colour) / 255
    return out


def save_rgba(path: Path, mask: np.ndarray, colour: tuple[int, int, int]) -> None:
    rgba = np.zeros(mask.shape + (4,), np.uint8)
    rgba[mask] = (*colour, 165)
    plt.imsave(path, rgba)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sites", default=str(ROOT / "docs" / "data" / "case_studies.json"))
    p.add_argument("--config", default=str(ROOT / "configs" / "v2" / "baseline.json"))
    p.add_argument("--results", default=str(ROOT / "docs" / "data" / "v2_results.json"),
                   help="result file holding the validation-fitted thresholds for this config")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fetch", action="store_true", help="download the imagery first")
    p.add_argument("--data", default=str(ROOT / "data" / "real"))
    p.add_argument("--runs", default=str(ROOT / "outputs" / "v2" / "runs"))
    p.add_argument("--outputs", default=str(ROOT / "outputs" / "v2"))
    p.add_argument("--figures", default=str(ROOT / "docs" / "img" / "v2"))
    p.add_argument("--report", default=str(ROOT / "docs" / "data" / "v2_case_studies.json"))
    p.add_argument("--tag", default="", help="inserted into figure names and output folders, e.g. 'final_'")
    args = p.parse_args(argv)

    spec = json.loads(Path(args.sites).read_text())
    config = json.loads(Path(args.config).read_text())
    results = json.loads(Path(args.results).read_text())
    seed = f"seed{args.seed}"
    thresholds = {
        "model": results["val"]["methods"]["trained"][seed]["threshold"],
        "diff": results["val"]["methods"]["diff_shared"]["-"]["threshold"],
    }
    post = make_postprocess(config.get("post"))
    score_cfg = config.get("score", {})
    checkpoint = Path(args.runs) / config.get("checkpoint_from", config["name"]) / seed / "best.pt"
    figures = Path(args.figures)
    figures.mkdir(parents=True, exist_ok=True)

    report = {"config": config["name"], "seed": args.seed, "tag": args.tag, "thresholds": thresholds,
              "threshold_source": "fitted on the OSCD validation scenes (Level-1C); applied here to Level-2A imagery",
              "fetch": spec["fetch"], "sites": []}
    for site in spec["sites"]:
        key = site["key"]
        if args.fetch:
            subprocess.run(fetch_command(site, spec["fetch"], Path(args.data)), check=True, cwd=ROOT)
        pairs = pair_images(Path(args.data) / key)
        if len(pairs) != 1:
            raise SystemExit(f"expected one before/after pair in {Path(args.data) / key}; run with --fetch")
        before_path, after_path = pairs[0]
        out = Path(args.outputs) / f"{args.tag}{key}"
        out.mkdir(parents=True, exist_ok=True)

        predict_args = ["--before", str(before_path), "--after", str(after_path), "--checkpoint", str(checkpoint),
                        "--output", str(out / "change.tif"), "--threshold", str(thresholds["model"]), "--device", "cpu",
                        "--tile-size", str(score_cfg.get("tile_size", 512)), "--overlap", str(score_cfg.get("overlap", 32))]
        for option in ("scale_weights",):
            if option in score_cfg:
                predict_args += [f"--{option.replace('_', '-')}", *[str(v) for v in score_cfg[option]]]
        for option in ("fusion", "stretch"):
            if option in score_cfg:
                predict_args += [f"--{option}", str(score_cfg[option])]
        predict.main(predict_args)
        model_score, profile = read_image(out / "change.tif")
        model_score = model_score[0]

        before, _ = read_image(before_path)
        after, _ = read_image(after_path)
        diff_score = baselines.shared_difference(before, after)
        masks = {"model": model_score >= thresholds["model"], "diff": diff_score >= thresholds["diff"]}
        if post is not None:
            masks = {name: post(mask) for name, mask in masks.items()}
            write_image(out / "change_mask.tif", masks["model"].astype(np.float32), profile)
        write_image(out / "diff.tif", diff_score, profile)
        write_image(out / "diff_mask.tif", masks["diff"].astype(np.float32), profile)

        nb, na = normalize_pair(before, after)
        before_rgb = np.clip(np.moveaxis(nb[[2, 1, 0]], 0, -1), 0, 1)
        after_rgb = np.clip(np.moveaxis(na[[2, 1, 0]], 0, -1), 0, 1)
        before_date, after_date = before_path.stem.split("_")[-1], after_path.stem.split("_")[-1]

        fig, axes = plt.subplots(1, 4, figsize=(16, 4.6), dpi=110)
        images = [before_rgb, after_rgb, overlay(after_rgb, masks["model"], MODEL_COLOUR), overlay(after_rgb, masks["diff"], DIFF_COLOUR)]
        titles = [f"Before ({before_date})", f"After ({after_date})",
                  f"Trained network: {masks['model'].mean():.0%} of the scene marked (orange)",
                  f"Colour difference: {masks['diff'].mean():.0%} of the scene marked (cyan)"]
        for ax, image, title in zip(axes, images, titles):
            ax.imshow(image, interpolation="nearest")
            ax.set_title(title, fontsize=9)
            ax.axis("off")
        fig.suptitle(f"{site['name']}. {site['size_km']} km across. Windows {site['before']} and {site['after']}. "
                     "Contains modified Copernicus Sentinel data.", fontsize=9)
        fig.tight_layout()
        fig.savefig(figures / f"case_{args.tag}{key}.jpg", pil_kwargs={"quality": 88})
        plt.close(fig)
        plt.imsave(figures / f"case_{args.tag}{key}_before.jpg", before_rgb, pil_kwargs={"quality": 85})
        plt.imsave(figures / f"case_{args.tag}{key}_after.jpg", after_rgb, pil_kwargs={"quality": 85})
        save_rgba(figures / f"case_{args.tag}{key}_model.png", masks["model"], MODEL_COLOUR)
        save_rgba(figures / f"case_{args.tag}{key}_diff.png", masks["diff"], DIFF_COLOUR)

        both = masks["model"] & masks["diff"]
        either = masks["model"] | masks["diff"]
        metadata = json.loads((Path(args.data) / key / "metadata.json").read_text())
        report["sites"].append({
            "key": key,
            "name": site["name"],
            "height": int(before.shape[1]),
            "width": int(before.shape[2]),
            "before_date": before_date,
            "after_date": after_date,
            "acquisitions": metadata.get("acquisitions"),
            "fetch_command": " ".join(["python", "scripts/fetch_sentinel2.py", *fetch_command(site, spec["fetch"], Path("data/real"))[2:]]),
            "model_marked_fraction": float(masks["model"].mean()),
            "diff_marked_fraction": float(masks["diff"].mean()),
            "marked_by_both_fraction": float(both.mean()),
            "agreement_iou": float(both.sum() / max(either.sum(), 1)),
            "model_otsu_threshold": float(otsu_threshold(model_score)),
            "diff_otsu_threshold": float(otsu_threshold(diff_score)),
            "model_score_median": float(np.median(model_score)),
            "diff_score_median": float(np.median(diff_score)),
        })
        print(f"{key}: network marked {masks['model'].mean():.1%}, colour difference marked {masks['diff'].mean():.1%}, "
              f"overlap IoU {report['sites'][-1]['agreement_iou']:.2f}")

    Path(args.report).write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {args.report}")


if __name__ == "__main__":
    main()
