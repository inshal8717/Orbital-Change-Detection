"""Pretrain one v2 configuration on the train split of the benchmark.

Reads the ``train`` section of a config in configs/v2/ and runs
``sat_change.train`` on ``<data>/train`` only. The checkpoint goes to
``outputs/v2/runs/<config name>/seed<seed>/``. Labels are not used: train
scenes have no masks on disk.

Usage:
    python scripts/train_v2.py --config configs/v2/baseline.json --seed 42
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from sat_change import train

ROOT = Path(__file__).resolve().parent.parent


def run_dir(config: dict, seed: int, runs: Path) -> Path:
    return runs / config["name"] / f"seed{seed}"


def train_args(config: dict, seed: int, data: Path, out: Path) -> list[str]:
    args = ["--data", str(data / "train"), "--output", str(out), "--seed", str(seed), "--device", "cpu"]
    for key, value in config["train"].items():
        args += [f"--{key.replace('_', '-')}", str(value)]
    return args


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data", default=str(ROOT / "data" / "benchmark"))
    p.add_argument("--runs", default=str(ROOT / "outputs" / "v2" / "runs"))
    args = p.parse_args(argv)

    config = json.loads(Path(args.config).read_text())
    out = run_dir(config, args.seed, Path(args.runs))
    train.main(train_args(config, args.seed, Path(args.data), out))
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    (out / "run.json").write_text(json.dumps({"config": config, "seed": args.seed, "commit": commit}, indent=2) + "\n")


if __name__ == "__main__":
    main()
