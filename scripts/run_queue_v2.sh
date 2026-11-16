#!/usr/bin/env bash
# Train a list of "<config name>:<seed>" jobs one after another, resuming any that were interrupted.
# Safe to run again after a crash or restart: finished runs are skipped, and a resumed run gives the
# same weights as an uninterrupted one.
#
#   bash scripts/run_queue_v2.sh pair_dense_t01:43 pair_dense_t01:44
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p outputs/v2
for job in "$@"; do
  name="${job%%:*}"
  seed="${job##*:}"
  python scripts/train_v2.py --config "configs/v2/${name}.json" --seed "${seed}" --resume \
    >> "outputs/v2/train_${name}_seed${seed}.log" 2>&1
done
