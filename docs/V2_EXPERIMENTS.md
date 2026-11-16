# v2 experiment log

The model-improvement loop from [`V2_MODEL_IMPROVEMENT.md`](V2_MODEL_IMPROVEMENT.md). Every entry was written in two
parts: the hypothesis **before** the run, and the result after it. Only validation scenes guide decisions.

## Protocol

- **Data.** OSCD, split by scene ([`data/splits_v2.json`](data/splits_v2.json)). Pretraining uses the 9 train scenes
  (imagery only). Decisions use the 5 validation scenes: abudhabi, aguasclaras, beirut, bercy, paris.
- **Primary metric.** F1 of the binary change map over all validation pixels pooled, where each validation scene is
  cut at a single global threshold fitted on the **other four** validation scenes (leave one scene out). This is
  "threshold chosen on validation" in a form that cannot fit a scene to itself, so it also punishes score maps whose
  scale drifts from scene to scene. The table calls it "Validation F1 (leave one scene out)".
- **Secondary metrics.** PR-AUC and AUROC of the pooled score map, which do not depend on any threshold.
- **Seeds.** 42, 43 and 44. Training changes are run with seed 42 first; a change that looks better is re-run with
  seeds 43 and 44 before it can be kept. Scoring-only changes are evaluated on all three seeds straight away.
- **Noise level.** The standard deviation of the primary metric over the three seeds of the current best
  configuration. For the baseline that is 0.014 (seeds give 0.135, 0.135 and 0.165).
- **Decision rule.** Keep a change only if the three-seed mean of the primary metric improves by more than **twice**
  the noise level of the current best. Otherwise discard it.
- **Baselines to beat.** Colour difference with a shared stretch scores 0.224 on the primary metric, and colour
  difference on raw values scores 0.268. The untrained network scores 0.092 ± 0.005.
- **Budget.** One training run is 320 steps of batch 64 on CPU (about 8 minutes; about 15 with the pair loss).
- **Reproduce any row.** `python scripts/train_v2.py --config configs/v2/<name>.json --seed <seed>`, then
  `python scripts/evaluate_v2.py --config configs/v2/<name>.json --seeds 42 43 44`. Results are written to
  `docs/data/v2_experiments/<name>.json`.

**Test-set exposure.** The plan orders the Part A test pass before this loop, so the baseline's test results were
known while the loop ran. No test score was computed for any candidate below, and no decision used a test number.
`docs/data/v2_test_ledger.json` lists every test pass.

## Summary

<!-- v2:experiments file=v2_experiments/index.json -->
<!-- /v2 -->

## Entries

### Experiment 0: baseline (the v1 method on real data)

- **Config.** `configs/v2/baseline.json`. SimCLR pretraining on single images, three feature scales averaged, shared
  stretch.
- **Result.** Primary metric 0.145 ± 0.014 over three seeds. PR-AUC 0.115 ± 0.008. This is the starting point.

### Experiment 1: fit the threshold on validation scenes instead of per-scene Otsu

- **Section.** A (decision threshold). Scoring only, no retraining.
- **Hypothesis (written before the run).** v1 cuts every scene with Otsu, which assumes two clear groups of scores.
  Real scenes have 0.3% to 4% changed pixels, so Otsu will split the unchanged pixels and mark far too many. A single
  threshold fitted on labelled validation scenes should raise precision a lot and raise F1. I would discard it if
  pooled validation F1 does not improve.
- **What kind of gain this is.** A threshold gain. The score maps are identical.

### Experiment 2: drop the coarsest feature scale from the score

- **Section.** B (score fusion across scales). Scoring only, no retraining.
- **Hypothesis (written before the run).** A per-scale check of the seed-42 baseline on validation showed the coarse
  stage (1/8 resolution) ranks pixels worst (PR-AUC 0.056, against 0.096 and 0.129 for the fine and middle stages).
  OSCD changes are small buildings and roads, a few pixels wide, and a 1/8-resolution map blurs them. Averaging only
  the fine and middle stages (weights 1, 1, 0) should raise the primary metric. I would discard it if the three-seed
  mean does not beat the baseline by more than 0.028.

### Experiment 3: add a dense temporal loss on before/after pairs of the train scenes

- **Section.** E (pretraining on pairs). Needs retraining.
- **Hypothesis (written before the run).** The v1 objective only ever sees one image at a time, so nothing teaches
  the features to ignore what differs between two dates for reasons other than change: season, sun angle, haze,
  crops. The failure panels show exactly that, with fields and bright soil marked as change. Adding a pixel-level
  contrastive loss between co-located patches of the two dates (same position is the positive, other positions are
  negatives) trains the very features that are compared at scoring time to agree where the ground is the same. It uses
  no mask; it only assumes most ground does not change. I expect PR-AUC and the primary metric to rise clearly. I
  would discard it if the seed-42 primary metric is not above the baseline's three-seed mean by more than 0.028.
- **Config.** `configs/v2/pair_dense.json`: baseline plus `pair_weight 1.0`, `pair_points 16`. One change: the added
  loss term. The single-image loss is kept as it was.
