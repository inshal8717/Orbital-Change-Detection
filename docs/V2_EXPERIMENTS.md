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
| # | Single change | Kind | Seeds | Validation F1 (leave one scene out) | Change vs best so far | Validation PR-AUC | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ref-a | Reference: colour difference, shared stretch (no network) | reference | none | 0.224 | – | 0.141 | to beat |
| ref-b | Reference: colour difference, raw values (no network) | reference | none | 0.268 | – | 0.181 | reference |
| ref-c | Reference: v1 network with random weights | reference | 42, 43, 44 | 0.092 ± 0.005 | – | 0.045 ± 0.001 | reference |
| 0 | v1 as it was: per-scene Otsu threshold | start | 42, 43, 44 | 0.083 ± 0.002 | – | 0.115 ± 0.008 | start |
| 1 | Threshold fitted on validation scenes instead of Otsu | threshold | 42, 43, 44 | 0.145 ± 0.014 | +0.063 | 0.115 ± 0.008 | kept |
| 2 | Drop the coarsest feature scale from the score | model (scoring) | 42, 43, 44 | 0.202 ± 0.004 | +0.057 | 0.137 ± 0.008 | kept |
| 3 | Add a dense temporal loss on before/after pairs of the train scenes | model (training) | 42, 43, 44 | 0.275 ± 0.001 | +0.073 | 0.201 ± 0.009 | kept |
| 4a | Put the coarse scale back (weights of experiment 3) | model (scoring) | 42, 43, 44 | 0.265 ± 0.006 | -0.010 | 0.192 ± 0.003 | discarded |
| 6 | Contrastive temperature 0.1 instead of 0.2 | model (training) | 42, 43, 44 | 0.292 ± 0.003 | +0.017 | 0.240 ± 0.009 | kept, final |
| 4b | Put the coarse scale back (weights of experiment 6) | model (scoring) | 42, 43, 44 | 0.287 ± 0.012 | -0.005 | 0.237 ± 0.007 | discarded |
| 5 | Remove marked groups smaller than 8 pixels | post-processing | 42, 43, 44 | 0.294 ± 0.003 | +0.001 | 0.240 ± 0.009 | discarded |
| ref-d | Reference: colour difference, shared stretch, with the same 8-pixel filter | reference | none | 0.264 | +0.040 | 0.141 | reference |
| ref-e | Reference: colour difference, raw values, with the same 8-pixel filter | reference | none | 0.320 | +0.052 | 0.181 | strongest reference |
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
- **Result.** Pooled validation F1 rose from 0.083 ± 0.002 (Otsu) to 0.145 ± 0.014 (leave one scene out), for
  every seed. Precision is what moved: Otsu marked between 10% and 62% of each validation scene, where 0.3% to 4%
  really changed. The same switch helps the colour
  difference more (0.137 to 0.224).
- **Decision: kept.** It is also the rule the plan requires (Step A4). It is a threshold gain, not a model gain.
- **Learned.** The gap between the F1 at the fitted threshold (0.185) and the leave-one-scene-out F1 (0.145) is
  large for the network and tiny for the colour difference. The network's scores are on a different scale in each
  scene.

### Experiment 2: drop the coarsest feature scale from the score

- **Section.** B (score fusion across scales). Scoring only, no retraining.
- **Hypothesis (written before the run).** A per-scale check of the seed-42 baseline on validation showed the coarse
  stage (1/8 resolution) ranks pixels worst (PR-AUC 0.056, against 0.096 and 0.129 for the fine and middle stages).
  OSCD changes are small buildings and roads, a few pixels wide, and a 1/8-resolution map blurs them. Averaging only
  the fine and middle stages (weights 1, 1, 0) should raise the primary metric. I would discard it if the three-seed
  mean does not beat the baseline by more than 0.028.
- **Config.** `configs/v2/scale_fine_mid.json`: the baseline checkpoints, scored with `scale_weights [1, 1, 0]`.
- **Result.** Primary metric 0.202 ± 0.004 against 0.145 ± 0.014, a gain of 0.057 and better for every seed (0.197,
  0.205, 0.205). PR-AUC rose from 0.115 to 0.137. The gap between fitted-threshold F1 and leave-one-scene-out F1
  shrank from 0.040 to 0.014, so the scores also became more comparable between scenes.
- **Decision: kept.** The gain is four times the baseline's noise level. This is a model gain in the sense that the
  score map changed; no weights changed.
- **Learned.** The coarse stage was adding scene-wide blobs that differ in strength from scene to scene. The network
  is still below both colour differences (0.224 and 0.268). New noise level: 0.004.

### Experiment 3: add a dense temporal loss on before/after pairs of the train scenes

- **Section.** E (pretraining on pairs). Needs retraining.
- **Hypothesis (written before the run).** The v1 objective only ever sees one image at a time, so nothing teaches
  the features to ignore what differs between two dates for reasons other than change: season, sun angle, haze,
  crops. The validation failure panels show exactly that: in `bercy` (late November against late August) the
  network marks most of the city, and in `paris` it marks clouds and a cloud shadow as large blobs. Adding a pixel-level
  contrastive loss between co-located patches of the two dates (same position is the positive, other positions are
  negatives) trains the very features that are compared at scoring time to agree where the ground is the same. It uses
  no mask; it only assumes most ground does not change. I expect PR-AUC and the primary metric to rise clearly. I
  would discard it if the seed-42 primary metric is not above the baseline's three-seed mean by more than 0.028.
- **Note.** The sentence about the failure panels was corrected about ten minutes after this entry was first
  committed, once the panels had been rendered and looked at. The run had not finished and no result was known.
- **Config.** `configs/v2/pair_dense.json`: baseline plus `pair_weight 1.0`, `pair_points 16`. One change: the added
  loss term. The single-image loss is kept as it was.
- **Scoring.** `configs/v2/pair_dense_fine_mid.json`: the same checkpoints scored with the current best rule (fine
  and middle scales), so that the loss term is the only difference from Experiment 2.
- **Result.** Primary metric 0.275 ± 0.001 (seeds: 0.275, 0.276, 0.274) against 0.202 ± 0.004. PR-AUC rose from
  0.137 to 0.201 and AUROC from 0.725 to 0.790.
- **Decision: kept.** The gain of 0.073 is far above the noise level. This is a model gain: the weights changed and
  the ranking metrics, which use no threshold, moved with the F1.
- **Learned.** The network now passes the shared-stretch colour difference on the primary metric (0.224) and is level
  with the raw one (0.268). The pooled number leans on one scene: `beirut` holds about half of all validation pixels
  and has by far the highest F1 (0.45). `bercy` (winter against summer) is still a failure, with F1 0.01. Averaged per
  scene instead of per pixel, the network scores 0.143, still below the colour difference's 0.153. New noise level:
  0.001.

### Experiment 4: with the pair-trained weights, put the coarse scale back

- **Section.** B (score fusion), re-checked after Experiment 3. Scoring only.
- **Why it exists.** `configs/v2/pair_dense.json` was written with the default three-scale average, so its first
  validation score was computed with the coarse scale included. That number (seed 42) was seen before this entry was
  written. It is logged as a control, to check that Experiment 2's decision still holds for the new weights.
- **Hypothesis.** The coarse stage is now trained with the dense loss too, so it may no longer hurt. If the
  three-scale average beats the two-scale one by more than twice the noise level, Experiment 2 is reversed.
- **Result.** With Experiment 3's weights: 0.265 ± 0.006 for three scales against 0.275 ± 0.001 for two. Checked
  again later with Experiment 6's weights (`configs/v2/pair_dense_t01_three_scales.json`): 0.287 ± 0.012 against
  0.292 ± 0.003.
- **Decision: discarded, both times.** Three scales are never better, and they are noisier between seeds.
  Experiment 2's decision stands.

### Experiment 5: remove small marked groups from the mask

- **Section.** G (post-processing). Scoring only.
- **Hypothesis (written before the run).** OSCD's labelled changes are buildings and roads, mostly groups of ten or
  more pixels. Isolated marked pixels and tiny groups are more likely to be noise. Dropping marked groups smaller than
  8 pixels (4-connected) should raise precision at little cost in recall. The network's masks are already smooth,
  because features are compared at half and quarter resolution, so I expect a small effect for the network and a
  larger one for the colour difference. I would discard it if the gain is not above twice the noise level.
- **Config.** `configs/v2/pair_dense_t01_post8.json`: the best configuration at the time (Experiment 6) with the
  filter applied to every method's mask. It was run after Experiment 6.
- **Result.** Network: 0.294 ± 0.003 against 0.292 ± 0.003, a gain of 0.001. Colour difference, shared stretch: 0.224
  to 0.264. Colour difference, raw values: 0.268 to 0.320.
- **Decision: discarded for the network.** The gain is below the noise level, as predicted.
- **Learned, and it matters.** The same filter lifts the raw colour difference to 0.320, which is **above the best
  network (0.292)**. So on validation the network beats the colour difference as the plan defines it (shared stretch,
  no filter), but not the strongest simple method found here. Both filtered references are in the summary table and
  were also scored on the test scenes.

### Experiment 6: contrastive temperature 0.1 instead of 0.2

- **Section.** D (training objective). Needs retraining.
- **Hypothesis (written before the run).** The dense loss compares each position with about a thousand others, most
  of them easy negatives. A lower temperature puts more weight on the hard ones, which are positions that look alike.
  That should make features more specific to what is at a position and could sharpen the response to small changes.
  The risk is more false marks on texture. I expect a small gain or none. I would discard it if the seed-42 primary
  metric does not beat Experiment 3's three-seed mean by more than twice its noise level.
- **Config.** `configs/v2/pair_dense_t01.json`: Experiment 3's config with `temperature 0.1`. The one temperature
  value is used by both loss terms.
- **Result.** Primary metric 0.292 ± 0.003 (seeds: 0.291, 0.297, 0.288) against 0.275 ± 0.001. PR-AUC 0.240 against
  0.201, AUROC 0.839 against 0.790. Per-scene mean F1 rose from 0.143 to 0.176.
- **Decision: kept.** The gain of 0.017 is small but well above twice the noise level, every seed is above every
  seed of Experiment 3, and the threshold-free metrics rose more than the F1 did. This is the final configuration.
- **Learned.** The largest per-scene change was `abudhabi` (F1 0.10 to 0.21). `bercy` stayed near zero (0.02).

## Why the loop stopped

It did not stop on one of the stopping rules in `V2_MODEL_IMPROVEMENT.md`. It stopped because the session ran out of
time: each training run takes 15 minutes on this 2-core machine, a kept change needs three, and the environment
restarted twice and killed running jobs. Six experiments were completed.

The closest rule is the third one (the best model beats the colour-difference baseline on validation by more than
the noise level). That part holds: 0.292 against 0.224. But the remaining ideas do not all need a large budget, so the
rule is not met.

Not tried:

- **Section C.** A per-band stretch fixed on the training scenes.
- **Section D.** Haze, brightness-shift and misregistration augmentations; a larger batch.
- **Section E, variants.** The weight of the pair loss; training on pairs only, without the single-image loss.
- **Section F.** Encoder width, a learned head on the feature difference, a larger backbone.
- **Section G.** Hole filling and morphological opening. Only the small-group filter was tested.

## Final configuration

`configs/v2/pair_dense_t01.json`, frozen in commit `525e807` before any improved model was scored on the test scenes.
Test results and what they mean are in [`V2_RESULTS.md`](V2_RESULTS.md) (section 6) and
[`V2_IMPROVEMENT_SUMMARY.md`](V2_IMPROVEMENT_SUMMARY.md).
