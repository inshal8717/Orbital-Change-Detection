# Model improvement loop

This file tells an AI agent how to improve the change-detection model on the `v2` branch by itself. It is a loop:
form a hypothesis, change one thing, measure it on validation data, keep or discard, and log the result. Read
`docs/V2_PLAN.md` first. Its ground rules still apply.

## Goal

Raise the **validation** score of the changed-pixel map, without touching the test scenes.

- **Primary metric:** F1 of the final binary mask on the validation scenes, with the threshold chosen on validation.
- **Secondary metrics:** AUROC and average precision (PR-AUC) of the score map. Report them too. A change that raises
  F1 by moving the threshold alone is a threshold win, not a model win, so note which kind it is.
- **Baselines to beat on validation:** the colour difference with shared stretch, and the same network with random
  weights. A model change only counts if it beats the colour difference on the primary metric.

## Hard rules

1. **Never look at test scenes while improving.** Only `val` scores may guide a decision. Score `test` once at the
   end, with the final configuration fixed. If you ran test by accident, say so in the log and mark the result as
   contaminated.
2. **Change one thing per experiment.** Record the single change, its reason and its result.
3. **Keep every experiment reproducible.** Fix the seed, save the config, and save the git commit hash.
4. **Do not change the metric or the split to make a result look better.** If a metric is wrong, fix it in a separate,
   logged commit and re-score all earlier experiments.
5. **No new labels.** Use only the benchmark masks already in the `val` split, and only for scoring. Do not hand-label
   anything.
6. **CPU budget.** Each training run should finish within about 30 minutes on CPU, so the loop can run many times. If
   a change needs more, make it a separate, clearly labelled "large" experiment.
7. **Keep the code working.** `pytest` must pass after each kept change.

## The loop

For each iteration:

1. **Read the log.** Open `docs/V2_EXPERIMENTS.md` (create it if missing) and list what has been tried.
2. **Pick one hypothesis** from the list below, or propose a new one with a reason grounded in a failure you saw. Prefer
   the change most likely to help given the failure panels.
3. **Write the hypothesis first** in the log: what you expect, and what result would make you discard it.
4. **Implement the change** in a branch-local commit, with the config saved under `configs/v2/`.
5. **Train or score on `train` and `val` only.** Use `scripts/evaluate_v2.py` with `--split val`. Do not pass
   `--split test`.
6. **Compare** with the current best kept configuration on the primary metric. Use the same threshold rule, and also
   report the validation threshold value.
7. **Decide.** Keep the change if the primary metric improves by more than the noise level. Estimate noise by re-running
   the current best with two other seeds. If the gain is within the noise, discard the change.
8. **Log the result** in `docs/V2_EXPERIMENTS.md`: hypothesis, change, commit hash, seeds, metrics, decision, and
   one sentence on what you learned.
9. **Update the best** configuration if the change was kept, and run `pytest`.

Stop when one of the stopping rules below is met, not when you run out of ideas.

## Hypotheses to try

These come from the code and the README's own limits. Work through them roughly in this order, and add new ones as
you learn.

### A. The decision threshold (cheapest, often largest gain)

- Replace Otsu with a threshold chosen on validation scenes: maximise F1, or pick the value that gives a target
  precision. Keep Otsu as a reported baseline.
- Try a threshold per scene versus one global threshold. A global one is more honest for deployment; keep it unless the
  per-scene one generalises.
- Try a learned threshold: a logistic fit on validation scores, using a small number of parameters.

### B. Score fusion across scales

- The current score averages three feature scales equally (`src/sat_change/model.py`). Search over the scale weights on
  validation, using a coarse grid. Keep the weights simple, such as one number per scale.
- Try a geometric mean instead of an arithmetic mean.

### C. Normalisation

- Compare the shared stretch with a per-band stretch fixed on training scenes (not per image, and not per pair).
- Check that the Sentinel-2 integer scale (`0..10000`) and reflectance (`0..1`) paths give the same result. Add a test if
  not.

### D. Training objective and augmentation

- Temperature of the NT-Xent loss: try a few values, and keep the one that helps validation.
- Augmentations: add realistic ones for satellite imagery, such as random brightness shifts that mimic season and haze,
  small random shifts (misregistration), and a mild blur that mimics resampling. Remove any that hurt.
- Batch size: larger batches give more negatives. Try the largest that fits on CPU.

### E. Pretraining on pairs

- The README notes that pretraining sees single images, but scoring uses one shared stretch for both dates. Try
  fine-tuning on pairs from the `train` scenes, where a positive is the same location at two dates with no change, and
  the features should match. Only use `train` scenes for this.
- Check that this does not use any change mask. Unchanged pixels can be found without masks, from the same region at
  two dates. If in doubt, skip it and log why.

### F. Architecture (only after A–E)

- Encoder width: the current widths are `32 → 64 → 128 → 256`. Try one smaller and one larger setting.
- Add a light decoder or a small learned head on the feature difference, and compare it with plain cosine distance.
  Keep it only if it wins on validation.
- A larger pretrained backbone is allowed only as a separate "large" experiment, because of the CPU budget.

### G. Post-processing

- Remove small connected components below a size threshold, and check the effect on validation precision.
- Fill small holes in the mask.
- Apply a light morphological opening. Measure, do not assume.

## What to watch for

- **Threshold wins are not model wins.** If a change only moves F1 through the threshold, say so.
- **Synthetic scenes are too easy.** A colour difference scores perfectly there. Do not conclude anything from
  synthetic results on real data. Use the real `val` split from Part A.
- **Regional false positives.** In the Gulf, haze, dust, bright sand and seasonal vegetation can look like change. Keep
  the failure panels from `docs/img/v2/` in view when choosing a hypothesis, and check whether a change fixes them
  without breaking the others.
- **Overfitting to validation.** Each kept change is a small risk of overfitting. Keep the number of kept changes small,
  and report the final test result once.

## Stopping rules

Stop and write a summary when any of these is true:

- Two full passes through section A to G produce no kept change.
- The best validation F1 has not improved for 10 experiments in a row.
- The best model beats the colour-difference baseline on validation by more than the noise level, and the remaining
  ideas need the "large" budget.
- A step needs data or labels that are not allowed under the hard rules.

## Final report

When the loop stops:

1. Score the final configuration once on `test`, and record the result in `docs/V2_RESULTS.md`.
2. Write `docs/V2_IMPROVEMENT_SUMMARY.md` with:
   - The starting score and the final validation score, with noise estimates.
   - Every kept change, in order, with its gain and its commit hash.
   - Every discarded change that was close, and why.
   - Whether the model now beats the colour-difference baseline on the test scenes, and by how much.
   - What is still unknown.
3. Update the README's v2 section with the test result. Say which gains came from the threshold and which from the
   model.

Do not claim a result that is not in the logs.
