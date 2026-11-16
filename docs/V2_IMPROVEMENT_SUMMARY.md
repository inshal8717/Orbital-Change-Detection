# v2 model improvement: summary

The final report of the loop in [`V2_MODEL_IMPROVEMENT.md`](V2_MODEL_IMPROVEMENT.md). The experiment-by-experiment
log is [`V2_EXPERIMENTS.md`](V2_EXPERIMENTS.md). Full test tables are in [`V2_RESULTS.md`](V2_RESULTS.md), section 6.

## The result in four lines

- **Validation (used to choose):** F1 rose from 0.083 to 0.292. The colour difference scores 0.224.
- **Test, the measure fixed in advance (all pixels pooled):** F1 rose from 0.127 to 0.185. The colour difference
  scores 0.270. **The improved model does not beat the baseline on this measure.**
- **Test, averaged per scene:** F1 0.273 against the colour difference's 0.241. The improved model wins on 5 or 6 of
  10 scenes.
- **Why they differ:** the model is strong on urban scenes and fails on three scenes with farmland, where crops
  changing with the season are marked as change. The validation scenes contain no farmland, so the loop could not see
  this.

## Starting and final scores

Primary metric: F1 over all validation pixels, each scene cut at a threshold fitted on the other four scenes. Three
seeds; ± is the standard deviation over seeds, which is the noise estimate.

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

Rows marked "reference" are not experiments. They are the methods the network is compared with.

## Kept changes, in order

| # | Change | Validation gain | Kind of gain | Config (`configs/v2/`) |
| --- | --- | ---: | --- | --- |
| 1 | Threshold fitted on validation scenes instead of per-scene Otsu | +0.063 | Threshold only. The score maps are identical | `baseline.json` |
| 2 | Drop the coarsest feature scale from the score | +0.057 | Model (scoring). No weights changed | `scale_fine_mid.json` |
| 3 | Dense temporal loss on before/after pairs of the train scenes | +0.073 | Model (training) | `pair_dense_fine_mid.json` |
| 6 | Contrastive temperature 0.1 instead of 0.2 | +0.017 | Model (training) | `pair_dense_t01.json` |

Gains are differences in the three-seed mean of the primary metric, taken from
[`data/v2_experiments/index.json`](data/v2_experiments/index.json). The numbering follows the experiment log. Every
experiment's hypothesis was committed before its result; `git log -- docs/V2_EXPERIMENTS.md` shows the order.

**Threshold gains against model gains.** Of the total validation gain of 0.209, 0.063 came from the threshold rule
and 0.147 from changes to the score map (0.057 from scoring, 0.090 from training). The threshold-free metrics confirm
the model part: validation PR-AUC went from 0.115 to 0.240 and AUROC from 0.715 to 0.839.

## Discarded changes

| # | Change | Result | Why discarded |
| --- | --- | --- | --- |
| 4 | Put the coarse scale back, with the pair-trained weights | -0.010 and -0.005 | Never better, and noisier between seeds |
| 5 | Remove marked groups smaller than 8 pixels | +0.001 | Below the noise level (0.003) for the network |

Experiment 5 was the close one in a different sense: it did nothing for the network but lifted the raw colour
difference from 0.268 to 0.320 on validation, above the best network. That stronger reference is reported beside the
others.

## Test result

Scored once, with `configs/v2/pair_dense_t01.json` frozen in commit `525e807`.

<!-- v2:test_combined file=v2_results_final.json base=v2_results.json -->
| Method | AUROC | PR-AUC | F1 | IoU | Precision | Recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v1 network, as it was (Part A) | 0.628 ± 0.005 | 0.075 ± 0.002 | 0.127 ± 0.003 | 0.068 ± 0.002 | 0.079 ± 0.003 | 0.330 ± 0.062 |
| Improved network (after the loop) | 0.785 ± 0.007 | 0.111 ± 0.004 | 0.185 ± 0.004 | 0.102 ± 0.002 | 0.125 ± 0.004 | 0.355 ± 0.002 |
| Improved scoring, untrained network | 0.694 ± 0.003 | 0.085 ± 0.003 | 0.155 ± 0.004 | 0.084 ± 0.002 | 0.092 ± 0.003 | 0.503 ± 0.021 |
| Colour difference, shared stretch | 0.738 | 0.214 | 0.270 | 0.156 | 0.261 | 0.278 |
| Colour difference, raw values | 0.760 | 0.291 | 0.330 | 0.197 | 0.442 | 0.263 |
<!-- /v2 -->

Does the model now beat the colour-difference baseline on the test scenes?

- **On pooled F1, no.** 0.185 against 0.270 (shared stretch), a gap of 0.085. Against the raw colour difference the
  gap is 0.145. With the 8-pixel filter the references are 0.275 and 0.331.
- **On per-scene mean F1, yes.** 0.273 against 0.241, a margin of 0.032. The spread of the per-scene difference is
  0.07 to 0.08, so the margin is within the scene-to-scene variation.
- **Against the v1 network, yes, clearly.** +0.058 pooled F1 and +0.157 AUROC, stable across seeds.
- **Against random weights, probably.** +0.029 pooled F1 and +0.070 per-scene F1 on average. In Part A the trained
  network was worse than the untrained one, so the pair loss is what makes pretraining useful.

## What is still unknown

- **Whether the farmland failure can be trained away.** The pair loss teaches the network that same place means same
  features. Nine train scenes may simply hold too few fields at different seasons. More unlabelled pairs would test
  this, and they need no labels.
- **Whether the validation choices would survive a better validation set.** Five urban scenes chose this model.
  A validation set with farmland might have chosen differently, for example a different threshold or fusion.
  Re-drawing the split now would use knowledge of the test scenes, so it would need a fresh test set.
- **How much is the small model.** One encoder size, 320 training steps, CPU only. Section F of the loop (width,
  learned head, larger backbone) was not tried.
- **The untested hypotheses.** Sections C and D, variants of E, and most of G. They are listed in the experiment log
  under "Why the loop stopped". The loop ended for lack of session time, not on one of its stopping rules.
- **Transfer to Level-2A imagery.** The UAE case studies run this model on surface-reflectance data it was not
  trained on. They look reasonable but are not scored.

## Notes on test-set hygiene

- The Part A test results of the v1 baseline were known while the loop ran, because the plan orders Part A first.
  No candidate was scored on test and no decision used a test number.
- Three test passes exist, all in [`data/v2_test_ledger.json`](data/v2_test_ledger.json): the v1 baseline, the final
  configuration, and the final configuration with the 8-pixel filter. The third was run after the second, to report
  the filtered colour-difference references. It did not change any choice.
