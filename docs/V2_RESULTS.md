# v2 results on real data

**Short answer.** On real Sentinel-2 scenes the v1 network does not beat a plain colour difference. It is clearly
worse, and pretraining does not help it: on the test scenes the same network with random weights scores slightly
higher. Section 6 below reports what changed after the model-improvement loop.

Every number in a table on this page is generated from a result file in `docs/data/` by
`scripts/render_v2_tables.py`. A test fails if a table and its file disagree.

## 1. What was tested

- **Data.** OSCD: 24 registered Sentinel-2 pairs with hand-drawn masks of urban change. Source, licence and bands are
  in [`DATA_SOURCES.md`](DATA_SOURCES.md).
- **Split, by scene** ([`data/splits_v2.json`](data/splits_v2.json)). 9 train scenes for pretraining (imagery only),
  5 validation scenes for choosing thresholds, and OSCD's 10 official test scenes.
- **Methods.** The v1 pipeline unchanged (`configs/v2/baseline.json`): a small CNN pretrained with a contrastive
  loss on single images, then cosine distance between before and after features at three scales. It is compared with
  the same network with random weights, and with two colour differences that use no network.
- **Seeds.** Three for each network (42, 43, 44). Tables show the mean ± standard deviation over seeds. The colour
  differences have no randomness.
- **Test passes.** The test scenes were scored once for this configuration. Every test pass is recorded in
  [`data/v2_test_ledger.json`](data/v2_test_ledger.json), and the script refuses a second pass without a stated reason.

How to read the columns: AUROC and PR-AUC measure how well a method ranks changed pixels above unchanged ones, before
any cut-off. PR-AUC is the stricter of the two when changed pixels are rare, as here (about 5% of test pixels). F1,
IoU, precision and recall score the final changed/unchanged map. All pixels of all test scenes are counted together
unless a table says otherwise.

## 2. Test results with a threshold chosen on validation scenes

One threshold per method, fitted on the five validation scenes, then applied unchanged to the ten test scenes.

<!-- v2:test_main file=v2_results.json -->
| Method | AUROC | PR-AUC | F1 | IoU | Precision | Recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Trained network | 0.628 ± 0.005 | 0.075 ± 0.002 | 0.127 ± 0.003 | 0.068 ± 0.002 | 0.079 ± 0.003 | 0.330 ± 0.062 |
| Same network, untrained (random weights) | 0.687 ± 0.008 | 0.081 ± 0.003 | 0.152 ± 0.003 | 0.082 ± 0.002 | 0.090 ± 0.003 | 0.488 ± 0.005 |
| Colour difference, shared stretch | 0.738 | 0.214 | 0.270 | 0.156 | 0.261 | 0.278 |
| Colour difference, raw values | 0.760 | 0.291 | 0.330 | 0.197 | 0.442 | 0.263 |
<!-- /v2 -->

The same results averaged per scene instead of per pixel (each scene counts once, whatever its size):

<!-- v2:test_scene_mean file=v2_results.json -->
| Method | AUROC | PR-AUC | F1 | IoU | Precision | Recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Trained network | 0.652 ± 0.006 | 0.176 ± 0.011 | 0.160 ± 0.010 | 0.096 ± 0.006 | 0.216 ± 0.027 | 0.299 ± 0.062 |
| Same network, untrained (random weights) | 0.719 ± 0.008 | 0.213 ± 0.021 | 0.208 ± 0.018 | 0.127 ± 0.013 | 0.162 ± 0.018 | 0.511 ± 0.027 |
| Colour difference, shared stretch | 0.740 | 0.254 | 0.241 | 0.150 | 0.378 | 0.255 |
| Colour difference, raw values | 0.735 | 0.254 | 0.220 | 0.137 | 0.398 | 0.204 |
<!-- /v2 -->

![Precision-recall curves on the test scenes](img/v2/pr_curves_test.png)

*Precision-recall curves, all test pixels pooled. Higher is better. Faint lines are the other two seeds.*

## 3. How the threshold was chosen

**Rule.** For each method, take the single threshold that gives the highest F1 over all validation pixels pooled.
Apply that one number to every test scene. Nothing is fitted on a test scene.

<!-- v2:val_thresholds file=v2_results.json -->
| Method | Threshold fitted on validation | Validation F1 at that threshold | Validation F1, leave one scene out | Validation F1, per-scene Otsu |
| --- | ---: | ---: | ---: | ---: |
| Trained network (seed 42) | 0.5656 | 0.169 | 0.135 | 0.083 |
| Trained network (seed 43) | 0.6928 | 0.187 | 0.135 | 0.080 |
| Trained network (seed 44) | 0.6689 | 0.199 | 0.165 | 0.085 |
| Same network, untrained (random weights) (seed 42) | 0.1466 | 0.099 | 0.098 | 0.083 |
| Same network, untrained (random weights) (seed 43) | 0.1009 | 0.098 | 0.091 | 0.076 |
| Same network, untrained (random weights) (seed 44) | 0.08712 | 0.095 | 0.085 | 0.077 |
| Colour difference, shared stretch | 0.8069 | 0.226 | 0.224 | 0.137 |
| Colour difference, raw values | 1885 | 0.266 | 0.268 | 0.146 |
<!-- /v2 -->

"Leave one scene out" cuts each validation scene with a threshold fitted on the other four. It is the fairer
estimate of what the threshold will do on a new scene. For the network it is well below the F1 at the fitted
threshold, which means the network's scores sit on a different scale from scene to scene and one global threshold
does not transfer well. The colour differences barely move.

**Compared with v1's rule (per-scene Otsu, no labels), on the test scenes:**

<!-- v2:test_otsu file=v2_results.json -->
| Method | AUROC | PR-AUC | F1 | IoU | Precision | Recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Trained network | 0.628 ± 0.005 | 0.075 ± 0.002 | 0.155 ± 0.001 | 0.084 ± 0.001 | 0.087 ± 0.001 | 0.698 ± 0.012 |
| Same network, untrained (random weights) | 0.687 ± 0.008 | 0.081 ± 0.003 | 0.247 ± 0.015 | 0.141 ± 0.010 | 0.154 ± 0.012 | 0.628 ± 0.039 |
| Colour difference, shared stretch | 0.738 | 0.214 | 0.237 | 0.134 | 0.150 | 0.564 |
| Colour difference, raw values | 0.760 | 0.291 | 0.233 | 0.132 | 0.147 | 0.566 |
<!-- /v2 -->

The validation threshold helps the colour differences (F1 0.237 to 0.270 and 0.233 to 0.330). It does not help the
networks: the trained network drops from 0.155 to 0.127 and the untrained one from 0.247 to 0.152. Otsu adapts to
each scene, which hides the scale drift; a single global threshold exposes it.

## 4. Network against the baselines, scene by scene

<!-- v2:paired file=v2_results.json split=test -->
| Comparison | Pooled F1 difference | 95% interval (scenes resampled) | Per-scene F1 difference, mean ± spread | Scenes the trained network wins |
| --- | ---: | ---: | ---: | ---: |
| Trained (seed 42) minus colour difference (shared stretch) | -0.140 | -0.197 to -0.039 | -0.069 ± 0.061 | 1 of 10 |
| Trained (seed 42) minus untrained network (same seed) | -0.023 | -0.096 to -0.007 | -0.059 ± 0.065 | 1 of 10 |
| Trained (seed 43) minus colour difference (shared stretch) | -0.141 | -0.204 to -0.024 | -0.082 ± 0.093 | 1 of 10 |
| Trained (seed 43) minus untrained network (same seed) | -0.028 | -0.075 to -0.002 | -0.045 ± 0.054 | 1 of 10 |
| Trained (seed 44) minus colour difference (shared stretch) | -0.147 | -0.211 to -0.036 | -0.093 ± 0.078 | 1 of 10 |
| Trained (seed 44) minus untrained network (same seed) | -0.025 | -0.082 to +0.010 | -0.040 ± 0.064 | 2 of 10 |
<!-- /v2 -->

The interval comes from resampling whole scenes 2,000 times, so it reflects how much the answer depends on which ten
scenes happen to be in the test set.

Per-scene F1 at the validation threshold and per-scene AUROC (networks: seed 42):

<!-- v2:per_scene file=v2_results.json split=test seed=seed42 -->
| Scene | Trained F1 | Trained AUROC | Untrained F1 | Untrained AUROC | Diff shared F1 | Diff shared AUROC | Diff raw F1 | Diff raw AUROC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| brasilia | 0.118 | 0.772 | 0.308 | 0.889 | 0.208 | 0.708 | 0.186 | 0.709 |
| chongqing | 0.207 | 0.731 | 0.326 | 0.830 | 0.215 | 0.756 | 0.080 | 0.774 |
| dubai | 0.134 | 0.806 | 0.180 | 0.806 | 0.270 | 0.788 | 0.290 | 0.788 |
| lasvegas | 0.495 | 0.855 | 0.514 | 0.895 | 0.598 | 0.863 | 0.621 | 0.873 |
| milano | 0.073 | 0.897 | 0.174 | 0.951 | 0.268 | 0.870 | 0.180 | 0.865 |
| montpellier | 0.417 | 0.853 | 0.533 | 0.947 | 0.497 | 0.843 | 0.473 | 0.830 |
| norcia | 0.012 | 0.387 | 0.022 | 0.497 | 0.061 | 0.661 | 0.060 | 0.641 |
| rio | 0.250 | 0.779 | 0.229 | 0.792 | 0.241 | 0.782 | 0.244 | 0.766 |
| saclay_w | 0.009 | 0.212 | 0.019 | 0.286 | 0.031 | 0.543 | 0.032 | 0.501 |
| valencia | 0.005 | 0.260 | 0.007 | 0.403 | 0.021 | 0.582 | 0.030 | 0.604 |
<!-- /v2 -->

## 5. Conclusions

**Does the network beat the colour-difference baseline on real data? No.** Pooled test F1 is 0.127 for the trained
network against 0.270 for the shared-stretch colour difference and 0.330 for the raw one. The gap to the shared-stretch
baseline is about 0.14 F1 for every seed, and the scene-resampled 95% interval stays below zero for every seed. Scene
by scene the picture is noisier: the trained network loses on 9 of 10 scenes, by 0.07 to 0.09 F1 on average, and the
spread of that difference across scenes is about as large as the difference itself (0.06 to 0.09). So the direction is
consistent, but ten scenes cannot pin the size of the gap down tightly. The ranking metrics agree: PR-AUC 0.075
against 0.214.

**Does pretraining help compared with random weights? No, not on this evidence.** On the test scenes the untrained
network is slightly better (F1 0.152 against 0.127, AUROC 0.687 against 0.628), and the trained network wins on only
1 or 2 of 10 scenes. On the validation scenes it was the other way round (leave-one-scene-out F1 0.145 against 0.092).
A result that flips between two sets of scenes is not evidence of a benefit.

**Where does it fail?** Three patterns, with examples from the validation scenes:

1. **Season and sun angle are read as change.** In `bercy` the first image is from late November (low sun, long
   shadows, bluish) and the second from late August. The network marks most of the city. The colour difference is
   noisy here too but marks far less.

   ![bercy, validation scene](img/v2/baseline_val_bercy.png)

2. **The score scale drifts between scenes, so one threshold misfires.** In `abudhabi` the threshold fitted across
   all validation scenes is too high for this scene, and the network misses most of the real construction (blue).

   ![abudhabi, validation scene](img/v2/baseline_val_abudhabi.png)

3. **Clouds and their shadows are marked as change.** In `paris` the second image contains two clouds and a cloud
   shadow. Both methods mark them. The network marks them as large solid blobs, which swamp the 0.3% of pixels that
   really changed.

   ![paris, validation scene](img/v2/baseline_val_paris.png)

On the test scenes the same weakness shows in the numbers: for `norcia`, `saclay_w` and `valencia` the trained
network's AUROC is between 0.20 and 0.39, which is worse than guessing. Its scores are highest where nothing in the mask
changed.

The failure panels are taken from validation scenes on purpose. The improvement loop is allowed to look at them, and
it must not look at test scenes.

**What this does not show.** OSCD labels only urban change, so a method that reacts to real but unlabelled change
(crops, water level) is penalised for it. The test set is ten scenes. The network is small and was trained for a few
minutes on a CPU on nine scenes. A larger or longer-trained model could behave differently.

## Reproduce

```bash
pip install -e ".[geo,test,figures]"
python scripts/fetch_benchmark.py                                   # download OSCD, about 513 MB
for seed in 42 43 44; do python scripts/train_v2.py --config configs/v2/baseline.json --seed $seed; done
python scripts/evaluate_v2.py --config configs/v2/baseline.json --seeds 42 43 44               # validation only
python scripts/evaluate_v2.py --config configs/v2/baseline.json --seeds 42 43 44 --split test \
    --rerun-reason "reproducing the published numbers"
python scripts/make_v2_figures.py pr
python scripts/make_v2_figures.py panels --config configs/v2/baseline.json \
    --results docs/data/v2_experiments/baseline.json --split val --prefix baseline_
python scripts/render_v2_tables.py
```

Training takes about 8 minutes per seed on a 2-core CPU. Checkpoints go to `outputs/v2/runs/`, which is not in git.
