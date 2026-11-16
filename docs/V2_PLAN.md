# v2 plan: real-data test and a local (UAE) case study

This is the work order for the `v2` branch. Work through the steps in order. Do not start a step until the previous
step's **Done when** check passes. Commit after each step with a message that names the step (for example
`v2 step 3: validation-based threshold`).

## Ground rules

- Work only on the `v2` branch. Never commit to `main`.
- Do not change the meaning of the existing results table in `README.md`. Add new results beside it. Keep the
  "withdrawn" note on the AUROC 0.98 figure.
- Do not tune anything on the test scenes. Any threshold, normalisation choice or hyperparameter is chosen on a
  validation region, and the test region is scored once.
- Every number that goes into the README must come from a script in `scripts/` and an output file in `outputs/` or
  `docs/data/`. No hand-typed metrics.
- Keep the existing tests passing (`pytest`). Add tests for every new module.
- Record the seed, the scene keys, the date windows and the cloud limit for every real-data run.

---

## Part A: test on real Sentinel-2 data (idea 1)

**Goal:** show whether the method works on real imagery, and whether it beats a plain colour difference there.

### Step A1. Choose labelled ground truth

Find a public change-detection benchmark with real satellite pairs and binary change masks. Candidates to check first:
the OSCD (Onera Satellite Change Detection) dataset, and any other public Sentinel-2 change dataset with masks.

- Confirm the licence allows research use and redistribution of derived results. Write the licence and URL into
  `docs/DATA_SOURCES.md`.
- Confirm the bands are compatible with `B02, B03, B04, B08` at 10 m. If the benchmark uses different bands or a
  different resolution, resample or select bands and record exactly what was done.
- Do not commit raw dataset files to git. Add the download path to `.gitignore`.

**Done when:** `docs/DATA_SOURCES.md` lists the dataset, licence, bands, resolution and how to download it, and a
script `scripts/fetch_benchmark.py` (or a documented manual step) produces `data/benchmark/` in the layout described
in the README's "Data layout" section.

### Step A2. Load the benchmark into the existing pipeline

- Write a loader that maps the benchmark's pairs and masks into the `data/real/<scene>/images` and `masks` layout.
- Split scenes into three groups: `train` (for pretraining), `val` (thresholds, model selection) and `test` (scored
  once). Split by scene, never by patch. Save the split as `docs/data/splits_v2.json`.
- Pretraining must use only the train scenes, and only single images, as the README describes.

**Done when:** a test checks that no scene appears in two splits, and that no mask is read as imagery.

### Step A3. Run the baselines on the real test scenes

Compute, on the test scenes only, using the same threshold rule for each method:

1. Colour difference, shared stretch (the simplest baseline)
2. Colour difference, raw values
3. The trained network from pretraining on the train scenes
4. The same network with random weights

Report AUROC, precision-recall AUC, F1, IoU, precision and recall. Also report precision-recall curves, because changed
pixels are rare.

**Done when:** `scripts/evaluate_v2.py` writes `docs/data/v2_results.json`, and a table of all four methods is in
`docs/V2_RESULTS.md` with the split, seed and thresholds stated.

### Step A4. Choose the threshold on validation data

Otsu on each scene is the current rule, and the README shows it over-marks changed pixels. Replace it with a threshold
chosen on the `val` scenes, for example the value that maximises F1 on validation, or a value that targets a fixed
precision. Apply that single threshold to the test scenes. Also run Otsu for comparison.

**Done when:** the chosen rule, the validation score that picked it, and the test result are in `docs/V2_RESULTS.md`,
and the test scenes were scored only once.

### Step A5. Answer the main question honestly

Write a short conclusion in `docs/V2_RESULTS.md` that answers these questions with the numbers:

- Does the network beat the colour-difference baseline on real data? By how much, and is the gap bigger than the
  spread across scenes?
- Does pretraining help compared with random weights?
- Where does it fail? Save at least three example panels (before, after, truth, prediction) for failures, not only
  successes.

If the network does not beat the baseline, say so plainly. That is a valid result.

**Done when:** the conclusion is written, the failure panels are in `docs/img/v2/`, and a reviewer could reproduce
every number with one command.

---

## Part B: a local case study in the UAE (idea 4)

**Goal:** a real, explainable before-and-after example for a place the project's owner can describe, with a clear
change that is visible to a non-expert.

### Step B1. Pick two or three sites, and check each one

Choose sites where a real change happened between two dates that Sentinel-2 can see. Candidates to check, not to
assume:

- Palm Jumeirah and nearby reclamation, Dubai
- Dubai Creek Harbour or other large construction zones
- Coastal reclamation or new islands elsewhere in the UAE

For each candidate:

- Find an independent source that documents the change and its approximate date (news article, official project page,
  or a published satellite image). Record the URL and the date.
- Pick acquisition windows that bracket the change, and check cloud cover with `--cloud-max`.

**Done when:** `docs/CASE_STUDIES.md` lists at least two sites, each with the change, the source URL, the date window,
and the coordinates used for `--center`.

### Step B2. Fetch and co-register the imagery

Use the existing `scripts/fetch_sentinel2.py` with each site's coordinates. Keep `--size` small enough that the scene
is local (a few km). Commit no imagery to git; keep the output in `data/real/`.

**Done when:** each site produces a before and after `.tif` pair on one grid, and the script's output is recorded in
`docs/CASE_STUDIES.md`.

### Step B3. Run the model and the baseline

- Run the trained model (from Part A) and the colour-difference baseline on each site, using the validation-chosen
  threshold from Step A4.
- Produce a score map and a mask for each. Use `sat_change.predict` for the model.

**Done when:** each site has both outputs saved in `outputs/v2/<site>/`.

### Step B4. Make a visual that a non-expert can read

For each site, make one figure with four panels: before, after, the model's changed-pixel overlay, and the
colour-difference overlay. Add a slider version for the web page. Label the source of the known change.

**Done when:** the figures are in `docs/img/v2/` and each has a caption that states the date window and the source.

### Step B5. Write up the case studies

In `docs/CASE_STUDIES.md`, for each site, state what the model found, what the baseline found, and what it missed or
wrongly flagged. Be specific about cloud, shadow and seasonal effects, since those are common false positives in the
Gulf region (bright sand, haze and dust can look like change).

**Done when:** each site has a paragraph on what it found, what it missed, and why.

---

## Part C: finish

### Step C1. Update the README

Add a "v2 results" section to `README.md` with:

- The Part A table and the conclusion, with a link to `docs/V2_RESULTS.md`.
- The Part B case studies, with a link to `docs/CASE_STUDIES.md`.
- The `Status` note at the top, updated to say what was tested on real data.

Do not remove the synthetic results. Keep them and label them as synthetic.

**Done when:** the README renders, every link resolves, and every number matches a file in `docs/data/` or `outputs/`.

### Step C2. Run the full check

- `pytest` passes.
- `python scripts/reproduce_results.py` and the new `scripts/evaluate_v2.py` run from a clean checkout with the
  commands in the README.
- No file over 5 MB is committed. No raw data is committed.

**Done when:** the commands succeed from a fresh clone, and `git status` is clean.

---

## Notes for whoever works on this

- The model is small and CPU-trained. Real Sentinel-2 scenes at 10 m are large. Use tiled inference, and keep scenes
  to a few km across.
- The README's current limits still apply: the network does not beat a colour difference on synthetic scenes, and the
  threshold over-marks changed pixels. Part A is designed to test those claims on real data, not to assume them.
- If a step's **Done when** cannot be met, stop, write down what blocked it in `docs/V2_BLOCKERS.md`, and do not
  change the requirement to fit.
