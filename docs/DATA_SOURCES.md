# Data sources (v2)

## OSCD: Onera Satellite Change Detection

The labelled real-data benchmark for v2.

| | |
| --- | --- |
| What it is | 24 registered Sentinel-2 image pairs with pixel-level urban change masks (new buildings, roads, construction sites) |
| Where | Brazil, the USA, Europe, the Middle East (including Abu Dhabi, Dubai and Beirut) and Asia |
| When | Image dates from 2015 to 2018; each pair is roughly one to three years apart |
| Official split | 14 training pairs and 10 test pairs. Labels for both are public |
| Home page | https://rcdaudt.github.io/oscd/ |
| Archive of record | IEEE DataPort, DOI [10.21227/asqe-7s69](https://ieee-dataport.org/open-access/oscd-onera-satellite-change-detection) |
| Download used here | https://huggingface.co/datasets/hkristen/oscd (the three original zip archives, unmodified) |

### Licence

- **Imagery.** The dataset README says the images are modified Copernicus Sentinel data from 2015 to 2018. Copernicus
  Sentinel data is free and open, with attribution. IEEE DataPort lists the dataset as Creative Commons Attribution.
- **Change masks.** The dataset README releases the change maps under Creative Commons BY-NC-SA, and asks anyone with
  a commercial purpose to contact the authors.

How this project uses it:

- Research use only. No commercial use.
- No raw imagery or label file is committed to this repository. `/data/` is in `.gitignore`.
- Metrics computed from the labels are published in this repository.
- Figures that show a label mask (the "truth" panels in `docs/img/v2/`) are derived from the labels, so those figures
  are shared under CC BY-NC-SA 4.0 with the citation below. The repository's MIT licence covers the code, not those
  figures.

This is the project's reading of the terms, not legal advice. Check the terms yourself before any use beyond research.

### Citation

Daudt, R.C., Le Saux, B., Boulch, A. and Gousseau, Y. (2018). Urban Change Detection for Multispectral Earth
Observation Using Convolutional Neural Networks. IEEE International Geoscience and Remote Sensing Symposium (IGARSS),
pp. 2115–2118.

Contains modified Copernicus Sentinel data, 2015–2018.

### Bands, resolution and processing level

- OSCD ships all 13 Sentinel-2 bands. This project uses `B02, B03, B04, B08` (blue, green, red, near-infrared). All
  four are native 10 m bands, so nothing is resampled.
- The bands are read from OSCD's registered folders (`imgs_1_rect`, `imgs_2_rect`), in which both dates are already
  cropped to one grid and co-registered by the dataset authors.
- Values are kept as the original digital numbers (reflectance × 10000). The pipeline's normalisation rescales them.
- **Processing level differs from the rest of the project.** OSCD is Level-1C (top-of-atmosphere). The project's own
  `scripts/fetch_sentinel2.py` downloads Level-2A (surface reflectance). A model or threshold fitted on OSCD is
  therefore applied to slightly different data in the UAE case studies. That is stated wherever it matters.
- The OSCD rasters carry no georeferencing, so the converted scenes are plain arrays (`.npy`).
- **Label encoding.** The label rasters store `1` for unchanged and `2` for changed. The dataset README says `0` and
  `1`. `scripts/fetch_benchmark.py` checks the values and writes masks as `0` (unchanged) and `1` (changed).

### How to download

```bash
pip install -e ".[geo]"
python scripts/fetch_benchmark.py
```

The script downloads three archives (about 513 MB), checks their SHA-256, extracts only the files it needs and writes:

```text
data/benchmark/
  raw/                               # the archives and extracted originals
  scenes.json                        # dates and sizes of every scene
  train/<scene>/images/<scene>_<YYYYMMDD>.npy     # no masks on disk for train scenes
  val/<scene>/images/...   val/<scene>/masks/<scene>.npy
  test/<scene>/images/...  test/<scene>/masks/<scene>.npy
```

SHA-256 of the archives:

| Archive | SHA-256 |
| --- | --- |
| `Onera Satellite Change Detection dataset - Images.zip` | `940b87887511058a933e67cd6d0e43e2eb825a55d8e79a50983dee7f23003656` |
| `Onera Satellite Change Detection dataset - Train Labels.zip` | `89fb54cd12ad0dbea6c447528139dec305b865294215434bf6dd170fb8fd3ca5` |
| `Onera Satellite Change Detection dataset - Test Labels.zip` | `2e195eaa1b788b99fa93ea8073e3780bc0b763000b0c49dbf70548acf1e5d67d` |

If the Hugging Face mirror is unavailable, download the same three archives from IEEE DataPort or the links on the
OSCD home page, put them in `data/benchmark/raw/` under the names above, and run the script again. It skips files that
are already present and still checks their hashes.

### Split

The split is by scene and is stored in [`docs/data/splits_v2.json`](data/splits_v2.json).

| Split | Scenes | Used for |
| --- | --- | --- |
| train (9) | beihai, bordeaux, cupertino, hongkong, mumbai, nantes, pisa, rennes, saclay_e | Self-supervised pretraining, imagery only. Labels are never opened |
| val (5) | abudhabi, aguasclaras, beirut, bercy, paris | Choosing thresholds and comparing model changes |
| test (10) | brasilia, chongqing, dubai, lasvegas, milano, montpellier, norcia, rio, saclay_w, valencia | Scored once per reported configuration |

The test split is OSCD's official test list. The five validation scenes were drawn from OSCD's official training list
with `random.Random(42).sample(sorted(train), 5)`.
