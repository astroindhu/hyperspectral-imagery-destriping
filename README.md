# Moment-matching row correction (ENVI HSI destriping)

This repo contains a **simple, reusable** Python implementation of *moment matching row correction* for ENVI hyperspectral cubes.

✅ Main feature: **moment-matching row destriping** (masked; zero pixels stay zero)  
✅ Works on a **single file** or **batch over a whole directory**  
✅ Saves outputs back as **ENVI** (`.hdr` + `.img`) while preserving metadata

---


## How the destriping works (moment-matching row correction)

Many hyperspectral cubes contain **horizontal striping artifacts**, caused by detector row-to-row gain/offset variations or scan-line calibration differences. These appear as systematic brightness differences across image rows, often consistent across spectral bands.

This repository implements a **moment-matching row correction** algorithm that removes such striping while preserving real spectral variability.

### Algorithm (per band)

For each spectral band independently:

1. Identify valid pixels using a mask (default: pixels whose spectra are not all zeros).
2. Compute the **global mean** and **global standard deviation** of the band using only valid pixels:
   - μ_global(b), σ_global(b)
3. For each image row *r*, compute the row statistics using only valid pixels:
   - μ_row(r,b), σ_row(r,b)
4. Apply a linear transformation to match the row distribution to the global distribution:

x_corr = (x - μ_row) / σ_row * σ_global + μ_global


5. Invalid pixels remain unchanged (set to 0).

### Sparse-row protection

If a row contains fewer than `min_valid_per_row` valid pixels, the correction is skipped for that row to prevent unstable statistics.

### Why this works

This approach corrects both **row-dependent offsets** (mean shifts) and **row-dependent gain variations** (standard deviation scaling). Because the correction is done independently for each band, it preserves spectral shape and real spatial variability while reducing striping artifacts.

---


## Files in this repo

- `load_data.py`  
  Find `.hdr` files, load ENVI cubes, create a valid-pixel mask (non-zero spectra), flatten/restore helpers.

- `save_envi.py`  
  Writes ENVI `.hdr` + `.img` with metadata preservation (used by `destripe.py` and `batch_run.py`).

- `destripe.py`  
  **Moment-matching row correction** (masked).  
  **This is the main script most users run for a single file.**

- `batch_run.py`  
  Batch runner: recursively scans a directory for `.hdr` files and processes them all.

- `pca.py` (optional)  
  PCA analysis of a cube → saves `pca_outputs.npz`.

- `plot.py` (optional)  
  Saves PNG figures (raw vs processed band, PCA scree, PCA RGB).

---


## 1) Install Python (easy option)

**Recommended:** install **Miniconda** (lightweight Python distribution).

- Download Miniconda: search “Miniconda download” and install for your OS.
- Open **Anaconda Prompt** (Windows) or **Terminal** (macOS/Linux).

Create a new environment (copy/paste):

```bash
conda create -n hsi_destripe python=3.11 -y
conda activate hsi_destripe
```

Install required packages (minimum for destriping + ENVI I/O):

```bash
pip install numpy spectral
```

Optional packages (only needed for PCA + plotting):

```bash
pip install scikit-learn matplotlib
```

---

## 2) Put these scripts in a folder

Download this repo (or copy these `.py` files) into a folder, e.g.

```
moment_matching/
  destripe.py
  batch_run.py
  load_data.py
  save_envi.py
  pca.py
  plot.py
```

---

## 3) Run destriping on ONE file (most common)

You need:
- input ENVI header: `INPUT.hdr`
- output header path: `OUTPUT.hdr` (it will also create `OUTPUT.img`)

**Example command:**

```bash
python destripe.py --input_hdr "/path/to/INPUT.hdr" --output_hdr "/path/to/OUTPUT_destriped.hdr"
```

### Optional: tune sparse-row handling
Rows with too few valid pixels are left unchanged (default: 10). You can change it:

```bash
python destripe.py --input_hdr "/path/to/INPUT.hdr" --output_hdr "/path/to/OUTPUT_destriped.hdr" --min_valid_per_row 50
```

---

## 4) Batch process MANY files (entire directory)

This will:
- scan for `.hdr` files under `--data_dir`
- write processed files under `--out_dir`
- keep the same subfolder structure as the input
- add a suffix to the output filename (default: `_mm`)

**Dry run first (recommended):**

```bash
python batch_run.py \
  --data_dir "/path/to/raw_datasets" \
  --out_dir  "/path/to/processed_datasets" \
  --dry_run
```

**Real run:**

```bash
python batch_run.py \
  --data_dir "/path/to/raw_datasets" \
  --out_dir  "/path/to/processed_datasets" \
  --glob "**/*.hdr" \
  --suffix "_destriped" \
  --overwrite
```

You can narrow which files are processed using `--glob`, for example:

```bash
--glob "**/*rad_emiss*.hdr"
```

---

---

## 4b) Batch pipeline: destripe + PCA + plots (one command)

If you want to destripe *and* generate PCA outputs and PNG figures for every file in a directory, use `batch_pipeline.py`.

### Run PCA and plots for BOTH the input (raw) cube and the destriped cube

```bash
python batch_pipeline.py   --data_dir "/path/to/raw_datasets"   --out_dir  "/path/to/processed_datasets"   --include "rad_emiss"   --suffix "_destriped"   --run_pca   --pca_on both   --pca_variance 0.99   --make_plots   --plots_on both   --band 20   --overwrite
```

Outputs (per input file):
- destriped ENVI: `*_destriped.hdr` + `*_destriped.img`
- PCA (raw): `*_destriped.raw.pca.npz`
- PCA (destriped): `*_destriped.destriped.pca.npz`
- figures: saved under `.../<same_folder>/figs/` (raw + destriped band images, scree plots, PCA RGBs)

Tip: do a dry run first:

```bash
python batch_pipeline.py --data_dir "/path/to/raw" --out_dir "/path/to/proc" --include "rad_emiss" --dry_run
```


## 5) Optional: PCA (quick-look QA)

```bash
python pca.py --input_hdr "/path/to/OUTPUT_destriped.hdr" --out_npz "pca_outputs.npz" --variance 0.99
```

---

## 6) Optional: Make plots (PNG files)

Compare a band image (raw vs processed):

```bash
python plot.py --out_dir "figs" --band 20 --raw_hdr "/path/to/INPUT.hdr" --proc_hdr "/path/to/OUTPUT_destriped.hdr"
```

PCA plots:

```bash
python plot.py --out_dir "figs" --pca_npz "pca_outputs.npz"
```

---

## Troubleshooting

**“spectral not found”**
- Run: `pip install spectral`

**“File not found”**
- Make sure your `.hdr` and its data file (often `.img`) are in the same folder.
- Use quotes around paths, especially on Windows.

**ENVI metadata looks odd**
- This code preserves metadata best-effort; some headers may contain fields that ENVI readers interpret differently.
  If you share one header, I can make the metadata preservation stricter for your specific instrument/software.

**Common issue: “byte order” missing in ENVI headers**

Some ENVI .hdr files omit the byte order field. Certain readers (including spectral-python in some cases) expect this field and may throw an error when loading.
Solution: this repo automatically patches missing byte order by assuming byte order = 0 (little-endian), which is the standard on most modern systems.
All outputs written by this repo will include a valid byte order field for maximum compatibility.

**Best practice: apply destriping only on ROI pixels (recommended)**

This destriping method performs best when the correction statistics (global mean and standard deviation) are computed only from the region of interest (ROI).

If the cube contains large non-target regions—such as atmosphere/sky pixels, vegetation or trees in the background, shadows, or other unwanted background materials (common in field hyperspectral or field spectrometer acquisitions)—then the computed global mean/std may be dominated by these pixels.

As a result, the moment-matching correction may become less effective and can lead to over-correction or under-correction of the actual target surface.

Recommendation: mask the cube to include only ROI pixels before running the moment-matching row correction.

---

## License
This project is licensed under the MIT License.
