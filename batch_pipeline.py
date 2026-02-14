"""batch_pipeline.py — batch pipeline: destripe + PCA + plots (optional) on raw and/or destriped.

This script is meant for "run everything on a folder" workflows.

Features
--------
- Recursively finds ENVI .hdr files under --data_dir
- Optional filename filtering (--include / --exclude)
- Writes destriped ENVI cubes to --out_dir (mirrors folder structure)
- Optionally runs PCA on the RAW cube and/or the DESTRIPED cube
- Optionally writes PNG figures (band images + PCA plots) for RAW and/or DESTRIPED products

Requirements
------------
Minimum (destriping + ENVI I/O):
    pip install numpy spectral

Optional (PCA):
    pip install scikit-learn

Optional (plots):
    pip install matplotlib
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from load_data import find_hdr_files, load_envi, valid_mask_nonzero
from destripe import moment_matching_row_correction
from save_envi import write_envi_cube

# Optional imports (only used if flags enabled)
try:
    from pca import run_pca
except Exception:
    run_pca = None  # type: ignore

try:
    from plot import plot_band, plot_pca_scree, plot_pca_rgb
except Exception:
    plot_band = plot_pca_scree = plot_pca_rgb = None  # type: ignore


def mirror_output_hdr(in_hdr: Path, data_dir: Path, out_dir: Path, suffix: str) -> Path:
    rel = in_hdr.relative_to(data_dir)
    out_stem = in_hdr.stem + suffix
    return (out_dir / rel.parent / out_stem).with_suffix(".hdr")


def _matches_filter(path: Path, include: Optional[str], exclude: Optional[str]) -> bool:
    name = path.name.lower()
    if include and include.lower() not in name:
        return False
    if exclude and exclude.lower() in name:
        return False
    return True


def _parse_choice(x: str) -> str:
    x = x.strip().lower()
    if x not in {"raw", "destriped", "both"}:
        raise ValueError("--pca_on/--plots_on must be one of: raw, destriped, both")
    return x


def _save_pca_npz(path: Path, pca_out: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        pca_scores=pca_out["scores"],
        pca_scores_norm=pca_out["scores_norm"],
        pca_components=pca_out["components"],
        pca_evr=pca_out["explained_variance_ratio"],
        pc_image=pca_out["pc_image"],
        flat_mask=pca_out["flat_mask"],
    )


def main():
    ap = argparse.ArgumentParser(description="Batch pipeline: destripe + PCA + plots (optional).")
    ap.add_argument("--data_dir", required=True, help="Root directory containing ENVI .hdr files")
    ap.add_argument("--out_dir", required=True, help="Root directory to write destriped ENVI outputs")
    ap.add_argument("--glob", default="**/*.hdr", help="Glob pattern under data_dir to find .hdr files")
    ap.add_argument("--suffix", default="_destriped", help="Suffix appended to output filename stem")
    ap.add_argument("--include", default=None, help="Only process files whose NAME contains this substring (case-insensitive)")
    ap.add_argument("--exclude", default=None, help="Skip files whose NAME contains this substring (case-insensitive)")
    ap.add_argument("--min_valid_per_row", type=int, default=10, help="Rows with fewer valid pixels remain unchanged")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite outputs if they exist")
    ap.add_argument("--dry_run", action="store_true", help="Only print what would be processed")

    # PCA options
    ap.add_argument("--run_pca", action="store_true", help="Run PCA and save NPZ outputs")
    ap.add_argument("--pca_on", default="destriped", help="Run PCA on: raw | destriped | both (default: destriped)")
    ap.add_argument("--pca_variance", type=float, default=0.99, help="Variance to keep for PCA (0-1)")
    ap.add_argument("--pca_no_scale", action="store_true", help="Disable StandardScaler before PCA")

    # Plot options
    ap.add_argument("--make_plots", action="store_true", help="Save PNG plots (band images and/or PCA plots)")
    ap.add_argument("--plots_on", default="destriped", help="Make plots for: raw | destriped | both (default: destriped)")
    ap.add_argument("--band", type=int, default=20, help="Band index for band-image plots")
    ap.add_argument("--fig_dirname", default="figs", help="Figure folder name created under each output folder")
    ap.add_argument("--rgb", default="2,1,0", help="RGB PCA components for PCA-RGB plot, e.g., '2,1,0'")

    args = ap.parse_args()
    pca_on = _parse_choice(args.pca_on)
    plots_on = _parse_choice(args.plots_on)

    data_dir = Path(args.data_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    hdrs_all: List[Path] = find_hdr_files(data_dir, args.glob)
    if not hdrs_all:
        print(f"No .hdr files found under {data_dir} with glob: {args.glob}")
        return

    hdrs = [h for h in hdrs_all if _matches_filter(h, args.include, args.exclude)]
    if not hdrs:
        print("No files matched include/exclude filters.")
        print(f"Total headers found: {len(hdrs_all)}")
        return

    # Dependency checks
    if args.run_pca and run_pca is None:
        raise ImportError("PCA requested but pca.py (and its deps) could not be imported. Install scikit-learn.")
    if args.make_plots and plot_band is None:
        raise ImportError("Plotting requested but plot.py (and its deps) could not be imported. Install matplotlib.")

    print(f"Found {len(hdrs)} header files to process (out of {len(hdrs_all)} total).")

    for i, hdr in enumerate(hdrs, 1):
        out_hdr = mirror_output_hdr(hdr, data_dir, out_dir, args.suffix)

        # PCA output paths
        raw_pca_npz = out_hdr.with_suffix(".raw.pca.npz")
        destriped_pca_npz = out_hdr.with_suffix(".destriped.pca.npz")

        # Figure output folder
        fig_dir = out_hdr.parent / args.fig_dirname

        if args.dry_run:
            print(f"[{i}/{len(hdrs)}] {hdr} -> {out_hdr}")
            if args.run_pca:
                if pca_on in {"raw", "both"}:
                    print(f"           PCA(raw)      -> {raw_pca_npz}")
                if pca_on in {"destriped", "both"}:
                    print(f"           PCA(destriped) -> {destriped_pca_npz}")
            if args.make_plots:
                print(f"           FIGS -> {fig_dir} (plots_on={plots_on})")
            continue

        if out_hdr.exists() and not args.overwrite:
            print(f"[{i}/{len(hdrs)}] SKIP (exists): {out_hdr}")
            continue

        try:
            data = load_envi(hdr)
            raw_cube = data.cube
            valid_mask = valid_mask_nonzero(raw_cube)

            # 1) destripe
            corrected = moment_matching_row_correction(
                raw_cube,
                valid_mask=valid_mask,
                min_valid_per_row=args.min_valid_per_row,
            )
            out_hdr.parent.mkdir(parents=True, exist_ok=True)
            write_envi_cube(out_hdr, corrected, metadata=data.metadata, force=args.overwrite)
            print(f"[{i}/{len(hdrs)}] OK destripe: {hdr.name} -> {out_hdr.name}")

            # 2) PCA (optional) on raw and/or destriped
            pca_raw_out = None
            pca_des_out = None
            if args.run_pca:
                if pca_on in {"raw", "both"}:
                    pca_raw_out = run_pca(
                        raw_cube,
                        valid_mask=valid_mask,
                        variance_to_keep=args.pca_variance,
                        scale_features=(not args.pca_no_scale),
                    )
                    _save_pca_npz(raw_pca_npz, pca_raw_out)
                    print(f"           OK PCA(raw): {raw_pca_npz.name}")

                if pca_on in {"destriped", "both"}:
                    pca_des_out = run_pca(
                        corrected,
                        valid_mask=valid_mask,
                        variance_to_keep=args.pca_variance,
                        scale_features=(not args.pca_no_scale),
                    )
                    _save_pca_npz(destriped_pca_npz, pca_des_out)
                    print(f"           OK PCA(destriped): {destriped_pca_npz.name}")

            # 3) plots (optional) for raw and/or destriped
            if args.make_plots:
                fig_dir.mkdir(parents=True, exist_ok=True)

                # Band plots
                if plots_on in {"raw", "both"}:
                    plot_band(raw_cube, args.band, f"Raw band {args.band}",
                              outpath=fig_dir / f"{hdr.stem}_raw_band{args.band:02d}.png")
                if plots_on in {"destriped", "both"}:
                    plot_band(corrected, args.band, f"Destriped band {args.band}",
                              outpath=fig_dir / f"{out_hdr.stem}_destriped_band{args.band:02d}.png")

                # PCA plots (only if PCA was run for that product)
                rgb = tuple(int(x) for x in args.rgb.split(","))

                if args.run_pca and pca_raw_out is not None and plots_on in {"raw", "both"}:
                    plot_pca_scree(pca_raw_out["explained_variance_ratio"],
                                   outpath=fig_dir / f"{hdr.stem}_raw_pca_scree.png")
                    pc_img = pca_raw_out["pc_image"]
                    if pc_img.shape[2] >= 3:
                        plot_pca_rgb(pc_img, rgb=rgb,
                                     outpath=fig_dir / f"{hdr.stem}_raw_pca_rgb.png")

                if args.run_pca and pca_des_out is not None and plots_on in {"destriped", "both"}:
                    plot_pca_scree(pca_des_out["explained_variance_ratio"],
                                   outpath=fig_dir / f"{out_hdr.stem}_destriped_pca_scree.png")
                    pc_img = pca_des_out["pc_image"]
                    if pc_img.shape[2] >= 3:
                        plot_pca_rgb(pc_img, rgb=rgb,
                                     outpath=fig_dir / f"{out_hdr.stem}_destriped_pca_rgb.png")

                print(f"           OK FIGS: {fig_dir}")

        except Exception as e:
            print(f"[{i}/{len(hdrs)}] FAIL: {hdr} ({type(e).__name__}: {e})")

    print(f"Done. Outputs in: {out_dir}")


if __name__ == "__main__":
    main()
