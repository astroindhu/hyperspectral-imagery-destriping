"""plot.py — plotting utilities (and saving) for HSI + PCA outputs.

CLI examples
------------
1) Compare raw vs corrected band:
   python plot.py --raw_hdr raw.hdr --proc_hdr corrected.hdr --out_dir figs --band 20

2) PCA scree + RGB from pca NPZ:
   python plot.py --pca_npz pca_outputs.npz --out_dir figs

Requirements
------------
pip install numpy matplotlib spectral
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

from load_data import load_envi


def savefig(fig, outpath: str | Path, dpi: int = 300) -> Path:
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    return outpath


def plot_band(cube: np.ndarray, band: int, title: str, outpath: Optional[str | Path] = None,
              vmin: Optional[float] = None, vmax: Optional[float] = None):
    img = cube[:, :, band]
    fig, ax = plt.subplots(figsize=(20, 10), dpi=300)
    im = ax.imshow(img, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Value")
    fig.tight_layout()
    if outpath is not None:
        savefig(fig, outpath)
    return fig, ax


def plot_pca_scree(evr: np.ndarray, outpath: Optional[str | Path] = None):
    evr = np.asarray(evr, dtype=np.float32)
    x = np.arange(1, evr.size + 1)
    fig, ax = plt.subplots(figsize=(15, 6), dpi=300)
    ax.plot(x, np.cumsum(evr), marker="o")
    ax.set_xlabel("Component")
    ax.set_ylabel("Cumulative explained variance")
    ax.set_title("PCA explained variance")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if outpath is not None:
        savefig(fig, outpath)
    return fig, ax


def plot_pca_rgb(pc_image: np.ndarray, rgb: Tuple[int, int, int] = (2, 1, 0), outpath: Optional[str | Path] = None):
    c0, c1, c2 = rgb
    if pc_image.ndim != 3 or pc_image.shape[2] < 3:
        raise ValueError("pc_image must be (H,W,C) with C>=3")
    rgb_img = pc_image[:, :, [c0, c1, c2]]
    fig, ax = plt.subplots(figsize=(20, 10), dpi=300)
    ax.imshow(rgb_img)
    ax.set_title(f"PCA RGB (components {rgb})")
    ax.axis("off")
    fig.tight_layout()
    if outpath is not None:
        savefig(fig, outpath)
    return fig, ax


def main():
    ap = argparse.ArgumentParser(description="Plot raw/processed bands and/or PCA outputs; saves PNGs.")
    ap.add_argument("--out_dir", required=True, help="Output directory for figures")
    ap.add_argument("--band", type=int, default=20, help="Band index for band images")
    ap.add_argument("--raw_hdr", default=None, help="Optional raw cube .hdr for band plot")
    ap.add_argument("--proc_hdr", default=None, help="Optional processed cube .hdr for band plot")
    ap.add_argument("--pca_npz", default=None, help="Optional PCA outputs .npz from pca.py")
    ap.add_argument("--rgb", default="2,1,0", help="RGB PCA components, e.g., '2,1,0'")
    args = ap.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # band plots
    if args.raw_hdr:
        raw = load_envi(args.raw_hdr).cube
        plot_band(raw, args.band, f"Raw band {args.band}", outpath=out_dir / f"band_{args.band:02d}_raw.png")
    if args.proc_hdr:
        proc = load_envi(args.proc_hdr).cube
        plot_band(proc, args.band, f"Processed band {args.band}", outpath=out_dir / f"band_{args.band:02d}_processed.png")

    # PCA plots
    if args.pca_npz:
        p = Path(args.pca_npz).expanduser().resolve()
        d = np.load(p, allow_pickle=False)
        evr = d["pca_evr"]
        pc_image = d["pc_image"]
        plot_pca_scree(evr, outpath=out_dir / "pca_scree_cumulative.png")
        rgb = tuple(int(x) for x in args.rgb.split(","))
        if pc_image.shape[2] >= 3:
            plot_pca_rgb(pc_image, rgb=rgb, outpath=out_dir / f"pca_rgb_{rgb[0]}_{rgb[1]}_{rgb[2]}.png")

    print(f"Saved figures to: {out_dir}")

if __name__ == "__main__":
    main()
