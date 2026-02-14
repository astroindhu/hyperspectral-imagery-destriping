"""pca.py — PCA utilities + CLI for ENVI cubes.

What it does
------------
- Load an ENVI cube
- Flatten valid pixels to (N,B)
- Optionally standardize per-band (recommended)
- Fit PCA keeping a target variance (e.g., 0.99)
- Save outputs to NPZ: scores, components, explained variance, and an (H,W,C) PCA score image

CLI
---
python pca.py --input_hdr /path/file_mm.hdr --out_npz pca_outputs.npz --variance 0.99

Requirements
------------
pip install numpy spectral scikit-learn
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from load_data import load_envi, valid_mask_nonzero, flatten_valid_spectra, restore_flat


def minmax_per_component(scores: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32)
    out = np.empty_like(scores)
    for k in range(scores.shape[1]):
        v = scores[:, k]
        out[:, k] = (v - v.min()) / (np.ptp(v) + eps)
    return out


def run_pca(
    cube: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    variance_to_keep: float = 0.99,
    scale_features: bool = True,
    random_state: int = 0,
) -> dict:
    if valid_mask is None:
        valid_mask = valid_mask_nonzero(cube)

    spectra_2d, flat_mask = flatten_valid_spectra(cube, valid_mask=valid_mask)

    X = spectra_2d.astype(np.float32)
    scaler = None
    if scale_features:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    pca = PCA(n_components=variance_to_keep, svd_solver="full", random_state=random_state)
    scores = pca.fit_transform(X).astype(np.float32)
    scores_norm = minmax_per_component(scores)

    H, W, _ = cube.shape
    pc_image = restore_flat(scores_norm, flat_mask, (H, W), fill_value=0.0).astype(np.float32)

    return {
        "scores": scores,
        "scores_norm": scores_norm,
        "components": pca.components_.astype(np.float32),
        "explained_variance_ratio": pca.explained_variance_ratio_.astype(np.float32),
        "pc_image": pc_image,
        "flat_mask": flat_mask,
    }


def main():
    ap = argparse.ArgumentParser(description="Run PCA on an ENVI cube and save NPZ outputs.")
    ap.add_argument("--input_hdr", required=True, help="Path to input ENVI .hdr (often corrected cube)")
    ap.add_argument("--out_npz", required=True, help="Output .npz path")
    ap.add_argument("--variance", type=float, default=0.99, help="Variance to keep (0-1)")
    ap.add_argument("--no_scale", action="store_true", help="Disable StandardScaler before PCA")
    args = ap.parse_args()

    data = load_envi(args.input_hdr)
    out = run_pca(data.cube, variance_to_keep=args.variance, scale_features=(not args.no_scale))

    out_npz = Path(args.out_npz).expanduser().resolve()
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        pca_scores=out["scores"],
        pca_scores_norm=out["scores_norm"],
        pca_components=out["components"],
        pca_evr=out["explained_variance_ratio"],
        pc_image=out["pc_image"],
        flat_mask=out["flat_mask"],
    )
    print(f"Saved: {out_npz}")

if __name__ == "__main__":
    main()
