"""destripe.py — moment-matching row correction (masked) for ENVI cubes.

What it does
------------
For each band independently:
  - compute global mean/std using valid pixels only
  - for each row, compute row mean/std using valid pixels only
  - rescale pixels in that row to match the global moments:
        corrected = (x - row_mean) / row_std * global_std + global_mean
  - invalid pixels remain 0

CLI usage (write ENVI)
----------------------
python destripe.py --input_hdr /path/file.hdr --output_hdr /out/file_mm.hdr

Requirements
------------
pip install numpy spectral
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np

from load_data import load_envi, valid_mask_nonzero
from save_envi import write_envi_cube


def moment_matching_row_correction(
    cube: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    min_valid_per_row: int = 10,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Destripe an ENVI hyperspectral cube using moment-matching row correction.

    Purpose
    -------
    Remove horizontal striping artifacts caused by row-dependent gain/offset variations.

    Method (per spectral band)
    --------------------------
    For each band b:
        - Compute global mean/std using valid pixels only:
              mu_global[b], sigma_global[b]
        - Compute mean/std for each row r using valid pixels only:
              mu_row[r,b], sigma_row[r,b]
        - Apply linear rescaling so that each row matches the global distribution:

              corrected = (x - mu_row) / sigma_row * sigma_global + mu_global

    Invalid pixels are assumed to be 0 and are excluded from statistics.

    Parameters
    ----------
    cube : np.ndarray
        Input cube of shape (H, W, B).
    valid_mask : np.ndarray, optional
        Boolean mask of shape (H, W). True = valid pixel.
        If None, valid pixels are inferred as pixels with any non-zero band value.
    min_valid_per_row : int
        If a row has fewer than this number of valid pixels, that row is not corrected.
        This avoids unstable corrections for sparse/empty rows.
    eps : float
        Numerical stability threshold to prevent division by zero.

    Returns
    -------
    corrected : np.ndarray
        Corrected cube of shape (H, W, B), float32.
    """

    # Sanity check: expect a hyperspectral cube
    if cube.ndim != 3:
        raise ValueError(f"Expected cube shape (H,W,B), got {cube.shape}")

    H, W, B = cube.shape

    # If no mask provided, infer valid pixels from non-zero spectra
    if valid_mask is None:
        valid_mask = valid_mask_nonzero(cube)

    cube = cube.astype(np.float32)

    # Expand valid mask into 3D so it matches cube shape
    mask3 = np.broadcast_to(valid_mask[:, :, None], (H, W, B))

    # Replace invalid pixels with NaN so nanmean/nanstd ignore them
    cube_nan = np.where(mask3, cube, np.nan)

    # Compute global mean/std per band, and row mean/std per band
    with np.errstate(all="ignore"):
        gmean = np.nanmean(cube_nan, axis=(0, 1))  # shape (B,)
        gstd  = np.nanstd(cube_nan, axis=(0, 1))   # shape (B,)

        rmean = np.nanmean(cube_nan, axis=1)       # shape (H,B)
        rstd  = np.nanstd(cube_nan, axis=1)        # shape (H,B)

    # Avoid divide-by-zero or very small std values
    gstd = np.where(gstd > eps, gstd, 1.0)
    rstd = np.where(rstd > eps, rstd, 1.0)

    # Moment-matching correction: shift+scale each row to match global moments
    corrected = (cube_nan - rmean[:, None, :]) / rstd[:, None, :] * gstd + gmean

    # Protect sparse rows: if not enough valid pixels, keep original row unchanged
    row_counts = np.sum(valid_mask, axis=1)  # shape (H,)
    sparse = row_counts < min_valid_per_row
    if np.any(sparse):
        corrected[sparse, :, :] = cube[sparse, :, :]

    # Restore invalid pixels back to zero
    corrected = np.where(mask3, corrected, 0.0)

    return corrected.astype(np.float32)



def main():
    ap = argparse.ArgumentParser(description="Moment-matching row destriping for ENVI cubes.")
    ap.add_argument("--input_hdr", required=True, help="Path to input ENVI .hdr")
    ap.add_argument("--output_hdr", required=True, help="Path to output ENVI .hdr")
    ap.add_argument("--min_valid_per_row", type=int, default=10, help="Rows with fewer valid pixels remain unchanged")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite output if it exists")
    args = ap.parse_args()

    data = load_envi(args.input_hdr)
    cube = data.cube
    md = data.metadata
    valid_mask = valid_mask_nonzero(cube)

    corrected = moment_matching_row_correction(cube, valid_mask=valid_mask, min_valid_per_row=args.min_valid_per_row)

    out_hdr = Path(args.output_hdr).expanduser().resolve()
    write_envi_cube(out_hdr, corrected, metadata=md, dtype=np.float32, force=args.overwrite)

    print(f"Saved: {out_hdr}")

if __name__ == "__main__":
    main()
