"""load_data.py — ENVI loader + directory search helpers.

Purpose
-------
- Find ENVI .hdr files under a data directory
- Load a cube (H, W, B) + metadata + optional wavelengths
- Build a valid-pixel mask (non-zero spectra)

Requirements
------------
pip install numpy spectral
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import spectral as spy


@dataclass
class EnviCube:
    cube: np.ndarray                 # (H, W, B), float32
    metadata: Dict                   # ENVI metadata dict (best-effort)
    wavelengths: Optional[np.ndarray]  # (B,) float32 if present
    hdr_path: Path


def find_hdr_files(data_dir: str | Path, glob_pattern: str = "**/*.hdr") -> List[Path]:
    """Recursively find ENVI .hdr files under data_dir."""
    data_dir = Path(data_dir).expanduser().resolve()
    return sorted([p for p in data_dir.glob(glob_pattern) if p.is_file() and p.suffix.lower() == ".hdr"])

def ensure_required_envi_fields(md: dict) -> dict:
    """
    ENVI headers sometimes omit required-ish fields that some readers expect.
    We patch them with safe defaults.
    """
    md = dict(md or {})

    keys_lower = {str(k).strip().lower(): k for k in md.keys()}

    # ENVI byte order: 0 = little-endian, 1 = big-endian
    if "byte order" not in keys_lower:
        md["byte order"] = "0"

    return md




def load_envi(hdr_path: str | Path, dtype=np.float32) -> EnviCube:
    """Load ENVI cube + metadata + wavelengths (if present)."""
    hdr_path = Path(hdr_path).expanduser().resolve()
    img = spy.envi.open(str(hdr_path))
    md = dict(img.metadata) if img.metadata else {}
    md = ensure_required_envi_fields(md)
    cube = np.asarray(img.load(), dtype=dtype)

    wavelengths = None
    wkey = None
    # ENVI headers often use 'wavelength' (case-insensitive)
    for k in md.keys():
        if str(k).lower() == "wavelength":
            wkey = k
            break
    if wkey is not None:
        try:
            wavelengths = np.array([float(x) for x in md[wkey]], dtype=np.float32)
        except Exception:
            wavelengths = None

    return EnviCube(cube=cube, metadata=md, wavelengths=wavelengths, hdr_path=hdr_path)


def valid_mask_nonzero(cube: np.ndarray) -> np.ndarray:
    """Return (H,W) mask where the spectrum is not all zeros."""
    if cube.ndim != 3:
        raise ValueError(f"Expected (H,W,B), got {cube.shape}")
    return np.any(cube != 0, axis=2)


def flatten_valid_spectra(cube: np.ndarray, valid_mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Flatten cube to (N_valid,B) and return flat mask (H*W,)."""
    if cube.ndim != 3:
        raise ValueError(f"Expected (H,W,B), got {cube.shape}")
    H, W, B = cube.shape
    flat = cube.reshape(-1, B)
    if valid_mask is None:
        valid_mask = valid_mask_nonzero(cube)
    flat_mask = valid_mask.reshape(-1)
    return flat[flat_mask], flat_mask


def restore_flat(flat_values: np.ndarray, flat_mask: np.ndarray, hw: Tuple[int, int], fill_value: float = 0.0) -> np.ndarray:
    """Restore (N_valid, C) or (N_valid,) to (H,W,C) or (H,W)."""
    H, W = hw
    n_pix = H * W
    if flat_mask.shape[0] != n_pix:
        raise ValueError("flat_mask length does not match H*W")

    if flat_values.ndim == 1:
        out = np.full((n_pix,), fill_value, dtype=np.float32)
        out[flat_mask] = flat_values.astype(np.float32)
        return out.reshape(H, W)

    if flat_values.ndim == 2:
        C = flat_values.shape[1]
        out = np.full((n_pix, C), fill_value, dtype=np.float32)
        out[flat_mask, :] = flat_values.astype(np.float32)
        return out.reshape(H, W, C)

    raise ValueError(f"flat_values must be 1D or 2D, got {flat_values.shape}")
