"""save_envi.py — write ENVI cubes (.hdr + .img) with metadata.

Used by destripe.py (and optionally by you directly).

Requirements
------------
pip install numpy spectral
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import spectral as spy


def _coerce_metadata_value(v):
    if isinstance(v, (int, float, np.integer, np.floating)):
        return str(v)
    return v


def sanitize_metadata(md: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in (md or {}).items():
        if k is None:
            continue
        out[str(k)] = _coerce_metadata_value(v)
    return out


def write_envi_cube(
    out_hdr_path: str | Path,
    cube: np.ndarray,
    metadata: Optional[Dict[str, Any]] = None,
    dtype: np.dtype = np.float32,
    interleave: str = "bsq",
    force: bool = True,
) -> Tuple[Path, Path]:
    """Write (H,W,B) cube to ENVI using spectral.envi.save_image."""
    out_hdr_path = Path(out_hdr_path)
    if out_hdr_path.suffix.lower() != ".hdr":
        out_hdr_path = out_hdr_path.with_suffix(".hdr")
    out_hdr_path.parent.mkdir(parents=True, exist_ok=True)

    cube = np.asarray(cube, dtype=dtype)

    md = sanitize_metadata(metadata or {})
    md["lines"] = str(cube.shape[0])
    md["samples"] = str(cube.shape[1])
    md["bands"] = str(cube.shape[2])
    md["interleave"] = interleave

    md["data type"] = "4" if np.dtype(dtype) == np.float32 else md.get("data type", "5")
    md["byte order"] = str(md.get("byte order", "0"))
    desc = md.get("description", "")
    if isinstance(desc, (list, tuple)):
        desc = " ".join([str(x) for x in desc])
    md["description"] = (str(desc) + " | written by save_envi.py").strip(" |")

    spy.envi.save_image(
        str(out_hdr_path),
        cube,
        dtype=dtype,
        force=force,
        interleave=interleave,
        metadata=md,
    )
    return out_hdr_path, out_hdr_path.with_suffix(".dat")


def main():
    ap = argparse.ArgumentParser(description="Write an ENVI cube from an NPZ containing `cube` array.")
    ap.add_argument("--npz", required=True, help="Input .npz containing `cube` and optionally `metadata`")
    ap.add_argument("--output_hdr", required=True, help="Output .hdr path")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite output if exists")
    args = ap.parse_args()

    d = np.load(Path(args.npz).expanduser().resolve(), allow_pickle=True)
    cube = d["cube"]
    md = {}
    if "metadata" in d:
        md = d["metadata"].item() if hasattr(d["metadata"], "item") else {}

    write_envi_cube(args.output_hdr, cube, metadata=md, dtype=np.float32, force=args.overwrite)
    print(f"Saved: {args.output_hdr}")

if __name__ == "__main__":
    main()
