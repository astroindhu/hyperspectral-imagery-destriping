"""batch_run.py — batch runner for moment-matching row correction on many ENVI files.

What it does
------------
- Recursively searches a data directory for ENVI `.hdr` files
- Applies moment-matching row correction (masked; keeps zero pixels as zero)
- Writes processed cubes back to ENVI (`.hdr` + `.img`) while preserving metadata
- Mirrors the input folder structure under the output directory

Typical usage
-------------
python batch_run.py \
  --data_dir "/path/to/raw_datasets" \
  --out_dir  "/path/to/processed_datasets" \
  --glob "**/*.hdr" \
  --suffix "_mm" \
  --overwrite

Requirements
------------
pip install numpy spectral
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from load_data import find_hdr_files, load_envi, valid_mask_nonzero
from destripe import moment_matching_row_correction
from save_envi import write_envi_cube


def mirror_output_hdr(in_hdr: Path, data_dir: Path, out_dir: Path, suffix: str) -> Path:
    """Create an output path mirroring input structure under out_dir."""
    rel = in_hdr.relative_to(data_dir)
    out_stem = in_hdr.stem + suffix
    return (out_dir / rel.parent / out_stem).with_suffix(".hdr")


def main():
    ap = argparse.ArgumentParser(description="Batch moment-matching row correction for ENVI cubes.")
    ap.add_argument("--data_dir", required=True, help="Root directory containing ENVI .hdr files")
    ap.add_argument("--out_dir", required=True, help="Root directory to write processed ENVI outputs")
    ap.add_argument("--glob", default="**/*.hdr", help="Glob pattern under data_dir to find .hdr files")
    ap.add_argument("--suffix", default="_mm", help="Suffix appended to output filename stem")
    ap.add_argument("--min_valid_per_row", type=int, default=10, help="Rows with fewer valid pixels remain unchanged")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite outputs if they exist")
    ap.add_argument("--dry_run", action="store_true", help="Only print what would be processed")
    args = ap.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    hdrs: List[Path] = find_hdr_files(data_dir, args.glob)
    if not hdrs:
        print(f"No .hdr files found under {data_dir} with glob: {args.glob}")
        return

    print(f"Found {len(hdrs)} header files.")

    for i, hdr in enumerate(hdrs, 1):
        out_hdr = mirror_output_hdr(hdr, data_dir, out_dir, args.suffix)

        if args.dry_run:
            print(f"[{i}/{len(hdrs)}] {hdr} -> {out_hdr}")
            continue

        if out_hdr.exists() and not args.overwrite:
            print(f"[{i}/{len(hdrs)}] SKIP (exists): {out_hdr}")
            continue

        try:
            data = load_envi(hdr)
            valid_mask = valid_mask_nonzero(data.cube)
            corrected = moment_matching_row_correction(
                data.cube,
                valid_mask=valid_mask,
                min_valid_per_row=args.min_valid_per_row,
            )
            write_envi_cube(out_hdr, corrected, metadata=data.metadata, force=args.overwrite)
            print(f"[{i}/{len(hdrs)}] OK: {hdr.name} -> {out_hdr.name}")
        except Exception as e:
            print(f"[{i}/{len(hdrs)}] FAIL: {hdr} ({type(e).__name__}: {e})")

    print(f"Done. Outputs in: {out_dir}")


if __name__ == "__main__":
    main()
