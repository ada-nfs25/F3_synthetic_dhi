#!/usr/bin/env python3
# Implementation developed with AI (Claude Code) assistance - see AI_USAGE.md.
"""Export the frozen dim-spot subset as an amplitude-only supplement."""

import json
import shutil
import sys
from pathlib import Path

import pandas as pd


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.export_aziz_style_supplement import (  # noqa: E402
    export_patch,
    write_checksums,
)


SOURCE_DIR = REPO / "data" / "F3_synthetic_dhi_dataset_p0_dim_spot"
BUILD_ROOT = REPO / "data" / "release_build" / "f3_dim_spot_supplement_v1"
OUTPUT_DIR = BUILD_ROOT / "F3_synthetic_dhi_dataset_p0_dim_spot"

EXPECTED_EXAMPLES = 201


def validate_labels(labels):
    if len(labels) != EXPECTED_EXAMPLES:
        raise RuntimeError(
            f"Expected {EXPECTED_EXAMPLES} labels, found {len(labels)}"
        )

    required = {
        "is_dhi",
        "kind",
        "tier",
        "il_center",
        "xl_center",
        "center_time_ms",
        "example_id",
        "split",
        "patch_file",
        "horizon",
    }
    missing = required - set(labels.columns)
    if missing:
        raise RuntimeError(f"Missing label columns: {sorted(missing)}")

    if labels["patch_file"].nunique() != EXPECTED_EXAMPLES:
        raise RuntimeError("patch_file values are not unique")
    if labels["example_id"].nunique() != EXPECTED_EXAMPLES:
        raise RuntimeError("example_id values are not unique")
    if set(labels["kind"].dropna().unique()) != {"dim_spot"}:
        raise RuntimeError("Expected every example to have kind='dim_spot'")
    if set(labels["horizon"].dropna().unique()) != {"H1", "H3"}:
        raise RuntimeError("Expected dim-spot examples from H1 and H3")


def main():
    if not SOURCE_DIR.exists():
        raise RuntimeError(f"Source dataset does not exist: {SOURCE_DIR}")

    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        raise RuntimeError(
            f"Output directory is not empty: {OUTPUT_DIR}\n"
            "Move or remove the previous release build before rerunning."
        )

    labels = pd.read_parquet(SOURCE_DIR / "labels.parquet")
    validate_labels(labels)

    patches_output = OUTPUT_DIR / "patches"
    patches_output.mkdir(parents=True, exist_ok=True)

    for number, row in enumerate(labels.itertuples(index=False), start=1):
        source_path = SOURCE_DIR / "patches" / row.patch_file
        destination_path = patches_output / row.patch_file

        if not source_path.is_file():
            raise RuntimeError(f"Missing source patch: {source_path}")

        export_patch(source_path, destination_path)
        print(f"[{number:03d}/{EXPECTED_EXAMPLES}] {row.patch_file}")

    shutil.copy2(SOURCE_DIR / "labels.csv", OUTPUT_DIR / "labels.csv")
    shutil.copy2(SOURCE_DIR / "labels.parquet", OUTPUT_DIR / "labels.parquet")

    manifest = json.loads(
        (SOURCE_DIR / "generation_manifest.json").read_text(encoding="utf-8")
    )
    manifest["distribution"] = {
        "dataset_id": "f3-dim-spot-supplement-v1",
        "version": 1,
        "n_examples": EXPECTED_EXAMPLES,
        "representation": "amplitude_and_ground_truth_only",
        "distributed_arrays": ["amplitude", "mask", "mask_3d"],
        "amplitude_shape": [96, 96, 125],
        "amplitude_dtype": "float32",
        "generator_code_distributed": False,
        "purpose": (
            "Frozen dim-spot diversity supplement for the 1208-example "
            "in-house XGBoost experiment."
        ),
    }
    (OUTPUT_DIR / "generation_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    provenance = """# Provenance

## Dataset

`f3-dim-spot-supplement-v1` contains 201 synthetic dim-spot examples on the
public F3 Demo 2023 seismic volume: 81 examples on horizon H1 and 120 on H3.

## Attribution

The host-reflectivity attenuation mechanic was adapted from Mohammed Aziz
Ketata's independent DHI-generation implementation. Dataset integration,
calibration against the F3 volume, generation, amplitude-only export,
validation, feature computation, and XGBoost evaluation were performed by
Nora Færevaag Solberg.

## Permission

Mohammed Aziz Ketata gave permission on 25 August 2026 for these generated
outputs to be redistributed for research reproducibility. This release does
not include his source code, calibration files, or wavelet.

## Contents

Each patch contains:

- `amplitude`: float32 array with shape `(96, 96, 125)`;
- `mask`: boolean 2D injection-footprint mask with shape `(96, 96)`;
- `mask_3d`: boolean ground-truth mask with shape `(96, 96, 125)`.

Derived seismic attributes are deliberately excluded. They are recomputed
from amplitude using the public `F3_synthetic_dhi` feature pipeline.

## Purpose

These examples represent amplitude attenuation rather than brightening and
were included to reduce reliance on brightness-correlated detector features.
They should be combined with the H1, H3, and Aziz-style slices and should not
be treated as an independent training corpus.
"""
    (OUTPUT_DIR / "PROVENANCE.md").write_text(
        provenance,
        encoding="utf-8",
    )

    write_checksums(OUTPUT_DIR)

    exported_labels = pd.read_parquet(OUTPUT_DIR / "labels.parquet")
    exported_patches = sorted(patches_output.glob("*.npz"))
    if len(exported_labels) != EXPECTED_EXAMPLES:
        raise RuntimeError("Exported label count changed unexpectedly")
    if len(exported_patches) != EXPECTED_EXAMPLES:
        raise RuntimeError(
            f"Expected {EXPECTED_EXAMPLES} exported patches, "
            f"found {len(exported_patches)}"
        )

    total_bytes = sum(
        path.stat().st_size for path in OUTPUT_DIR.rglob("*") if path.is_file()
    )
    print()
    print(f"Export complete: {OUTPUT_DIR}")
    print(f"Examples: {len(exported_labels)}")
    print(f"Patch files: {len(exported_patches)}")
    print(f"Total size: {total_bytes / 1024**2:.1f} MiB")
    print(f"Checksums: {OUTPUT_DIR / 'SHA256SUMS'}")


if __name__ == "__main__":
    main()
