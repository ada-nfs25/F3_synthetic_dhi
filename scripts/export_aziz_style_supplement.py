#!/usr/bin/env python3
"""Export the frozen Aziz-style subset as an amplitude-only supplement.

This exports generated outputs only. It does not distribute the collaborator's
generator source, calibration files, or wavelet.
"""

import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]

SOURCE_DIR = (
    REPO / "data" / "F3_synthetic_dhi_dataset_p0_aziz_style"
)
BUILD_ROOT = (
    REPO / "data" / "release_build" / "f3_aziz_style_supplement_v1"
)
OUTPUT_DIR = (
    BUILD_ROOT / "F3_synthetic_dhi_dataset_p0_aziz_style"
)

EXPECTED_EXAMPLES = 28
EXPECTED_AMPLITUDE_SHAPE = (96, 96, 125)
EXPECTED_MASK_SHAPE = (96, 96)
EXPECTED_MASK_3D_SHAPE = (96, 96, 125)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
        "dataset_source",
    }
    missing = required - set(labels.columns)
    if missing:
        raise RuntimeError(f"Missing label columns: {sorted(missing)}")

    if labels["patch_file"].nunique() != EXPECTED_EXAMPLES:
        raise RuntimeError("patch_file values are not unique")

    if labels["example_id"].nunique() != EXPECTED_EXAMPLES:
        raise RuntimeError("example_id values are not unique")

    sources = set(labels["dataset_source"].dropna().unique())
    if sources != {"aziz_style"}:
        raise RuntimeError(
            f"Expected only dataset_source='aziz_style', found {sources}"
        )


def export_patch(source_path, destination_path):
    with np.load(source_path) as source:
        required = {"attribute_stack", "channel_names", "mask", "mask_3d"}
        missing = required - set(source.files)
        if missing:
            raise RuntimeError(
                f"{source_path} is missing arrays: {sorted(missing)}"
            )

        channel_names = source["channel_names"].tolist()
        if "amplitude" not in channel_names:
            raise RuntimeError(
                f"{source_path} has no amplitude channel: {channel_names}"
            )

        amplitude_index = channel_names.index("amplitude")
        amplitude = np.asarray(
            source["attribute_stack"][amplitude_index],
            dtype=np.float32,
        )
        mask = np.asarray(source["mask"], dtype=bool)
        mask_3d = np.asarray(source["mask_3d"], dtype=bool)

    if amplitude.shape != EXPECTED_AMPLITUDE_SHAPE:
        raise RuntimeError(
            f"{source_path}: amplitude shape is {amplitude.shape}, "
            f"expected {EXPECTED_AMPLITUDE_SHAPE}"
        )
    if mask.shape != EXPECTED_MASK_SHAPE:
        raise RuntimeError(
            f"{source_path}: mask shape is {mask.shape}, "
            f"expected {EXPECTED_MASK_SHAPE}"
        )
    if mask_3d.shape != EXPECTED_MASK_3D_SHAPE:
        raise RuntimeError(
            f"{source_path}: mask_3d shape is {mask_3d.shape}, "
            f"expected {EXPECTED_MASK_3D_SHAPE}"
        )
    if not np.isfinite(amplitude).all():
        raise RuntimeError(f"{source_path}: amplitude contains NaN or infinity")

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination_path,
        amplitude=amplitude,
        mask=mask,
        mask_3d=mask_3d,
    )

    # Confirm that serialization did not alter any array.
    with np.load(destination_path) as exported:
        if set(exported.files) != {"amplitude", "mask", "mask_3d"}:
            raise RuntimeError(
                f"{destination_path}: unexpected exported arrays "
                f"{exported.files}"
            )
        if not np.array_equal(exported["amplitude"], amplitude):
            raise RuntimeError(
                f"{destination_path}: exported amplitude differs from source"
            )
        if not np.array_equal(exported["mask"], mask):
            raise RuntimeError(
                f"{destination_path}: exported mask differs from source"
            )
        if not np.array_equal(exported["mask_3d"], mask_3d):
            raise RuntimeError(
                f"{destination_path}: exported mask_3d differs from source"
            )


def write_checksums(root):
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    )

    lines = [
        f"{sha256(path)}  {path.relative_to(root).as_posix()}"
        for path in files
    ]
    (root / "SHA256SUMS").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main():
    if not SOURCE_DIR.exists():
        raise RuntimeError(f"Source dataset does not exist: {SOURCE_DIR}")

    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        raise RuntimeError(
            f"Output directory is not empty: {OUTPUT_DIR}\n"
            "Move or remove the previous release build before rerunning."
        )

    source_labels_path = SOURCE_DIR / "labels.parquet"
    labels = pd.read_parquet(source_labels_path)
    validate_labels(labels)

    patches_output = OUTPUT_DIR / "patches"
    patches_output.mkdir(parents=True, exist_ok=True)

    for number, row in enumerate(labels.itertuples(index=False), start=1):
        source_path = SOURCE_DIR / "patches" / row.patch_file
        destination_path = patches_output / row.patch_file

        if not source_path.is_file():
            raise RuntimeError(f"Missing source patch: {source_path}")

        export_patch(source_path, destination_path)
        print(f"[{number:02d}/{EXPECTED_EXAMPLES}] {row.patch_file}")

    # Copy the original label tables byte-for-byte.
    shutil.copy2(SOURCE_DIR / "labels.csv", OUTPUT_DIR / "labels.csv")
    shutil.copy2(SOURCE_DIR / "labels.parquet", OUTPUT_DIR / "labels.parquet")

    original_manifest = json.loads(
        (SOURCE_DIR / "generation_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    original_manifest["distribution"] = {
        "dataset_id": "f3-aziz-style-supplement-v1",
        "version": 1,
        "n_examples": EXPECTED_EXAMPLES,
        "representation": "amplitude_and_ground_truth_only",
        "distributed_arrays": [
            "amplitude",
            "mask",
            "mask_3d",
        ],
        "amplitude_shape": list(EXPECTED_AMPLITUDE_SHAPE),
        "amplitude_dtype": "float32",
        "generator_code_distributed": False,
        "purpose": (
            "Frozen collaborator-generator diversity supplement for the "
            "1208-example in-house XGBoost experiment."
        ),
    }
    (OUTPUT_DIR / "generation_manifest.json").write_text(
        json.dumps(original_manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    provenance = """# Provenance

## Dataset

`f3-aziz-style-supplement-v1` contains 28 synthetic examples generated on
the public F3 Demo 2023 seismic volume using a collaborator's independent
DHI-generation implementation and calibration.

## Attribution

Generator and calibration: Mohammed Aziz Ketata.

Dataset integration, amplitude-only export, validation, feature computation,
and XGBoost evaluation: Nora Færevaag Solberg.

## Contents

Each patch contains:

- `amplitude`: float32 array with shape `(96, 96, 125)`;
- `mask`: boolean 2D injection-footprint mask with shape `(96, 96)`;
- `mask_3d`: boolean ground-truth mask with shape `(96, 96, 125)`.

Derived seismic attributes are deliberately excluded. They are recomputed
from amplitude using the public `f3-synthetic-dhi` pipeline.

## Purpose

This small subset was included to expose the detector to examples produced
under a second synthetic-generation convention, reducing reliance on
artifacts specific to the primary generator. It should be combined with the
H1, H3, and dim-spot subsets and should not be treated as an independent
training corpus.
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

    total_bytes = sum(path.stat().st_size for path in OUTPUT_DIR.rglob("*")
                      if path.is_file())

    print()
    print(f"Export complete: {OUTPUT_DIR}")
    print(f"Examples: {len(exported_labels)}")
    print(f"Patch files: {len(exported_patches)}")
    print(f"Total size: {total_bytes / 1024**2:.1f} MiB")
    print(f"Checksums: {OUTPUT_DIR / 'SHA256SUMS'}")


if __name__ == "__main__":
    main()