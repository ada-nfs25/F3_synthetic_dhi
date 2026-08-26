#!/usr/bin/env python3
# Implementation developed with AI (Claude Code) assistance - see AI_USAGE.md.
"""Download and verify the frozen 201-example dim-spot supplement."""

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.download_aziz_style_supplement import (  # noqa: E402
    download,
    safely_extract,
    sha256,
    verify_internal_checksums,
)


URL = (
    "https://github.com/ada-nfs25/F3_synthetic_dhi/releases/download/"
    "dim-spot-supplement-v1/f3-dim-spot-supplement-v1.tar.gz"
)
ARCHIVE_SHA256 = (
    "fa27e6d7c8c42c5670737e3dc5bbf3f7"
    "bdabfea8473d84aac139bfce54eb099b"
)
ARCHIVE_NAME = "f3-dim-spot-supplement-v1.tar.gz"
DATASET_DIRNAME = "F3_synthetic_dhi_dataset_p0_dim_spot"
EXPECTED_EXAMPLES = 201
EXPECTED_SHAPE = (96, 96, 125)


def validate_dataset(dataset_dir):
    labels_path = dataset_dir / "labels.parquet"
    patches_dir = dataset_dir / "patches"

    if not labels_path.is_file():
        raise RuntimeError("Downloaded dataset has no labels.parquet")
    if not patches_dir.is_dir():
        raise RuntimeError("Downloaded dataset has no patches directory")

    labels = pd.read_parquet(labels_path)
    if len(labels) != EXPECTED_EXAMPLES:
        raise RuntimeError(
            f"Expected {EXPECTED_EXAMPLES} labels, found {len(labels)}"
        )
    if labels["patch_file"].nunique() != EXPECTED_EXAMPLES:
        raise RuntimeError("Patch filenames are not unique")
    if set(labels["kind"].dropna().unique()) != {"dim_spot"}:
        raise RuntimeError("Expected every example to have kind='dim_spot'")
    if set(labels["horizon"].dropna().unique()) != {"H1", "H3"}:
        raise RuntimeError("Expected dim-spot examples from H1 and H3")

    patch_files = sorted(patches_dir.glob("*.npz"))
    if len(patch_files) != EXPECTED_EXAMPLES:
        raise RuntimeError(
            f"Expected {EXPECTED_EXAMPLES} patches, found {len(patch_files)}"
        )

    expected_names = set(labels["patch_file"])
    actual_names = {path.name for path in patch_files}
    if actual_names != expected_names:
        raise RuntimeError("Patch files do not exactly match labels.parquet")

    for path in patch_files:
        with np.load(path) as patch:
            if set(patch.files) != {"amplitude", "mask", "mask_3d"}:
                raise RuntimeError(
                    f"{path.name} has unexpected arrays: {patch.files}"
                )

            amplitude = patch["amplitude"]
            if amplitude.shape != EXPECTED_SHAPE:
                raise RuntimeError(
                    f"{path.name}: unexpected amplitude shape "
                    f"{amplitude.shape}"
                )
            if amplitude.dtype != np.float32:
                raise RuntimeError(
                    f"{path.name}: unexpected amplitude dtype "
                    f"{amplitude.dtype}"
                )
            if not np.isfinite(amplitude).all():
                raise RuntimeError(
                    f"{path.name}: amplitude contains NaN or infinity"
                )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO / "data" / DATASET_DIRNAME,
    )
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise RuntimeError(
            f"Output already exists; refusing to overwrite: {output_dir}"
        )

    output_dir.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        prefix="f3-dim-spot-download-"
    ) as temporary:
        temporary_dir = Path(temporary)
        archive_path = temporary_dir / ARCHIVE_NAME
        extraction_dir = temporary_dir / "extracted"
        extraction_dir.mkdir()

        download(URL, archive_path)

        actual_archive_sha256 = sha256(archive_path)
        if actual_archive_sha256 != ARCHIVE_SHA256:
            raise RuntimeError(
                "Archive checksum mismatch:\n"
                f"expected: {ARCHIVE_SHA256}\n"
                f"actual:   {actual_archive_sha256}"
            )
        print("Archive SHA-256: OK")

        safely_extract(archive_path, extraction_dir)

        extracted_dataset = extraction_dir / DATASET_DIRNAME
        if not extracted_dataset.is_dir():
            raise RuntimeError(
                f"Archive does not contain {DATASET_DIRNAME}"
            )

        verify_internal_checksums(extracted_dataset)
        print("Internal checksums: OK")

        validate_dataset(extracted_dataset)
        print("Dataset validation: OK")

        shutil.move(str(extracted_dataset), str(output_dir))

    print()
    print(f"Installed {EXPECTED_EXAMPLES} examples at:")
    print(output_dir)


if __name__ == "__main__":
    main()
