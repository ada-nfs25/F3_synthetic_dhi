#!/usr/bin/env python3
"""Download and verify the frozen 28-example Aziz-style supplement."""

import argparse
import hashlib
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]

URL = (
    "https://github.com/ada-nfs25/F3_synthetic_dhi/releases/download/"
    "aziz-style-supplement-v1/f3-aziz-style-supplement-v1.tar.gz"
)
ARCHIVE_SHA256 = (
    "00a5b6da0f708da9639b1c3f1e82b9d6"
    "a06f25125c22aae111a9e1bf384ff651"
)
DATASET_DIRNAME = "F3_synthetic_dhi_dataset_p0_aziz_style"
EXPECTED_EXAMPLES = 28
EXPECTED_SHAPE = (96, 96, 125)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url, destination):
    print(f"Downloading {url}")
    with urllib.request.urlopen(url) as response:
        total = int(response.headers.get("Content-Length", 0))
        received = 0

        with destination.open("wb") as output:
            while block := response.read(1024 * 1024):
                output.write(block)
                received += len(block)

                if total:
                    print(
                        f"\r  {received / 1024**2:.1f} / "
                        f"{total / 1024**2:.1f} MiB",
                        end="",
                    )

    print()


def safely_extract(archive_path, extraction_dir):
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            if member.issym() or member.islnk():
                raise RuntimeError(
                    f"Archive contains a link, refusing extraction: "
                    f"{member.name}"
                )

            member_path = (extraction_dir / member.name).resolve()
            try:
                member_path.relative_to(extraction_dir.resolve())
            except ValueError as error:
                raise RuntimeError(
                    f"Unsafe archive path: {member.name}"
                ) from error

        archive.extractall(extraction_dir, filter="data")


def verify_internal_checksums(dataset_dir):
    checksums_path = dataset_dir / "SHA256SUMS"
    if not checksums_path.is_file():
        raise RuntimeError("Downloaded dataset has no SHA256SUMS file")

    for line in checksums_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue

        expected, relative_name = line.split(maxsplit=1)
        relative_name = relative_name.strip()
        if relative_name.startswith("*"):
            relative_name = relative_name[1:]
        if relative_name.startswith("./"):
            relative_name = relative_name[2:]

        path = dataset_dir / relative_name
        if not path.is_file():
            raise RuntimeError(
                f"Checksum manifest references missing file: {relative_name}"
            )

        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(
                f"Checksum mismatch for {relative_name}: "
                f"expected {expected}, got {actual}"
            )


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

    sources = set(labels["dataset_source"].dropna().unique())
    if sources != {"aziz_style"}:
        raise RuntimeError(
            f"Unexpected dataset_source values: {sorted(sources)}"
        )

    patch_files = sorted(patches_dir.glob("*.npz"))
    if len(patch_files) != EXPECTED_EXAMPLES:
        raise RuntimeError(
            f"Expected {EXPECTED_EXAMPLES} patches, found {len(patch_files)}"
        )

    expected_names = set(labels["patch_file"])
    actual_names = {path.name for path in patch_files}
    if actual_names != expected_names:
        raise RuntimeError(
            "Patch files do not exactly match labels.parquet"
        )

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
        prefix="f3-aziz-style-download-"
    ) as temporary:
        temporary_dir = Path(temporary)
        archive_path = temporary_dir / (
            "f3-aziz-style-supplement-v1.tar.gz"
        )
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