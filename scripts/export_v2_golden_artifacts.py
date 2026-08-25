#!/usr/bin/env python3
"""Export the small, sanitised golden artifacts for quick v2 verification."""

import shutil
from pathlib import Path

import pandas as pd


REPO = Path(__file__).resolve().parents[1]
SOURCE_DIR = REPO / "data" / "v2_features"
OUTPUT_DIR = REPO / "results" / "golden"

EXPECTED_EXAMPLES = 1208
EXPECTED_SOURCE_COUNTS = {
    "H1": 683,
    "H3": 296,
    "aziz_style": 28,
    "dim_spot": 201,
}


def main():
    source_features = SOURCE_DIR / "v2_combined_features.parquet"
    source_predictions = SOURCE_DIR / "v2_loso_predictions.csv"

    for path in (source_features, source_predictions):
        if not path.is_file():
            raise RuntimeError(f"Required source artifact is missing: {path}")

    features = pd.read_parquet(source_features)
    predictions = pd.read_csv(source_predictions)

    if len(features) != EXPECTED_EXAMPLES:
        raise RuntimeError(
            f"Expected {EXPECTED_EXAMPLES} feature rows, found {len(features)}"
        )
    if len(predictions) != EXPECTED_EXAMPLES:
        raise RuntimeError(
            f"Expected {EXPECTED_EXAMPLES} prediction rows, "
            f"found {len(predictions)}"
        )

    actual_counts = {
        key: int(value)
        for key, value in features.groupby("dataset_source").size().items()
    }
    if actual_counts != EXPECTED_SOURCE_COUNTS:
        raise RuntimeError(
            f"Expected source counts {EXPECTED_SOURCE_COUNTS}, "
            f"found {actual_counts}"
        )

    if features["global_id"].tolist() != predictions["global_id"].tolist():
        raise RuntimeError(
            "Feature and prediction rows do not have identical global_id order"
        )

    if "patch_path" in features.columns:
        features = features.drop(columns="patch_path")

    text_columns = [
        column
        for column in features.columns
        if pd.api.types.is_object_dtype(features[column].dtype)
        or isinstance(features[column].dtype, pd.StringDtype)
    ]
    for column in text_columns:
        values = features[column].dropna().astype(str)
        if values.str.contains(str(Path.home()), regex=False).any():
            raise RuntimeError(
                f"Personal absolute path remains in feature column {column}"
            )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    features.to_parquet(
        OUTPUT_DIR / "v2_combined_features.parquet",
        index=False,
    )
    shutil.copy2(
        source_predictions,
        OUTPUT_DIR / "v2_loso_predictions.csv",
    )

    print(f"Golden feature rows: {len(features)}")
    print(f"Golden prediction rows: {len(predictions)}")
    print(f"Wrote sanitized artifacts to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
