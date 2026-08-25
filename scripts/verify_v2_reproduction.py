#!/usr/bin/env python3
"""Verify the regenerated 1208-example v2 experiment against frozen results."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.train_xgb_v2 import (  # noqa: E402
    V1_STYLE_FEATURE_COLS,
    V2_FEATURE_COLS,
)


DEFAULT_ARTIFACTS_DIR = REPO / "data" / "v2_features"
DEFAULT_EXPECTED_PATH = REPO / "results" / "v2_expected_results.json"

PRIMARY_MODEL_NAME = "xgboost_dhi_14feature_v2.json"
SECONDARY_MODEL_NAME = "xgboost_dhi_14feature_v1style_secondary.json"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metrics(y_true, probabilities, predictions):
    return {
        "roc_auc": roc_auc_score(y_true, probabilities),
        "pr_auc": average_precision_score(y_true, probabilities),
        "recall": recall_score(y_true, predictions),
        "precision": precision_score(y_true, predictions),
        "f1": f1_score(y_true, predictions),
    }


def compare_metrics(name, actual, expected, tolerance, errors):
    for metric_name, expected_value in expected.items():
        actual_value = actual[metric_name]
        if not np.isclose(
            actual_value,
            expected_value,
            rtol=0.0,
            atol=tolerance,
        ):
            errors.append(
                f"{name}.{metric_name}: expected {expected_value}, "
                f"got {actual_value}"
            )


def load_model(path, expected_features, errors):
    try:
        model = XGBClassifier()
        model.load_model(path)
    except Exception as error:
        errors.append(f"Could not load model {path}: {error}")
        return

    actual_features = model.get_booster().feature_names
    if actual_features != expected_features:
        errors.append(
            f"{path.name}: expected feature names {expected_features}, "
            f"got {actual_features}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=DEFAULT_ARTIFACTS_DIR,
    )
    parser.add_argument(
        "--expected",
        type=Path,
        default=DEFAULT_EXPECTED_PATH,
    )
    args = parser.parse_args()

    artifacts_dir = args.artifacts_dir.resolve()
    expected_path = args.expected.resolve()
    features_path = artifacts_dir / "v2_combined_features.parquet"
    predictions_path = artifacts_dir / "v2_loso_predictions.csv"
    primary_model_path = artifacts_dir / PRIMARY_MODEL_NAME
    secondary_model_path = artifacts_dir / SECONDARY_MODEL_NAME

    required_paths = [
        expected_path,
        features_path,
        predictions_path,
        primary_model_path,
        secondary_model_path,
    ]
    missing_paths = [path for path in required_paths if not path.is_file()]
    if missing_paths:
        raise RuntimeError(
            "Required reproduction artifacts are missing:\n- "
            + "\n- ".join(map(str, missing_paths))
        )

    expected = json.loads(expected_path.read_text(encoding="utf-8"))
    features = pd.read_parquet(features_path)
    predictions = pd.read_csv(predictions_path)
    errors = []

    expected_features = expected["feature_names"]
    if V2_FEATURE_COLS != expected_features:
        errors.append(
            "train_xgb_v2.V2_FEATURE_COLS differs from the frozen feature list"
        )
    if V1_STYLE_FEATURE_COLS != [
        f"v1style_{name}" for name in expected_features
    ]:
        errors.append(
            "train_xgb_v2.V1_STYLE_FEATURE_COLS differs from the frozen "
            "feature list"
        )

    if len(features) != expected["examples"]:
        errors.append(
            f"examples: expected {expected['examples']}, got {len(features)}"
        )
    if len(predictions) != expected["examples"]:
        errors.append(
            "prediction rows: expected "
            f"{expected['examples']}, got {len(predictions)}"
        )

    actual_counts = {
        key: int(value)
        for key, value in features.groupby("dataset_source").size().items()
    }
    if actual_counts != expected["source_counts"]:
        errors.append(
            f"source counts: expected {expected['source_counts']}, "
            f"got {actual_counts}"
        )

    feature_labels = features["is_dhi"].astype(bool)
    actual_positives = int(feature_labels.sum())
    actual_negatives = int((~feature_labels).sum())
    if actual_positives != expected["positives"]:
        errors.append(
            f"positives: expected {expected['positives']}, "
            f"got {actual_positives}"
        )
    if actual_negatives != expected["negatives"]:
        errors.append(
            f"negatives: expected {expected['negatives']}, "
            f"got {actual_negatives}"
        )

    actual_sites = int(features["site_id"].nunique())
    if actual_sites != expected["sites"]:
        errors.append(
            f"sites: expected {expected['sites']}, got {actual_sites}"
        )

    all_feature_columns = V2_FEATURE_COLS + V1_STYLE_FEATURE_COLS
    missing_feature_columns = [
        name for name in all_feature_columns if name not in features.columns
    ]
    if missing_feature_columns:
        errors.append(
            f"Missing feature columns: {missing_feature_columns}"
        )
    else:
        values = features[all_feature_columns].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            errors.append("Feature table contains NaN or infinite values")

    if not features["global_id"].is_unique:
        errors.append("Feature table global_id values are not unique")
    if not predictions["global_id"].is_unique:
        errors.append("Prediction global_id values are not unique")
    if features["global_id"].tolist() != predictions["global_id"].tolist():
        errors.append(
            "Feature and prediction rows do not have identical global_id order"
        )

    prediction_labels = predictions["is_dhi"].astype(bool)
    if not np.array_equal(feature_labels.to_numpy(), prediction_labels.to_numpy()):
        errors.append("Feature and prediction labels differ")

    primary_probabilities = predictions["loso_probability"].to_numpy(float)
    secondary_probabilities = predictions[
        "v1style_loso_probability"
    ].to_numpy(float)
    primary_predictions = predictions["loso_prediction"].astype(bool).to_numpy()
    secondary_predictions = predictions[
        "v1style_loso_prediction"
    ].astype(bool).to_numpy()

    if not np.isfinite(primary_probabilities).all():
        errors.append("Primary LOSO probabilities contain NaN or infinity")
    if not np.isfinite(secondary_probabilities).all():
        errors.append("Secondary LOSO probabilities contain NaN or infinity")
    if not np.array_equal(
        primary_predictions,
        primary_probabilities >= 0.5,
    ):
        errors.append("Primary LOSO predictions do not use threshold 0.5")
    if not np.array_equal(
        secondary_predictions,
        secondary_probabilities >= 0.5,
    ):
        errors.append("Secondary LOSO predictions do not use threshold 0.5")

    primary_metrics = metrics(
        prediction_labels,
        primary_probabilities,
        primary_predictions,
    )
    secondary_metrics = metrics(
        prediction_labels,
        secondary_probabilities,
        secondary_predictions,
    )
    tolerance = expected["metric_absolute_tolerance"]
    compare_metrics(
        "primary_loso",
        primary_metrics,
        expected["primary_loso"],
        tolerance,
        errors,
    )
    compare_metrics(
        "secondary_loso",
        secondary_metrics,
        expected["secondary_loso"],
        tolerance,
        errors,
    )

    actual_model_hashes = {
        "primary": sha256(primary_model_path),
        "secondary": sha256(secondary_model_path),
    }
    if actual_model_hashes != expected["model_sha256"]:
        errors.append(
            f"model SHA-256: expected {expected['model_sha256']}, "
            f"got {actual_model_hashes}"
        )

    load_model(primary_model_path, expected_features, errors)
    load_model(secondary_model_path, expected_features, errors)

    actual = {
        "examples": len(features),
        "source_counts": actual_counts,
        "positives": actual_positives,
        "negatives": actual_negatives,
        "sites": actual_sites,
        "primary_loso": primary_metrics,
        "secondary_loso": secondary_metrics,
        "model_sha256": actual_model_hashes,
    }
    print(json.dumps(actual, indent=2))

    if errors:
        raise RuntimeError(
            "v2 reproduction verification failed:\n- "
            + "\n- ".join(errors)
        )

    print("\nV2 reproduction verified successfully.")


if __name__ == "__main__":
    main()
