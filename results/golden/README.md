# Golden v2 verification artifacts

This directory contains the small, committed reference outputs for quickly
verifying the reported 1208-example XGBoost experiment without regenerating
the full patch corpus:

- `v2_combined_features.parquet`: the sanitized feature/label table, with the
  machine-specific `patch_path` column removed;
- `v2_loso_predictions.csv`: the frozen primary and secondary LOSO
  probabilities and predictions.

The canonical frozen model JSON files remain in `data/v2_features/`. Verify
the golden artifacts, metrics, feature schema, model feature names, and model
hashes with:

```bash
python scripts/verify_v2_reproduction.py \
  --artifacts-dir results/golden \
  --models-dir data/v2_features
```

These golden artifacts provide the quick examiner route. They do not replace
the documented full route that reconstructs the corpus from the public F3
inputs and frozen collaborator-derived supplements.
