# Reproducibility guide

Two routes are provided for the frozen 1,208-example v2 experiment. The
quick route verifies the committed reference evaluation and model identities.
The full route rebuilds the derived dataset and reruns the evaluation.

The verified code and golden artifacts are pinned at
[`dataset-v2-1208`](https://github.com/ada-nfs25/F3_synthetic_dhi/tree/dataset-v2-1208).

## Quick examiner verification

This route was tested successfully from a clean clone. It uses approximately
1 MB of committed artifacts and does not regenerate the patch corpus or
retrain the models.

```bash
git clone https://github.com/ada-nfs25/F3_synthetic_dhi.git
cd F3_synthetic_dhi
git checkout dataset-v2-1208

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python scripts/verify_v2_reproduction.py \
  --artifacts-dir results/golden \
  --models-dir data/v2_features
```

Expected final line:

```text
V2 reproduction verified successfully.
```

The verifier checks:

- 1,208 examples and the 683/296/201/28 source counts;
- 627 positives, 581 negatives, and 164 LOSO sites;
- the feature schema, finite feature values, IDs, and labels;
- the recorded primary and secondary LOSO metrics;
- the model feature names and exact SHA-256 hashes.

The full-precision expected values are recorded in
[`results/v2_expected_results.json`](results/v2_expected_results.json). The
primary model's LOSO ROC-AUC is 0.755378 and the secondary v1-style ablation's
is 0.765532.

## Full reconstruction

This route:

1. regenerates H1 and H3 from public F3 inputs;
2. downloads the frozen dim-spot and Aziz-style amplitude supplements;
3. recomputes the derived attributes and scalar features for all four sources;
4. retrains the primary and secondary XGBoost models; and
5. compares the reconstructed evaluation against the expected result.

Allow approximately 40–50 GB of free storage. This route is documented but
has not been rerun end-to-end from a separate cold clone.

### 1. Obtain the public F3 inputs

Follow [`data_raw/README.md`](data_raw/README.md) for the public sources. The
minimum required layout is:

```text
data_raw/
|-- Seismic_data.sgy
|-- Velocity_functions.txt
`-- horizons/
    |-- H1.xyz
    `-- H3.xyz
```

Build the trace index while using the verified code:

```bash
git checkout dataset-v2-1208
python scripts/build_f3_trace_index.py
```

### 2. Regenerate the 683-example H1 slice

The exact H1 slice predates a later tier-5 change and must therefore be
generated at commit `b66325a`. The uncommitted raw inputs and generated trace
index remain available when switching commits.

```bash
git checkout b66325a

python scripts/regenerate_p0_dataset.py \
  --output-dir data/F3_synthetic_dhi_dataset_p0_radius6_15 \
  --n-per-tier 50 \
  --n-hard-negatives-per-kind 50 \
  --seed 1

python scripts/add_site_background_draws.py \
  --dataset-dir data/F3_synthetic_dhi_dataset_p0_radius6_15 \
  --n-extra-per-site 8 \
  --seed 101
```

### 3. Build H3 and obtain the frozen supplements

Return to the verified code, regenerate H3, and download the two supplements:

```bash
git checkout dataset-v2-1208

python scripts/regenerate_p0_h3_shallow_dataset.py
python scripts/download_dim_spot_supplement.py
python scripts/download_aziz_style_supplement.py
```

The supplement archives and their internal files are checksum-verified by the
downloaders:

- [Dim-spot supplement v1](https://github.com/ada-nfs25/F3_synthetic_dhi/releases/tag/dim-spot-supplement-v1)
- [Aziz-style supplement v1](https://github.com/ada-nfs25/F3_synthetic_dhi/releases/tag/aziz-style-supplement-v1)

These are attributed amplitude inputs, not stored features or blind-test
results. `build_v2_features.py` accepts both regenerated full-stack patches
and downloaded amplitude-only patches and validates the expected
683/296/201/28 source counts.

### 4. Recompute features, retrain, and verify

```bash
python scripts/build_v2_features.py
python scripts/train_xgb_v2.py
python scripts/verify_v2_reproduction.py
```

The final verification compares the reconstructed counts, features,
predictions, metrics, and models with the frozen expected result.
