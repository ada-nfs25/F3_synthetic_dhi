# F3 Synthetic DHI Generator

AI-assisted (Claude) pipeline for generating synthetic Direct Hydrocarbon
Indicator (DHI) examples on the F3 Demo seismic dataset, plus the DHI
detection model (feature extraction, XGBoost training, blind-test scoring)
built on top of that data. Outputs feed the graded IRP submission,
`irp-nfs25`. See `AI_USAGE.md` for what's AI-implemented vs. my own
intellectual contribution.

## What's here
- `src/dhi_pipeline/injection.py` - wedge-model reservoir injection (Ricker
  wavelet, petrophysics-derived reflection coefficients, tuning-thickness
  calibration, severity tiers), 3D horizon-conformant injection.
- `src/dhi_pipeline/dim_spot.py` - dim-spot positives (genuine host-
  reflectivity attenuation, not an additive wedge event).
- `src/dhi_pipeline/horizons.py` - matches Zenodo interpretation horizon
  picks to the F3 trace grid, for structural conformance.
- `src/dhi_pipeline/scenarios.py` - randomised scenario sampling (severity
  tiers, hard negatives) for dataset generation.
- `src/dhi_pipeline/attributes.py` - 8-channel attribute stack
  (`compute_attribute_stack`: envelope, instantaneous phase/frequency, RMS
  amplitude, sweetness, band ratio, local variance), deterministic from raw
  amplitude.
- `src/dhi_pipeline/ratio_features.py` - the 14 scalar ratio features built
  on top of the attribute stack that actually feed the XGBoost classifier.
- `src/dhi_pipeline/calibration.py` - per-volume rank/quantile decision
  calibration (top-K by within-batch rank, not a fixed global threshold).
- `utils/seismic_io.py` - segyio read helpers (inline/crossline/timeslice/
  sub-volume).
- `notebooks/synthetic_dhi_generation.ipynb` - full walkthrough: background
  patch calibration, severity tier design, amplitude calibration, and the
  3D horizon-conformant injection demo.
- `scripts/regenerate_p0_dataset.py` and
  `scripts/regenerate_p0_h3_shallow_dataset.py` - seeded regeneration of the
  H1 and H3 slices from public F3 inputs.
- `scripts/download_dim_spot_supplement.py` - download and verify the frozen
  201-example dim-spot supplement from the public
  [GitHub Release](https://github.com/ada-nfs25/F3_synthetic_dhi/releases/tag/dim-spot-supplement-v1).
- `scripts/download_aziz_style_supplement.py` - download and verify the
  frozen 28-example collaborator-generator supplement from the public
  [GitHub Release](https://github.com/ada-nfs25/F3_synthetic_dhi/releases/tag/aziz-style-supplement-v1).
- `scripts/regenerate_p0_dim_spot_dataset.py` and
  `scripts/regenerate_p0_aziz_style_dataset.py` - provenance-only historical
  integrations requiring collaborator-owned code that is not distributed;
  they are not part of the examiner reproduction route.
- `scripts/build_v2_features.py` - recompute the 14 features fresh from raw
  amplitude across the combined P0 dataset (not from stored attribute
  channels, which mix three code versions - see script docstring).
- `scripts/train_xgb_v2.py` - LOSO-evaluate and fit the frozen XGBoost
  models (primary: P1-fixed features; secondary: v1-style ablation).
- `results/golden/` and `scripts/verify_v2_reproduction.py` - small,
  committed reference artifacts and a strict verifier for reproducing the
  reported dataset counts, LOSO metrics, feature schema, and frozen model
  hashes without rebuilding the large patch corpus.
- `scripts/score_blind_round1.py`, `scripts/score_round2_blind_set.py`,
  `scripts/run_blind_predictions.py` - blind-exchange scoring against a
  collaborator's held-out patch sets.
- `scripts/export_amplitude_only_dataset.py` - amplitude-only dataset
  export (drops the 7 derived attribute channels, ~11x smaller; the
  channels are deterministically recomputable from amplitude via
  `compute_attribute_stack`).

## Detector development (P0-P2) and blind-exchange testing
Working with a collaborator (Aziz) doing an independent DHI-detection
project on a different F3 region, this repo went through several rounds of
cross-generator blind testing (round 1: frozen v1 detector on his blind set,
0.643 ROC-AUC vs. 0.820 internal LOSO; P0-P2: diversified the training data
to 1208 examples across H1/H3/dim_spot/aziz-style sources and fixed feature
bugs; round 2: pre-registered blind exchange, both directions - see
`blind_tests.md` for the full numbered history, not reproduced here). Round
3 (union-of-domains training + LODO validation) is in progress.

## Data dependencies
The required F3 Demo 2023 SEG-Y volume, interpretation horizons, and velocity
functions are not committed because of their size. See `data_raw/README.md`
for their public sources, exact expected paths, and a distinction between
functional inputs and exploratory notebook inputs.

The dataset scripts construct the F3 trace index automatically from the
SEG-Y volume when it is absent. It can also be built explicitly with
`scripts/build_f3_trace_index.py`.

## Reproducing the derived dataset and models

The verified code and golden artifacts are pinned at
[`dataset-v2-1208`](https://github.com/ada-nfs25/F3_synthetic_dhi/tree/dataset-v2-1208).

Two routes are available:

- a tested, lightweight examiner route that verifies the committed golden
  evaluation, expected metrics, feature schema, and frozen model identities;
- a full 40–50 GB route that regenerates H1/H3 from public F3 inputs,
  downloads the frozen dim-spot/Aziz-style supplements, recomputes all
  features, and reruns model training and evaluation.

See [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) for the complete commands,
expected output, public data requirements, and testing status of each route.

## AI assistance

Claude Code assisted with implementation and documentation throughout this
repository. See [AI_USAGE.md](AI_USAGE.md) for the complete scope and
attribution.