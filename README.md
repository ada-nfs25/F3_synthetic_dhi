# F3 Synthetic DHI Generator

AI-assisted (Claude) pipeline for generating synthetic Direct Hydrocarbon
Indicator (DHI) examples on the F3 Demo seismic dataset, for use as training/
test data in a separate model-development repo. Built with supervisor
approval to use AI assistance for the synthetic data generation code itself
- the detection/classification model built on top of this data is separate,
independently-authored work.

## What's here
- `src/dhi_pipeline/injection.py` - wedge-model reservoir injection (Ricker
  wavelet, petrophysics-derived reflection coefficients, tuning-thickness
  calibration, severity tiers), 3D horizon-conformant injection.
- `src/dhi_pipeline/horizons.py` - matches Zenodo interpretation horizon
  picks to the F3 trace grid, for structural conformance.
- `utils/seismic_io.py` - segyio read helpers (inline/crossline/timeslice/
  sub-volume).
- `notebooks/synthetic_dhi_generation.ipynb` - full walkthrough: background
  patch calibration (dominant frequency, tuning thickness), severity tier
  design, amplitude calibration, and the 3D horizon-conformant injection
  demo.

## Data dependencies
Reads directly from the F3 Demo 2023 dataset and Zenodo interpretation
labels (paths hardcoded in the notebook, same locations as the companion
IRP repo's `data/README.md`). `data/*.pkl` are regeneratable local caches
(trace geometry, horizon coordinate lookup) - gitignored, not checked in.

## Status
Proof-of-concept stage: one location, one severity tier, verified against
real data. Not yet built: randomised multi-location/multi-tier sampling,
non-conformant hard negatives, spatial train/test partitioning, or a
finalised patch+label dataset export - see the notebook's closing section
for the planned next steps.
