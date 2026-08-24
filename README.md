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
Raw inputs (F3 Demo 2023 SEG-Y volume, Zenodo interpretation horizons/masks,
velocity functions, AI cube, well logs, and the real bright-spot pick) are
DVC-tracked in `data_raw/` - see `data_raw/README.md` for sources, sizes,
and which files are actually required by `build_dataset()` vs.
exploratory-only. Not committed to git directly; pull with:

```bash
pip install -r requirements.txt
dvc pull
```

**Access:** the DVC remote (`.dvc/config`) is this project's storage on
Imperial's CX3 HPC cluster, reached over SSH
(`ssh://.../rds/general/user/nfs25/home/dvc-storage`). CX3 doesn't support
SSH key auth, so `dvc pull` will prompt for your own CX3 password. This
works for anyone with an Imperial HPC account **and** read access to that
directory - currently restricted to the `hpc-ggorman` Unix group. Everything
in `data_raw/` is also available directly from its original public sources
(see `data_raw/README.md`) if you don't have CX3 access.

`data/*.pkl` are regeneratable local caches (trace geometry, horizon
coordinate lookup) - gitignored, not checked in, rebuilt automatically by
the notebooks.
