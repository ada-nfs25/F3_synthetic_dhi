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

**Caveat:** the DVC remote (`.dvc/config`) is a local path inside this
machine's synced Imperial College OneDrive folder, so `dvc pull` currently
only works for someone with access to that same shared folder - not a cold
clone from an arbitrary machine.

`data/*.pkl` are regeneratable local caches (trace geometry, horizon
coordinate lookup) - gitignored, not checked in, rebuilt automatically by
the notebooks.
