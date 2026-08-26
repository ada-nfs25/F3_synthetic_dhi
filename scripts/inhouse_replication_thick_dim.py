#!/usr/bin/env python3
# Implementation developed with AI (Claude Code) assistance - see AI_USAGE.md.
"""In-house replication of round 1's two suspected failure modes on our own
F3 volume, scored with the frozen v1 detector - Aziz's round-2 plan (see
share_nora_round2/ROUND2_PLAN.md P0)'s "one cheap confirmation worth an
afternoon" before touching blind data or retraining anything:

    inject thick (>=1.5x tuning) beds and dim spots on your own volume and
    run frozen v1 on them. If recall collapses there too, the diagnosis is
    confirmed without touching blind data.

Thick beds: our own injection pipeline (src/dhi_pipeline/injection.py),
thickness explicitly swept from 1.5x-3.5x our measured tuning thickness
(tier4_obvious's continuous range already reaches ~1.56x-2.55x; this pushes
further to make the "resolved bed" regime unambiguous), tier4-style cues
(full gas contrast, flat spot, polarity reversal).

Dim spots: no such anomaly exists in our own injection.py at all, so this
borrows Aziz's dim_spot mechanic (external/aziz_dhi_lib/primitives.py -
genuine host-reflectivity attenuation, not an additive event) from the
round-2 bundle, applied with OUR OWN calibration (RC constants, measured
tuning thickness, Ricker wavelet) rather than his - isolating the mechanism
under test from a calibration swap. See HUMAN_COLLABORATION.md for the
provenance of external/aziz_dhi_lib.

Both are scored with the same frozen 14-feature model and the same
feature/crop pipeline used for round 1 (scripts/run_blind_predictions.py),
so results are directly comparable to that run.
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import segyio
from xgboost import XGBClassifier

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'external' / 'aziz_dhi_lib' / '..'))  # external/aziz_dhi_lib is itself the `dhi_lib` package dir's contents

from src.dhi_pipeline.attributes import compute_attribute_stack  # noqa: E402
from src.dhi_pipeline.horizons import (  # noqa: E402
    HorizonSurface, build_coordinate_lookup, find_structural_highs, load_horizon_surface,
)
from src.dhi_pipeline.injection import (  # noqa: E402
    RC_GAS_SAND, RC_WATER_SAND, V_GAS_SAND, estimate_amplitude_scale,
    inject_dhi_anomaly_3d, ricker_wavelet, wedge_peak_amplitude,
)

sys.path.insert(0, str(REPO / 'external'))
from aziz_dhi_lib.primitives import build_anomaly  # noqa: E402

# --- verbatim from scripts/run_blind_predictions.py: must match training exactly ---
FEATURE_CHANNELS = ['envelope', 'sweetness', 'inst_freq', 'band_ratio']
NEAR_TOP_HALF_MS = 60
DOUBLET_SPATIAL_RADIUS = 5
DOUBLET_TIME_HALFWIDTH = 15
SIGNED_POLARITY_SPATIAL_RADIUS = 2


def ratio_features_from_stack(stack, channel_names):
    """Verbatim copy - see scripts/run_blind_predictions.py."""
    n_samples = stack.shape[-1]
    dt_ms = 4.0
    mid = n_samples // 2

    near_top_half = int(round(NEAR_TOP_HALF_MS / dt_ms))
    windows = {'near_top': (max(mid - near_top_half, 0), min(mid + near_top_half, n_samples))}

    features = {}
    for ch in FEATURE_CHANNELS:
        arr = stack[channel_names.index(ch)]
        for window_name, (lo, hi) in windows.items():
            window = arr[..., lo:hi]
            for stat_name, stat_fn in [('mean', np.mean), ('p90', lambda a: np.percentile(a, 90)), ('maxabs', lambda a: np.max(np.abs(a)))]:
                window_stat = stat_fn(window)
                whole_stat = stat_fn(arr)
                features[f'{ch}_{stat_name}_{window_name}_ratio'] = window_stat / (whole_stat + 1e-10)

    amplitude = stack[channel_names.index('amplitude')]
    ni, nx, nt = amplitude.shape
    mi, mx, mt = ni // 2, nx // 2, nt // 2

    central = amplitude[mi - DOUBLET_SPATIAL_RADIUS:mi + DOUBLET_SPATIAL_RADIUS + 1,
                         mx - DOUBLET_SPATIAL_RADIUS:mx + DOUBLET_SPATIAL_RADIUS + 1,
                         mt - DOUBLET_TIME_HALFWIDTH:mt + DOUBLET_TIME_HALFWIDTH + 1]
    traces = central.reshape(-1, 31)
    traces = traces - np.median(traces, axis=1, keepdims=True)
    scale = np.sqrt(np.sum(traces ** 2, axis=1)) + 1e-10
    acf_lag1 = np.sum(traces[:, :-1] * traces[:, 1:], axis=1) / scale ** 2
    features['doublet_autocorrelation'] = 1.0 - np.median(acf_lag1)

    centre_vals = amplitude[mi - SIGNED_POLARITY_SPATIAL_RADIUS:mi + SIGNED_POLARITY_SPATIAL_RADIUS + 1,
                             mx - SIGNED_POLARITY_SPATIAL_RADIUS:mx + SIGNED_POLARITY_SPATIAL_RADIUS + 1, mt]
    features['signed_polarity'] = np.sum(centre_vals) / (np.sum(np.abs(centre_vals)) + 1e-10)

    return features


def score_patch(raw_patch, dt_ms, model, feature_names):
    stack, channel_names = compute_attribute_stack(raw_patch, dt_ms / 1000.0)
    features = ratio_features_from_stack(stack, channel_names)
    X = pd.DataFrame([features])[feature_names]
    return float(model.predict_proba(X)[0, 1])


def extract_patch(f, iline_map, il_center, xl_center, center_time_ms, samples_ms,
                   il_extent=96, xl_extent=96, time_extent_ms=500):
    il_lo, il_hi = int(il_center - il_extent // 2), int(il_center + il_extent // 2)
    xl_lo, xl_hi = int(xl_center - xl_extent // 2), int(xl_center + xl_extent // 2)
    inline_axis = np.arange(il_lo, il_hi)
    xl_axis = np.arange(xl_lo, xl_hi)
    raw = np.full((len(inline_axis), len(xl_axis), len(samples_ms)), np.nan, dtype=np.float32)
    for i, il in enumerate(inline_axis):
        for j, xl in enumerate(xl_axis):
            idx = iline_map.get((int(il), int(xl)))
            if idx is not None:
                raw[i, j] = f.trace[idx]

    t_mask = (samples_ms >= center_time_ms - time_extent_ms / 2) & \
             (samples_ms <= center_time_ms + time_extent_ms / 2)
    raw_patch = np.nan_to_num(raw[:, :, t_mask], nan=0.0)
    patch_time_axis_ms = samples_ms[t_mask]
    return raw_patch, inline_axis, xl_axis, patch_time_axis_ms


def local_horizon_grid_s(horizon, inline_axis, xl_axis, patch_time_origin_ms):
    """dhi_lib's build_anomaly indexes its `volume` argument's time axis from
    sample 0 = t_s 0.0 (it's designed for a full, zero-origin survey volume -
    see inject.apply_catalog). Our patches are cropped to a local 500ms
    window starting at patch_time_origin_ms, so horizon times must be
    rebased to that same local origin or every event lands far outside the
    patch's actual sample range and silently no-ops (empty slice)."""
    grid_ms = np.array([[horizon.time_at(int(il), int(xl)) for xl in xl_axis] for il in inline_axis])
    return (grid_ms - patch_time_origin_ms) / 1000.0


def main():
    raw_dir = REPO / 'data_raw'
    segy_path = raw_dir / 'Seismic_data.sgy'
    with (REPO / 'data/f3_trace_index.pkl').open('rb') as fh:
        index = pickle.load(fh)
    iline_map = index['iline_map']

    manifest = json.loads((REPO / 'data/F3_synthetic_dhi_dataset_p0_radius6_15/generation_manifest.json').read_text())
    dt_ms, velocity_mps, freq_hz = manifest['dt_ms'], manifest['velocity_mps'], manifest['dominant_frequency_hz']
    dt_s = dt_ms / 1000.0

    coords = build_coordinate_lookup(str(segy_path))
    surface = load_horizon_surface(str(raw_dir / 'horizons/H1.xyz'), coords['ilxl_array'], coords['xy_array'])
    horizon = HorizonSurface(surface)
    highs = find_structural_highs(surface)
    print(f'{len(highs)} structural highs available')

    # empirical tuning thickness (same definition as SEVERITY_TIERS' ~7.07m comment,
    # recomputed against this run's own calibrated velocity/frequency rather than
    # hardcoded, since regenerate_p0_dataset.py's calibration is data-dependent)
    sweep_m = np.linspace(1.0, 20.0, 400)
    peak_amps = [wedge_peak_amplitude(th, velocity_mps, RC_GAS_SAND, freq_hz) for th in sweep_m]
    tuning_thickness_m = float(sweep_m[np.argmax(peak_amps)])
    tuning_thickness_s = 2 * tuning_thickness_m / velocity_mps
    print(f'empirical tuning thickness: {tuning_thickness_m:.2f} m ({tuning_thickness_s * 1000:.2f} ms TWT)')

    model = XGBClassifier()
    model.load_model(str(REPO.parent / 'irp-nfs25' / 'models' / 'xgboost_dhi_14feature_loso_v1.json'))
    feature_names = list(model.get_booster().feature_names)

    rng = np.random.default_rng(20260822)
    sites = highs.sample(n=min(8, len(highs)), random_state=20260822).reset_index(drop=True)

    results = []

    with segyio.open(str(segy_path), ignore_geometry=True) as f:
        samples_ms = f.samples.astype(float)

        # --- thick beds: our own pipeline, explicit thickness sweep >=1.5x tuning ---
        for thickness_frac in (1.5, 2.0, 2.5, 3.0, 3.5):
            thickness_m = thickness_frac * tuning_thickness_m
            for _, site in sites.iterrows():
                il_center, xl_center = int(site['inline']), int(site['crossline'])
                center_time_ms = horizon.time_at(il_center, xl_center)
                raw_patch, inline_axis, xl_axis, patch_time_ms = extract_patch(
                    f, iline_map, il_center, xl_center, center_time_ms, samples_ms)
                amp_scale = estimate_amplitude_scale(raw_patch, reference_rc=0.05)
                try:
                    injected, twt_thickness_ms = inject_dhi_anomaly_3d(
                        raw_patch, patch_time_ms, inline_axis, xl_axis, horizon,
                        thickness_m=thickness_m, velocity_mps=velocity_mps,
                        reflection_coefficient=RC_GAS_SAND * 1.0, freq_hz=freq_hz,
                        il_center=il_center, xl_center=xl_center, il_radius=10, xl_radius=10,
                        flat_spot=True, polarity_reversal=True, amplitude_scale=amp_scale,
                    )
                except ValueError as e:
                    print(f'  skip thick {thickness_frac}x @ ({il_center},{xl_center}): {e}')
                    continue
                proba = score_patch(injected, dt_ms, model, feature_names)
                results.append(dict(kind='thick_bed', thickness_frac_of_tuning=thickness_frac,
                                     thickness_m=thickness_m, il_center=il_center, xl_center=xl_center,
                                     confidence=proba, is_dhi=proba >= 0.5))
                print(f'thick {thickness_frac}x tuning @ ({il_center},{xl_center}): confidence={proba:.4f}')

        # --- dim spots: Aziz's dim_spot mechanic, our own calibration ---
        wavelet = ricker_wavelet(freq_hz, dt_s, length_ms=120)
        for tier in (1, 2, 3, 4):
            for _, site in sites.iterrows():
                il_center, xl_center = int(site['inline']), int(site['crossline'])
                center_time_ms = horizon.time_at(il_center, xl_center)
                raw_patch, inline_axis, xl_axis, patch_time_ms = extract_patch(
                    f, iline_map, il_center, xl_center, center_time_ms, samples_ms)
                horizon_grid_s = local_horizon_grid_s(horizon, inline_axis, xl_axis, patch_time_ms[0])
                local_site = (raw_patch.shape[0] // 2, raw_patch.shape[1] // 2)
                try:
                    delta = build_anomaly(
                        kind='dim_spot', tier=tier, volume=raw_patch, dt_s=dt_s,
                        horizon_twt_s=horizon_grid_s, site=local_site, size_ilxl=(20, 20),
                        rc_gas=RC_GAS_SAND, rc_brine=RC_WATER_SAND, thickness_s=tuning_thickness_s,
                        wavelet=wavelet, rng=np.random.default_rng(int(rng.integers(0, 2**31))),
                        tuning_thickness_s=tuning_thickness_s, apply_sag=True,
                        v_gas_mps=V_GAS_SAND, v_background_mps=velocity_mps,
                    )
                except ValueError as e:
                    print(f'  skip dim_spot tier{tier} @ ({il_center},{xl_center}): {e}')
                    continue
                injected = raw_patch + delta
                proba = score_patch(injected, dt_ms, model, feature_names)
                results.append(dict(kind='dim_spot', tier=tier, il_center=il_center, xl_center=xl_center,
                                     confidence=proba, is_dhi=proba >= 0.5))
                print(f'dim_spot tier{tier} @ ({il_center},{xl_center}): confidence={proba:.4f}')

    df = pd.DataFrame(results)
    out_path = REPO / 'data' / 'inhouse_replication_thick_dim_results.csv'
    df.to_csv(out_path, index=False)

    print('\n=== summary ===')
    thick = df[df.kind == 'thick_bed']
    dim = df[df.kind == 'dim_spot']
    print(f'thick beds: {thick.is_dhi.sum()}/{len(thick)} flagged (recall={thick.is_dhi.mean():.3f})')
    print(thick.groupby('thickness_frac_of_tuning').is_dhi.agg(['mean', 'count']))
    print(f'\ndim spots: {dim.is_dhi.sum()}/{len(dim)} flagged (recall={dim.is_dhi.mean():.3f})')
    print(dim.groupby('tier').is_dhi.agg(['mean', 'count']))
    print(f'\nwrote {out_path}')


if __name__ == '__main__':
    main()
