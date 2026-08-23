#!/usr/bin/env python3
"""Run the frozen 14-feature XGBoost detector on Aziz's blind patch batch.

Reads raw amplitude directly from each anonymized SEG-Y patch (headers give
no real coordinates, only renumbered 1..96 il/xl - see index.json), crops to
the 125-sample window centred on the given center_time_ms *before* computing
attributes (matches training - see dataset.py's generate_example, and the
crop-order check earlier in this exchange), then runs the frozen model.
Never touches data_raw/Seismic_data.sgy or any other real-F3-derived file.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import segyio
from xgboost import XGBClassifier

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.dhi_pipeline.attributes import compute_attribute_stack  # noqa: E402
from src.dhi_pipeline.calibration import rank_quantile  # noqa: E402

FEATURE_CHANNELS = ['envelope', 'sweetness', 'inst_freq', 'band_ratio']
NEAR_TOP_HALF_MS = 60
DOUBLET_SPATIAL_RADIUS = 5
DOUBLET_TIME_HALFWIDTH = 15
SIGNED_POLARITY_SPATIAL_RADIUS = 2


def ratio_features_from_stack(stack, channel_names):
    """Verbatim copy of the notebook's version (irp-nfs25/dhi_xgb_detector.ipynb,
    cell defining ratio_features_from_stack) - kept identical deliberately, not
    reimplemented, so there's no risk of the two drifting apart."""
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


def predict_blind_patch(sgy_path, center_time_ms, model, feature_names, time_extent_ms=500):
    with segyio.open(str(sgy_path), ignore_geometry=True) as f:
        n_traces = f.tracecount
        n_il = n_xl = int(round(n_traces ** 0.5))
        full_time_axis_ms = f.samples.astype(float)  # authoritative, straight from the file
        dt_ms = float(full_time_axis_ms[1] - full_time_axis_ms[0])
        raw = np.array([f.trace[i] for i in range(n_traces)]).reshape(n_il, n_xl, len(full_time_axis_ms))

    t_mask = (full_time_axis_ms >= center_time_ms - time_extent_ms / 2) & \
             (full_time_axis_ms <= center_time_ms + time_extent_ms / 2)
    cropped = raw[:, :, t_mask].astype(np.float32)

    stack, channel_names = compute_attribute_stack(cropped, dt_ms / 1000.0)
    features = ratio_features_from_stack(stack, channel_names)

    X_blind = pd.DataFrame([features])[feature_names]  # explicit reindex to training's exact column order
    proba = float(model.predict_proba(X_blind)[0, 1])
    return proba


def main():
    blind_dir = Path.home() / 'Desktop' / 'blind_exchange_ak7024'
    index = json.loads((blind_dir / 'index.json').read_text())

    model = XGBClassifier()
    model.load_model(str(REPO.parent / 'irp-nfs25' / 'models' / 'xgboost_dhi_14feature_loso_v1.json'))
    feature_names = list(model.get_booster().feature_names)
    print(f'loaded model, {len(feature_names)} features: {feature_names}')

    results = []
    for entry in index['patches']:
        patch_id = entry['patch_id']
        center_time_ms = entry['center_time_ms']
        sgy_path = blind_dir / f'{patch_id}.sgy'

        proba = predict_blind_patch(sgy_path, center_time_ms, model, feature_names)
        results.append({
            'schema_version': '1.1',
            'detector_side': 'nora',
            'blind_id': patch_id,
            'tier': None,
            'confidence': proba,
            'predicted_time_ms': center_time_ms,
        })

    # Round 2, P1 item 3: the raw 0.5 threshold has failed twice (LOSO, round-1
    # blind) on per-volume score offsets - decide is_dhi from the within-batch
    # rank instead (median split: top half of this batch by score), and keep
    # the raw confidence in the output for audit rather than discarding it.
    quantiles = rank_quantile([r['confidence'] for r in results])
    for result, q in zip(results, quantiles):
        result['rank_quantile'] = float(q)
        result['is_dhi'] = bool(q >= 0.5)
        print(f"{result['blind_id']}: confidence={result['confidence']:.4f} "
              f"rank_quantile={q:.3f} is_dhi={result['is_dhi']}")

    out_path = blind_dir / 'nora_detections_round1.json'
    out_path.write_text(json.dumps(results, indent=2) + '\n')
    n_pos = sum(r['is_dhi'] for r in results)
    print(f'\n{len(results)} patches scored, {n_pos} flagged is_dhi, wrote {out_path}')


if __name__ == '__main__':
    main()
