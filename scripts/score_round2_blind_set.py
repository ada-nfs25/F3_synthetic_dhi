#!/usr/bin/env python3
"""
Score Aziz's round-2 blind patch set with BOTH frozen models - the
pre-registered primary (v2: P0-diversified data, P1-fixed features -
adaptive band_ratio, swept-lag doublet) and the declared secondary
(same P0 data, v1-style features - fixed band_ratio, lag-1 doublet),
per the freeze decision (round 2): primary decides the
pre-registered success/partial/miss outcome, secondary is reported
alongside as a clean ablation - nothing here decides which "wins";
that's what the blind result is for.

Decision rule: top-K by within-batch confidence rank, K = round(n_patches *
32/96) - matching the round-2 pre-registration's disclosed composition (32
tier-balanced DHI positives / 96 total), not a median split. Computed
independently per model, since primary/secondary can rank patches
differently.

Writes two separate schema-v1.1-compliant detection files (one per model),
not a combined non-standard format, so either can be scored the same way
round 1's were.
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
from src.dhi_pipeline.ratio_features import ratio_features_from_stack  # noqa: E402

PRIMARY_MODEL_PATH = REPO / 'data' / 'v2_features' / 'xgboost_dhi_14feature_v2.json'
SECONDARY_MODEL_PATH = REPO / 'data' / 'v2_features' / 'xgboost_dhi_14feature_v1style_secondary.json'
V1_STYLE_DOMINANT_FREQ_HZ = 60.0
DISCLOSED_POSITIVE_FRACTION = 32 / 96
"""From the round-2 pre-registration's disclosed blind-set composition (32
tier-balanced DHI positives / 96 total) - NOT the actual round-2 blind set
itself (never seen before scoring), so using it to set K is label-free with
respect to this specific batch and doesn't violate "nothing tuned against a
received blind set"."""

V2_FEATURE_COLS = [
    'envelope_mean_near_top_ratio', 'envelope_p90_near_top_ratio', 'envelope_maxabs_near_top_ratio',
    'sweetness_mean_near_top_ratio', 'sweetness_p90_near_top_ratio', 'sweetness_maxabs_near_top_ratio',
    'inst_freq_mean_near_top_ratio', 'inst_freq_p90_near_top_ratio', 'inst_freq_maxabs_near_top_ratio',
    'band_ratio_mean_near_top_ratio', 'band_ratio_p90_near_top_ratio', 'band_ratio_maxabs_near_top_ratio',
    'doublet_autocorrelation', 'signed_polarity',
]


def compute_both_feature_sets(full_amplitude, dt_s, t_mask):
    """
    Round 2 (Aziz, index.json note): "take your 125-sample window centred on
    center_time_ms AFTER computing attributes" - the opposite order from
    round 1 (crop to 500ms, then compute). Attributes here are Hilbert-
    transform/filter-based (envelope, inst_freq, band_ratio, ...), which have
    edge artifacts on a short window; computing them on the full 463-sample
    trace first gives real margin on both sides, then t_mask crops the
    resulting STACK, not the raw amplitude. channel_names/ordering is
    unaffected by cropping the time axis, so this is a straightforward swap
    of what gets cropped, not a different attribute pipeline.
    """
    from src.dhi_pipeline.attributes import compute_band_ratio

    full_stack, channel_names = compute_attribute_stack(full_amplitude, dt_s)
    full_v1_style_band_ratio = compute_band_ratio(
        full_amplitude, dt_s=dt_s, dominant_freq_hz=V1_STYLE_DOMINANT_FREQ_HZ)

    stack = full_stack[:, :, :, t_mask]
    v1_style_stack = stack.copy()
    v1_style_stack[channel_names.index('band_ratio')] = full_v1_style_band_ratio[:, :, t_mask]

    v2_features = ratio_features_from_stack(stack, channel_names)
    v1_style_features = ratio_features_from_stack(v1_style_stack, channel_names, doublet_lags=(1,))
    return v2_features, v1_style_features


def predict_blind_patch(sgy_path, center_time_ms, primary_model, secondary_model,
                         feature_names, time_extent_ms=500):
    with segyio.open(str(sgy_path), ignore_geometry=True) as f:
        n_traces = f.tracecount
        n_il = n_xl = int(round(n_traces ** 0.5))
        full_time_axis_ms = f.samples.astype(float)
        dt_ms = float(full_time_axis_ms[1] - full_time_axis_ms[0])
        raw = np.array([f.trace[i] for i in range(n_traces)]).reshape(n_il, n_xl, len(full_time_axis_ms))

    t_mask = (full_time_axis_ms >= center_time_ms - time_extent_ms / 2) & \
             (full_time_axis_ms <= center_time_ms + time_extent_ms / 2)
    full_amplitude = raw.astype(np.float32)

    v2_features, v1_style_features = compute_both_feature_sets(full_amplitude, dt_ms / 1000.0, t_mask)

    X_primary = pd.DataFrame([v2_features])[feature_names]
    X_secondary = pd.DataFrame([v1_style_features])[feature_names]
    primary_proba = float(primary_model.predict_proba(X_primary)[0, 1])
    secondary_proba = float(secondary_model.predict_proba(X_secondary)[0, 1])
    return primary_proba, secondary_proba


def top_k_decisions(confidences, k):
    """Direct top-K by confidence rank (K = round(n * disclosed positive
    fraction)), not a quantile-threshold indirection - matches the
    established "top-K/top-X%-by-rank" convention (round 1's recall_at_k,
    Aziz's own reverse-leg "top 40% by rank") directly rather than through
    an approximate quantile cutoff."""
    order = np.argsort(-np.asarray(confidences))
    is_positive = np.zeros(len(confidences), dtype=bool)
    is_positive[order[:k]] = True
    return is_positive


def score_one_model(results, confidences, label):
    k = round(len(confidences) * DISCLOSED_POSITIVE_FRACTION)
    decisions = top_k_decisions(confidences, k)
    scored = []
    for r, conf, is_dhi in zip(results, confidences, decisions):
        scored.append({
            'schema_version': '1.1',
            'detector_side': 'nora',
            'blind_id': r['blind_id'],
            'is_dhi': bool(is_dhi),
            'tier': None,
            'confidence': conf,
            'predicted_time_ms': r['center_time_ms'],
        })
        print(f"[{label}] {r['blind_id']}: confidence={conf:.4f} is_dhi={is_dhi}")
    return scored


def main():
    if len(sys.argv) != 2:
        print('usage: score_round2_blind_set.py <blind_dir (contains index.json + blind_*.sgy)>', file=sys.stderr)
        sys.exit(1)
    blind_dir = Path(sys.argv[1])
    index = json.loads((blind_dir / 'index.json').read_text())

    primary_model = XGBClassifier()
    primary_model.load_model(str(PRIMARY_MODEL_PATH))
    secondary_model = XGBClassifier()
    secondary_model.load_model(str(SECONDARY_MODEL_PATH))
    feature_names = list(primary_model.get_booster().feature_names)
    assert feature_names == V2_FEATURE_COLS == list(secondary_model.get_booster().feature_names), \
        'primary/secondary feature column order must match V2_FEATURE_COLS exactly'

    results, primary_confs, secondary_confs = [], [], []
    for entry in index['patches']:
        patch_id = entry['patch_id']
        center_time_ms = entry['center_time_ms']
        sgy_path = blind_dir / f'{patch_id}.sgy'
        p_conf, s_conf = predict_blind_patch(
            sgy_path, center_time_ms, primary_model, secondary_model, feature_names)
        results.append({'blind_id': patch_id, 'center_time_ms': center_time_ms})
        primary_confs.append(p_conf)
        secondary_confs.append(s_conf)

    k = round(len(results) * DISCLOSED_POSITIVE_FRACTION)
    print(f'\n{len(results)} patches, K={k} (top {DISCLOSED_POSITIVE_FRACTION:.1%} by rank, '
          f'per the disclosed round-2 composition)\n')

    primary_detections = score_one_model(results, primary_confs, 'PRIMARY (v2)')
    secondary_detections = score_one_model(results, secondary_confs, 'SECONDARY (v1-style)')

    primary_path = blind_dir / 'nora_primary_detections_round2.json'
    secondary_path = blind_dir / 'nora_secondary_detections_round2.json'
    primary_path.write_text(json.dumps(primary_detections, indent=2) + '\n')
    secondary_path.write_text(json.dumps(secondary_detections, indent=2) + '\n')
    print(f'\nwrote {primary_path}\nwrote {secondary_path}')


if __name__ == '__main__':
    main()
