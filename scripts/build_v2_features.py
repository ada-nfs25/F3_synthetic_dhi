#!/usr/bin/env python3
"""
P2: combine the P0-diversified datasets (H1, H3, dim_spot) and compute the
v2 feature set fresh from each patch's raw amplitude.

Deliberately does NOT reuse each patch's stored `attribute_stack` (band_ratio
channel specifically): H1/H3/dim_spot were generated at three different
points in this round's work, against three different versions of
compute_band_ratio (H1: the original fixed 15/45Hz; H3: an early buggy
full-patch-window auto-estimate; dim_spot: after the narrow-window bugfix -
see attributes.py's compute_band_ratio/estimate_dominant_frequency
docstrings). Trusting the stored channel would silently mix three
differently-defined band_ratios into one training set, exactly the kind of
site/generator leakage this round's work is trying to remove. Recomputing
from the raw `amplitude` channel (unaffected by any of that - it's the
survey signal, stored as generated) with the CURRENT code gives every patch
a consistently-defined feature regardless of when it was generated.

Checkpointed like the v1 notebook's own feature cache (irp-nfs25/notebooks/
dhi_xgb_detector.ipynb): writes the running table after every patch, so an
interrupted run resumes instead of restarting - this session has already
hit disk-full and mid-run crashes more than once.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.dhi_pipeline.attributes import compute_attribute_stack, compute_band_ratio  # noqa: E402
from src.dhi_pipeline.ratio_features import ratio_features_from_stack  # noqa: E402

V1_STYLE_DOMINANT_FREQ_HZ = 60.0
"""compute_band_ratio's low_frac=0.25/high_frac=0.75 defaults reproduce the
old fixed 15/45Hz bands exactly at dominant_freq_hz=60.0 (this survey's own
measured value) - see attributes.compute_band_ratio's docstring. Passing
this explicitly, instead of auto-estimating, gives the v1-style band_ratio
channel for the ablation below without needing a second code path."""

DATASETS = {
    'H1': 'F3_synthetic_dhi_dataset_p0_radius6_15',
    'H3': 'F3_synthetic_dhi_dataset_p0_h3_shallow',
    'dim_spot': 'F3_synthetic_dhi_dataset_p0_dim_spot',
    'aziz_style': 'F3_synthetic_dhi_dataset_p0_aziz_style',
}
OUT_DIR = REPO / 'data' / 'v2_features'
FEATURES_CACHE_PATH = OUT_DIR / 'ratio_features_v2_cache.parquet'
CACHE_KEY_PATH = OUT_DIR / 'ratio_features_v2_cache_key.json'
CACHE_KEY = dict(feature_version='v2_p1fixed_14feature_plus_v1_style_ablation',
                  doublet_lags=list(range(1, 6)), v1_style_dominant_freq_hz=V1_STYLE_DOMINANT_FREQ_HZ)


def load_combined_labels():
    frames = []
    for source, dirname in DATASETS.items():
        labels = pd.read_parquet(REPO / 'data' / dirname / 'labels.parquet')
        labels = labels.copy()
        labels['dataset_source'] = source
        labels['patch_path'] = str(REPO / 'data' / dirname / 'patches') + '/' + labels['patch_file']
        # global uniqueness across sources: local example_id collides between datasets
        labels['global_id'] = source + ':' + labels['example_id'].astype(str)
        frames.append(labels)
    combined = pd.concat(frames, ignore_index=True, sort=False)

    # site identity for LOSO grouping: the true structural site where tracked
    # (H1's later jittered background draws), else the row's own centre - see
    # module docstring / irp-nfs25 notebook's site_id construction, which this
    # generalises (v1's LOSO cell instead excluded the jittered rows entirely
    # since a plain il_center/xl_center site_id can't group them correctly).
    if 'site_il_center' not in combined.columns:
        combined['site_il_center'] = np.nan
        combined['site_xl_center'] = np.nan
    site_il = combined['site_il_center'].fillna(combined['il_center'])
    site_xl = combined['site_xl_center'].fillna(combined['xl_center'])
    combined['site_id'] = combined['dataset_source'] + ':' + site_il.astype(str) + '_' + site_xl.astype(str)

    return combined


def compute_patch_features(patch_path):
    """
    Returns the v2 candidate 14 features (adaptive band_ratio, swept-lag
    doublet) plus a v1_style_-prefixed ablation set (fixed-60Hz band_ratio,
    lag-1 doublet - i.e. v1's exact feature definitions) computed on the
    SAME patch, so LOSO can compare old-vs-new feature definitions fairly on
    the new P0-diversified data rather than on v1's original thin-bed-only
    training set (Aziz, ROUND2_PLAN.md P1 item 1: "your earlier rejection
    was measured on the thin-bed-only training set... re-evaluate on the
    diversified set from P0, not the old one").
    """
    data = np.load(patch_path)
    amplitude = data['attribute_stack'][list(data['channel_names']).index('amplitude')]

    stack, channel_names = compute_attribute_stack(amplitude, dt_s=0.004)
    v2_features = ratio_features_from_stack(stack, channel_names)

    v1_style_stack = stack.copy()
    v1_style_stack[channel_names.index('band_ratio')] = compute_band_ratio(
        amplitude, dt_s=0.004, dominant_freq_hz=V1_STYLE_DOMINANT_FREQ_HZ)
    v1_style_features = ratio_features_from_stack(v1_style_stack, channel_names, doublet_lags=(1,))

    return {**v2_features, **{f'v1style_{k}': v for k, v in v1_style_features.items()}}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    labels = load_combined_labels()
    print(f'{len(labels)} combined examples: '
          f'{labels.groupby("dataset_source").size().to_dict()}')
    print(f'{labels.site_id.nunique()} unique sites for LOSO grouping')

    cached_key = json.loads(CACHE_KEY_PATH.read_text()) if CACHE_KEY_PATH.exists() else None
    if cached_key != CACHE_KEY and FEATURES_CACHE_PATH.exists():
        print('feature config changed since last run, starting a fresh table')
        FEATURES_CACHE_PATH.unlink()

    features = (pd.read_parquet(FEATURES_CACHE_PATH) if FEATURES_CACHE_PATH.exists()
                else pd.DataFrame(columns=['global_id']))
    done_ids = set(features['global_id'])
    remaining = labels[~labels['global_id'].isin(done_ids)]
    print(f'{len(done_ids)} already cached, {len(remaining)} remaining')

    for i, (_, row) in enumerate(remaining.iterrows()):
        result = compute_patch_features(row['patch_path'])
        features = pd.concat(
            [features, pd.DataFrame([dict(global_id=row['global_id'], **result)])], ignore_index=True)
        features.to_parquet(FEATURES_CACHE_PATH, index=False)
        CACHE_KEY_PATH.write_text(json.dumps(CACHE_KEY))
        if (i + 1) % 100 == 0 or (i + 1) == len(remaining):
            print(f'  {i + 1}/{len(remaining)} done ({len(features)}/{len(labels)} total)')

    combined = labels.merge(features, on='global_id', how='inner')
    if len(combined) != len(labels):
        raise RuntimeError(f'{len(labels) - len(combined)} examples missing features after merge')

    out_path = OUT_DIR / 'v2_combined_features.parquet'
    combined.to_parquet(out_path, index=False)
    print(f'\nwrote {len(combined)} examples x features to {out_path}')


if __name__ == '__main__':
    main()
