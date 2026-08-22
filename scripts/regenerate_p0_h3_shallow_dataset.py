#!/usr/bin/env python3
"""Generate a shallow-depth-band diversity supplement on horizon H3 - Aziz's
round-2 plan (ROUND2_PLAN.md P0 item 1): "inject across multiple horizons/
depth bands, including a shallow band (~1000-1250ms) matching the round-1
shift."

H1 (our only horizon used so far) sits at a mean two-way time of ~1654ms.
The F3 Demo dataset ships 9 interpreted horizons (H1-H9); H3 sits at
~1217ms - inside Aziz's requested band - and has real structural highs/lows
in both the existing train (150-345) and test (365-670) inline ranges (20/24
highs, 8/24 lows), so this is a genuinely structurally-conformant second
depth band, not a flat-time approximation.

This is a SEPARATE, additive dataset directory (not merged into the existing
P0 H1 dataset) - anything downstream that builds a training set for P1/P2
should concatenate this directory's labels with the H1 one. Kept separate
deliberately so the existing, already-validated H1 dataset stays untouched.

Reuses the exact same build_dataset/scenario machinery as the H1 regen
(regenerate_p0_dataset.py) - only the horizon and its structural highs/lows
differ - so tier5_resolved (added for P0 item 2) is generated here too, for
free, combining the two P0 diversity axes in one run.
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import pandas as pd
import segyio

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.dhi_pipeline.dataset import build_dataset  # noqa: E402
from src.dhi_pipeline.horizons import (  # noqa: E402
    HorizonSurface,
    build_coordinate_lookup,
    find_structural_highs,
    find_structural_lows,
    load_horizon_surface,
)
from src.dhi_pipeline.scenarios import FOOTPRINT_RADIUS_RANGE, TIER_RANGES  # noqa: E402
from utils.seismic_io import read_inline  # noqa: E402


def calibration(segy_path, velocity_path, iline_map, xlines):
    """Same calibration as regenerate_p0_dataset.py - deliberately duplicated
    rather than imported, matching that script's own precedent of standalone
    dataset-regen scripts (see regenerate_dummy_exchange.py). Calibration is
    survey-level (dt/velocity/dominant frequency), not horizon-specific, so
    this must produce the SAME numbers as the H1 regen - reusing the same
    reference window rather than deriving one from H3 keeps the two dataset
    chunks on identical calibration, which matters for combining them later.
    """
    il, xl_lo, xl_hi, t_lo, t_hi = 400, 1000, 1150, 650, 800
    import numpy as np
    with segyio.open(str(segy_path), ignore_geometry=True) as f:
        samples = f.samples.astype(float)
        dt_ms = float(samples[1] - samples[0])
        raw = read_inline(f, iline_map, xlines, il=il, n_samples=len(samples))
        header = f.header[iline_map[(il, (xl_lo + xl_hi) // 2)]]
        scalar = abs(header[segyio.TraceField.SourceGroupScalar]) or 1
        cdp_x = header[segyio.TraceField.CDP_X] / scalar
        cdp_y = header[segyio.TraceField.CDP_Y] / scalar

    time_sel = (samples >= t_lo) & (samples <= t_hi)
    traces = raw[xl_lo - xlines[0]:xl_hi - xlines[0] + 1, time_sel]
    tapered = (traces - traces.mean(axis=1, keepdims=True)) * np.hanning(traces.shape[1])
    freqs = np.fft.rfftfreq(512, d=dt_ms / 1000.0)
    spectrum = np.abs(np.fft.rfft(tapered, n=512, axis=1)).mean(axis=0)
    dominant_freq = float(freqs[np.argmax(spectrum * (freqs >= 5))])

    velocities = pd.read_csv(
        velocity_path, sep=r'\s+', skiprows=2,
        names=['cdp_x', 'cdp_y', 'time_ms', 'vrms', 'vint', 'vavg', 'depth_m'],
    )
    locations = velocities[['cdp_x', 'cdp_y']].drop_duplicates().to_numpy()
    nearest = locations[np.argmin(np.hypot(locations[:, 0] - cdp_x, locations[:, 1] - cdp_y))]
    block = velocities[(velocities.cdp_x == nearest[0]) & (velocities.cdp_y == nearest[1])]
    velocity = float(np.interp((t_lo + t_hi) / 2, block.time_ms, block.vint))
    return dt_ms, velocity, dominant_freq


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path,
                         default=REPO / 'data/F3_synthetic_dhi_dataset_p0_h3_shallow')
    parser.add_argument('--n-per-tier', type=int, default=15)
    parser.add_argument('--n-hard-negatives-per-kind', type=int, default=15)
    parser.add_argument('--seed', type=int, default=2)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error(f'output directory must be absent or empty: {args.output_dir}')

    raw = REPO / 'data_raw'
    segy_path = raw / 'Seismic_data.sgy'
    with (REPO / 'data/f3_trace_index.pkl').open('rb') as fh:
        index = pickle.load(fh)
    iline_map, inlines, xlines = index['iline_map'], index['inlines'], index['xlines']
    dt_ms, velocity, dominant_freq = calibration(
        segy_path, raw / 'Velocity_functions.txt', iline_map, xlines,
    )

    coords = build_coordinate_lookup(str(segy_path))
    surface = load_horizon_surface(
        str(raw / 'horizons/H3.xyz'), coords['ilxl_array'], coords['xy_array'],
    )
    horizon = HorizonSurface(surface)
    highs = find_structural_highs(surface)
    lows = find_structural_lows(surface, edge_margin=80)
    print(f'H3: mean time {surface.time_ms.mean():.1f}ms, '
          f'{len(highs)} structural highs, {len(lows)} structural lows')

    labels = build_dataset(
        output_dir=str(args.output_dir), segy_path=str(segy_path), iline_map=iline_map,
        inlines=inlines, xlines=xlines, horizon=horizon, dt_ms=dt_ms,
        velocity_mps=velocity, freq_hz=dominant_freq,
        train_inline_range=(150, 345), test_inline_range=(365, 670),
        structural_highs=highs, structural_lows=lows,
        n_per_tier=args.n_per_tier,
        n_hard_negatives_per_kind=args.n_hard_negatives_per_kind,
        n_background_per_site=1, seed=args.seed,
    )
    labels['horizon'] = 'H3'
    labels.to_parquet(args.output_dir / 'labels.parquet', index=False)
    labels.to_csv(args.output_dir / 'labels.csv', index=False)

    manifest = {
        'horizon': 'H3',
        'horizon_mean_time_ms': float(surface.time_ms.mean()),
        'seed': args.seed,
        'footprint_radius_range_traces': list(FOOTPRINT_RADIUS_RANGE),
        'dt_ms': dt_ms,
        'velocity_mps': velocity,
        'dominant_frequency_hz': dominant_freq,
        'n_examples': len(labels),
        'counts': {
            '|'.join(map(str, key)): int(value)
            for key, value in labels.groupby(['split', 'kind']).size().items()
        },
        'tiers': list(TIER_RANGES),
        'note': 'Additive P0 diversity supplement (depth band). Concatenate '
                'with the H1 P0 dataset for P1/P2 training, do not use alone.',
    }
    (args.output_dir / 'generation_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
