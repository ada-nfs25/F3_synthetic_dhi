#!/usr/bin/env python3
"""Generate dim_spot positive examples - Aziz's round-2 plan (ROUND2_PLAN.md
P0 item 3): "brightness-correlated features ... need counter-examples where
the anomaly is an amplitude decrease."

Uses src/dhi_pipeline/dim_spot.py (Aziz's dim_spot attenuation mechanic from
external/aziz_dhi_lib, our own calibration - see that module's docstring),
across both horizons now available (H1 and the new H3 shallow band from
regenerate_p0_h3_shallow_dataset.py), tiers 1-4, both train/test inline
splits.

Separate, additive dataset directory - concatenate with the H1 and H3
datasets' labels for P1/P2 training. Does not generate its own background
controls: paired backgrounds at the same real structural-high sites already
exist in the H1/H3 datasets (a background patch is a valid clean control
regardless of which positive kind happens to share its site), so generating
new ones here would be redundant.
"""
import json
import shutil
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import segyio

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.dhi_pipeline.dim_spot import generate_dim_spot_example  # noqa: E402
from src.dhi_pipeline.horizons import (  # noqa: E402
    HorizonSurface, build_coordinate_lookup, find_structural_highs, load_horizon_surface,
)
from src.dhi_pipeline.injection import RC_GAS_SAND, RC_WATER_SAND, V_GAS_SAND, wedge_peak_amplitude  # noqa: E402
from utils.seismic_io import load_or_build_trace_index  # noqa: E402

N_PER_TIER_PER_SPLIT = 15
SPLITS = {'train': (150, 345), 'test': (365, 670)}
MIN_FREE_BYTES = 2 * 1024 ** 3
"""Stop gracefully (keep what's written) once free disk space on patches_dir's
filesystem drops below this, instead of crashing mid-write - see
dataset.build_dataset's own min_free_bytes docstring for why this exists."""


def empirical_tuning_thickness_m(velocity_mps, freq_hz):
    sweep_m = np.linspace(1.0, 20.0, 400)
    peak_amps = [wedge_peak_amplitude(th, velocity_mps, RC_GAS_SAND, freq_hz) for th in sweep_m]
    return float(sweep_m[np.argmax(peak_amps)])


def read_subvolume(f, iline_map, inline_axis, xl_axis, n_samples):
    out = np.full((len(inline_axis), len(xl_axis), n_samples), np.nan)
    for i, il in enumerate(inline_axis):
        for j, xl in enumerate(xl_axis):
            idx = iline_map.get((int(il), int(xl)))
            if idx is not None:
                out[i, j] = f.trace[idx]
    return out


def main():
    raw_dir = REPO / 'data_raw'
    segy_path = raw_dir / 'Seismic_data.sgy'
    output_dir = REPO / 'data/F3_synthetic_dhi_dataset_p0_dim_spot'
    patches_dir = output_dir / 'patches'
    patches_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((REPO / 'data/F3_synthetic_dhi_dataset_p0_radius6_15/generation_manifest.json').read_text())
    dt_ms, velocity_mps, freq_hz = manifest['dt_ms'], manifest['velocity_mps'], manifest['dominant_frequency_hz']
    dt_s = dt_ms / 1000.0
    tuning_thickness_m = empirical_tuning_thickness_m(velocity_mps, freq_hz)
    tuning_thickness_s = 2 * tuning_thickness_m / velocity_mps
    print(f'empirical tuning thickness: {tuning_thickness_m:.2f}m ({tuning_thickness_s * 1000:.2f}ms TWT)')

    index = load_or_build_trace_index(
        segy_path,
        REPO / 'data/f3_trace_index.pkl',
    )
    iline_map, inlines, xlines = index['iline_map'], index['inlines'], index['xlines']
    coords = build_coordinate_lookup(str(segy_path))

    horizons = {}
    for name in ('H1', 'H3'):
        surface = load_horizon_surface(str(raw_dir / f'horizons/{name}.xyz'), coords['ilxl_array'], coords['xy_array'])
        horizons[name] = (HorizonSurface(surface), find_structural_highs(surface))

    rng = np.random.default_rng(3)
    rows = []
    example_id = 0
    skip_counts = {}
    low_disk_stop = False

    with segyio.open(str(segy_path), ignore_geometry=True) as f:
        samples_ms = f.samples.astype(float)
        for horizon_name, (horizon, highs) in horizons.items():
            if low_disk_stop:
                break
            for split_name, (il_lo, il_hi) in SPLITS.items():
                if low_disk_stop:
                    break
                split_highs = highs[highs.inline.between(il_lo, il_hi)].reset_index(drop=True)
                if len(split_highs) == 0:
                    print(f'  skip {horizon_name}/{split_name}: no structural highs in range')
                    continue

                il_extent = xl_extent = 96
                cache_il_lo = max(inlines[0], int(split_highs.inline.min() - il_extent // 2 - 2))
                cache_il_hi = min(inlines[-1], int(split_highs.inline.max() + il_extent // 2 + 2))
                cache_xl_lo = max(xlines[0], int(split_highs.crossline.min() - xl_extent // 2 - 2))
                cache_xl_hi = min(xlines[-1], int(split_highs.crossline.max() + xl_extent // 2 + 2))
                cache_inline_axis = np.arange(cache_il_lo, cache_il_hi + 1)
                cache_xl_axis = np.arange(cache_xl_lo, cache_xl_hi + 1)
                cached_subvol = read_subvolume(f, iline_map, cache_inline_axis, cache_xl_axis, samples_ms.size)

                key = f'{horizon_name}|{split_name}'
                skip_counts[key] = {}
                low_disk_stop = False
                for tier in (1, 2, 3, 4):
                    if low_disk_stop:
                        break
                    for i in range(N_PER_TIER_PER_SPLIT):
                        if shutil.disk_usage(patches_dir).free < MIN_FREE_BYTES:
                            warnings.warn(
                                f'stopping early in {key}: free disk space fell below '
                                f'{MIN_FREE_BYTES / 1024**3:.1f}GB after {example_id} example(s) '
                                f'written. Returning what was generated so far.'
                            )
                            low_disk_stop = True
                            break
                        site = split_highs.iloc[rng.integers(len(split_highs))]
                        il_center, xl_center = int(site['inline']), int(site['crossline'])
                        result, skip_reason = generate_dim_spot_example(
                            tier, il_center, xl_center, cached_subvol, cache_inline_axis, cache_xl_axis,
                            samples_ms, horizon, dt_ms, freq_hz, RC_GAS_SAND, RC_WATER_SAND,
                            tuning_thickness_s, V_GAS_SAND, velocity_mps,
                            np.random.default_rng(int(rng.integers(0, 2**31))),
                        )
                        if result is None:
                            skip_counts[key][skip_reason] = skip_counts[key].get(skip_reason, 0) + 1
                            continue
                        fname = f'dimspot_{example_id:04d}.npz'
                        np.savez_compressed(patches_dir / fname,
                                             attribute_stack=result['attribute_stack'],
                                             channel_names=np.array(result['channel_names']),
                                             mask=result['mask'], mask_3d=result['mask_3d'])
                        row = dict(result['label'], example_id=example_id, split=split_name,
                                   patch_file=fname, horizon=horizon_name)
                        rows.append(row)
                        example_id += 1
                print(f'{horizon_name}/{split_name}: skip reasons {skip_counts[key]}')

    labels = pd.DataFrame(rows)
    if len(labels) == 0:
        raise RuntimeError('zero dim_spot examples generated - every scenario was skipped')
    labels.to_parquet(output_dir / 'labels.parquet', index=False)
    labels.to_csv(output_dir / 'labels.csv', index=False)

    manifest_out = {
        'kind': 'dim_spot',
        'seed': 3,
        'dt_ms': dt_ms, 'velocity_mps': velocity_mps, 'dominant_frequency_hz': freq_hz,
        'tuning_thickness_m': tuning_thickness_m,
        'n_examples': len(labels),
        'counts': {
            '|'.join(map(str, key)): int(value)
            for key, value in labels.groupby(['split', 'horizon', 'tier']).size().items()
        },
        'note': 'Additive P0 diversity supplement (dim_spot). No paired '
                'backgrounds generated here - reuse the H1/H3 datasets\' '
                'background patches at the same sites for P1/P2 training.',
    }
    (output_dir / 'generation_manifest.json').write_text(json.dumps(manifest_out, indent=2) + '\n')
    print(f'\n{len(labels)} dim_spot examples written to {output_dir}')
    print(json.dumps(manifest_out['counts'], indent=2))


if __name__ == '__main__':
    main()
