#!/usr/bin/env python3
"""Add extra background patches per site for the site-identity diagnostic.

17 background observations for 17 site classes can't support a site-ID classifier. This draws several more
per site rather than regenerating the dataset - background patches carry no
injection (thickness_m=0, no radius), so they're independent of the frozen
anomaly set and can be added on top of it without touching anything else.

The existing 17 backgrounds sit at the exact site centre, so drawing more at
that same single point would just duplicate one patch N times - no use to a
classifier. Instead each new draw is offset by a small random jitter (within
the anomaly footprint's own max radius, so it stays representative of that
site's local geology rather than wandering into a different structural
feature) using a seed kept separate from the frozen dataset's own seed, so
this can be re-run/extended without disturbing the original draws.
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import segyio

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.dhi_pipeline.dataset import _read_subvolume, generate_example  # noqa: E402
from src.dhi_pipeline.horizons import HorizonSurface, build_coordinate_lookup, load_horizon_surface  # noqa: E402
from src.dhi_pipeline.scenarios import FOOTPRINT_RADIUS_RANGE  # noqa: E402

# matches the anomaly footprint's own max radius (FOOTPRINT_RADIUS_RANGE[1]) -
# a background jittered any further could drift outside the site's own local
# geology and stop being representative of that site
JITTER_TRACES = FOOTPRINT_RADIUS_RANGE[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-dir', type=Path,
                         default=REPO / 'data/F3_synthetic_dhi_dataset_p0_radius6_15')
    parser.add_argument('--n-extra-per-site', type=int, default=8,
                         help='backgrounds added per site, on top of the existing 1 (default: 8, giving 9 total)')
    parser.add_argument('--seed', type=int, default=101,
                         help='separate from the frozen dataset\'s own seed=1, so this can be re-run independently')
    parser.add_argument('--dry-run', action='store_true',
                         help='build and report scenarios without writing any patches or touching labels/manifest')
    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    patches_dir = dataset_dir / 'patches'
    manifest = json.loads((dataset_dir / 'generation_manifest.json').read_text())
    dt_ms, velocity_mps, freq_hz = manifest['dt_ms'], manifest['velocity_mps'], manifest['dominant_frequency_hz']

    labels = pd.read_csv(dataset_dir / 'labels.csv')
    sites = labels[labels['kind'] == 'background'][['il_center', 'xl_center', 'split', 'center_time_ms']] \
        .drop_duplicates(subset=['il_center', 'xl_center']).reset_index(drop=True)
    # 3/17 sites' existing background sits at the fixed flat_top_time_ms=1400 fallback
    # rather than horizon.time_at() - at those locations the horizon is deep enough
    # (~1700-1780ms) that a +-250ms window runs off the end of the recording, so
    # every horizon-centred jittered draw there would fail with 'time_bounds'. Detect
    # which convention each site's existing background actually used and replicate it,
    # rather than assuming every site is horizon-centred.
    sites['fixed_time'] = np.isclose(sites['center_time_ms'], 1400.0)
    n_fixed = int(sites['fixed_time'].sum())
    print(f'{len(sites)} sites ({n_fixed} using the fixed flat_top_time_ms=1400 fallback), '
          f'adding {args.n_extra_per_site} backgrounds each '
          f'({len(sites) * args.n_extra_per_site} new patches)')

    raw = REPO / 'data_raw'
    segy_path = raw / 'Seismic_data.sgy'
    with (REPO / 'data/f3_trace_index.pkl').open('rb') as fh:
        index = pickle.load(fh)
    iline_map, inlines, xlines = index['iline_map'], index['inlines'], index['xlines']

    coords = build_coordinate_lookup(str(segy_path))
    surface = load_horizon_surface(str(raw / 'horizons/H1.xyz'), coords['ilxl_array'], coords['xy_array'])
    horizon = HorizonSurface(surface)

    il_extent = xl_extent = 96
    time_extent_ms = 500
    rng = np.random.default_rng(args.seed)

    example_id = int(labels['example_id'].max()) + 1
    new_rows = []
    skipped = 0

    with segyio.open(str(segy_path), ignore_geometry=True) as f:
        samples_ms = f.samples.astype(float)
        for split_name, split_sites in sites.groupby('split'):
            # build every jittered scenario for this split up front, so the working
            # sub-volume can be read once (same F10 pattern as build_dataset) instead
            # of once per patch
            scenarios = []
            for _, site in split_sites.iterrows():
                for _ in range(args.n_extra_per_site):
                    jitter_il = int(rng.integers(-JITTER_TRACES, JITTER_TRACES + 1))
                    jitter_xl = int(rng.integers(-JITTER_TRACES, JITTER_TRACES + 1))
                    kwargs = dict(
                        velocity_mps=velocity_mps, freq_hz=freq_hz, thickness_m=0.0,
                        il_center=site['il_center'] + jitter_il, xl_center=site['xl_center'] + jitter_xl,
                        il_radius=np.nan, xl_radius=np.nan, rotation_deg=0.0,
                    )
                    if site['fixed_time']:
                        kwargs['flat_top_time_ms'] = 1400
                    label = dict(
                        is_dhi=False, kind='background', tier='background',
                        il_center=site['il_center'] + jitter_il, xl_center=site['xl_center'] + jitter_xl,
                        thickness_m=0.0, reflection_coefficient=0.0,
                        flat_spot=False, polarity_reversal=False,
                        # keep the original (un-jittered) site coordinates too, so rows
                        # from the same site are still groupable for the site-ID split
                        site_il_center=site['il_center'], site_xl_center=site['xl_center'],
                    )
                    scenarios.append((kwargs, label))

            il_centers = [k['il_center'] for k, _ in scenarios]
            xl_centers = [k['xl_center'] for k, _ in scenarios]
            cache_il_lo = max(inlines[0], int(min(il_centers) - il_extent // 2))
            cache_il_hi = min(inlines[-1], int(max(il_centers) + il_extent // 2))
            cache_xl_lo = max(xlines[0], int(min(xl_centers) - xl_extent // 2))
            cache_xl_hi = min(xlines[-1], int(max(xl_centers) + xl_extent // 2))
            cache_inline_axis = np.arange(cache_il_lo, cache_il_hi + 1)
            cache_xl_axis = np.arange(cache_xl_lo, cache_xl_hi + 1)
            cached_subvol = _read_subvolume(f, iline_map, cache_inline_axis, cache_xl_axis, samples_ms.size)

            for kwargs, label in scenarios:
                result, skip_reason = generate_example(
                    kwargs, label, cached_subvol, cache_inline_axis, cache_xl_axis, samples_ms,
                    horizon, dt_ms, il_extent, xl_extent, time_extent_ms,
                )
                if result is None:
                    skipped += 1
                    continue

                fname = f'example_{example_id:04d}.npz'
                if not args.dry_run:
                    np.savez_compressed(patches_dir / fname,
                                         attribute_stack=result['attribute_stack'],
                                         channel_names=np.array(result['channel_names']),
                                         mask=result['mask'], mask_3d=result['mask_3d'])
                row = dict(result['label'], example_id=example_id, split=split_name, patch_file=fname)
                new_rows.append(row)
                example_id += 1

    if skipped:
        print(f'{skipped} jittered draws skipped (ran off the survey edge or time window)')

    if args.dry_run:
        print(f'[dry run] would add {len(new_rows)} background patches - no files written')
        per_site = pd.DataFrame(new_rows).groupby(['site_il_center', 'site_xl_center']).size()
        print(per_site.to_string())
        return

    new_labels = pd.DataFrame(new_rows)
    combined = pd.concat([labels, new_labels], ignore_index=True)

    # keep the original labels files, in case anything about this needs reverting
    (dataset_dir / 'labels_pre_siteid_backup.csv').write_text((dataset_dir / 'labels.csv').read_text())
    labels.to_parquet(dataset_dir / 'labels_pre_siteid_backup.parquet', index=False)

    combined.to_csv(dataset_dir / 'labels.csv', index=False)
    combined.to_parquet(dataset_dir / 'labels.parquet', index=False)

    manifest['site_background_draws'] = {
        'seed': args.seed, 'n_extra_per_site': args.n_extra_per_site,
        'jitter_traces': JITTER_TRACES, 'n_added': len(new_rows), 'n_skipped': skipped,
    }
    manifest['n_examples'] = len(combined)
    (dataset_dir / 'generation_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')

    print(f'added {len(new_rows)} background patches, {len(combined)} examples total')
    print(f'backup of original labels: labels_pre_siteid_backup.csv/.parquet')


if __name__ == '__main__':
    main()
