#!/usr/bin/env python3
"""
P0 item 4 (Aziz, ROUND2_PLAN.md): "our injection library is in the bundle...
so you can generate our-style examples on your own volume - same physics
family, different calibration/wavelet/thickness ladder. Train on both
styles; that is what breaks generator dependence."

Uses external/aziz_dhi_lib's catalog API (plan_catalog -> apply_catalog ->
qc_report, per README_usage.md) with HIS calibration.json and wavelet.npy on
OUR H3 horizon, train/test splits matching our own convention (see the
horizon-depth note near where H3 is loaded below for why not H1). Kept
separate from our own generator (dataset.py/injection.py, untouched) -
purely additive, like H3 and dim_spot.

Site-separation mismatch: his MIN_SITE_SEPARATION (32 traces, sized to just
exceed his own largest footprint) is far smaller than our 96x96 patch
convention, so a naive extraction would pull neighbouring anomalies' signal
into "isolated" patches. First attempt requested a larger catalog than
needed and discarded anything closer than 96 after the fact - wasteful
(~85% discarded) and it still let the planner exhaust the valid region
chasing a count it could never place at the real spacing we needed. Fixed
properly instead: aziz_dhi_lib.inject.MIN_SITE_SEPARATION is monkey-patched
to MIN_PATCH_SEPARATION before calling plan_catalog, so the planner places
sites already respecting OUR spacing requirement rather than its own -
nothing to discard afterward, and the requested count is something the
planner can actually satisfy.

dim_spot is deliberately excluded from the requested catalog
(n_dim_spot_per_tier=0): P0.3 already covers that DHI kind via our own
calibration, and mixing his differently-calibrated dim_spot in as well
isn't needed to test "both generators' conventions" for the kinds we don't
already have from another source.
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import segyio

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
BUNDLE = Path.home() / 'Desktop' / 'BLIND TESTS' / 'ROUND 2' / 'share_nora_round2'
if str(REPO / 'external') not in sys.path:
    sys.path.insert(0, str(REPO / 'external'))

import aziz_dhi_lib.inject as aziz_inject  # noqa: E402
from aziz_dhi_lib.attributes import reliable_time_window  # noqa: E402
from aziz_dhi_lib.constants import SUPERVISED_PATCH_HALF_SAMPLES  # noqa: E402
from aziz_dhi_lib.inject import apply_catalog, plan_catalog  # noqa: E402
from aziz_dhi_lib.qc import qc_report  # noqa: E402
from src.dhi_pipeline.attributes import compute_attribute_stack  # noqa: E402
from src.dhi_pipeline.horizons import (  # noqa: E402
    HorizonSurface, build_coordinate_lookup, load_horizon_surface,
)

SPLITS = {'train': (100, 345), 'test': (365, 670)}
# Fixed per-split seeds, not hash(split_name): Python randomizes string hashing
# per-process by default, so the old `hash(split_name) % (2**31)` drew a
# different RNG stream on every run and made this script non-reproducible.
SPLIT_SEEDS = {'train': 11, 'test': 22}
"""train's lower bound extended vs the H1/H3/dim_spot convention (150) into
otherwise-unused survey inline range - measured directly: 100-150 has 99.8%
depth-window validity (vs 90.4% for 150-345 alone), meaningfully more usable
area for the same site-separation constraint. No site-disjointness risk:
H1/H3/dim_spot never place anything in 100-150, and it stays outside test's
range too. test's own unused edge (670-750) was checked and is WORSE
(29.2% valid, not 37.1%), so left as the H1/H3/dim_spot convention."""
IL_EXTENT = XL_EXTENT = 96
TIME_EXTENT_MS = 500
MIN_PATCH_SEPARATION = 70
"""Chebyshev. The exact non-contamination bound is patch half-width (48) +
max footprint half-width (FOOTPRINT_IL/XL_RANGE max 30, half 15) = 63 - a
neighbour's footprint cannot reach into our extraction window beyond that.
70 adds a small margin. 96 (the full patch width) was unnecessarily
conservative and, combined with our train split's narrow 196-trace inline
range, capped capacity at ~18 sites regardless of requested count -
196 traces / 96 separation only fits ~2 rows. At 70 the same range fits
closer to 3, meaningfully more room without any actual contamination risk."""
CATALOG_PARAMS_BY_SPLIT = {
    # flat_spot_only dropped entirely: this survey region's relief is too
    # flat at the required footprint scale to reliably resolve a flat spot
    # against his wavelet's dominant period (observed directly - a
    # 28-candidate relief-aware pool couldn't find even one feasible site).
    # bright_spot still includes tiers 3-4 (TIER_PARAMS forces flat_spot=True
    # there, not configurable per-tier through this parameter).
    'train': dict(n_dhi_per_tier=2, n_dim_spot_per_tier=0, n_flat_spot_only_per_tier=0, n_negatives_per_kind=3),
    # ceiling is somewhere between 20 (this) and 32 (fails) requested - not
    # worth bisecting further for a marginal gain.
    # test's H3 grid sits systematically deeper than train's (mean 1.243s vs
    # 1.185s) - only ~37% of it falls inside dhi_lib's viable depth window,
    # vs ~90% for train (measured directly) - so its usable candidate area
    # is much smaller for the same requested count.
    'test': dict(n_dhi_per_tier=1, n_dim_spot_per_tier=0, n_flat_spot_only_per_tier=0, n_negatives_per_kind=1),
}


def build_horizon_grid_and_valid_mask(horizon, inline_axis, xl_axis, depth_window_s):
    """(n_il, n_xl) grids: time in TWT SECONDS (dhi_lib's convention, vs our
    own time_ms), and True only where HorizonSurface has an EXACT pick
    (dhi_lib's `valid_mask` semantics - never site on gap-filled structure)
    AND the horizon time sits inside dhi_lib's own viable depth window for
    this trace length (see _preflight_horizon_depths - H3's own real relief
    puts ~10% of its grid past the window's deep edge; folding the window
    into valid_mask reuses the same "don't site here" mechanism rather than
    inventing a second one, since plan_catalog's preflight check is
    all-or-nothing against whatever sites get selected)."""
    grid_s = np.empty((len(inline_axis), len(xl_axis)))
    valid = np.empty((len(inline_axis), len(xl_axis)), dtype=bool)
    for i, il in enumerate(inline_axis):
        for j, xl in enumerate(xl_axis):
            grid_s[i, j] = horizon.time_at(int(il), int(xl)) / 1000.0
            valid[i, j] = (int(il), int(xl)) in horizon.lookup
    lo, hi = depth_window_s
    valid &= (grid_s >= lo) & (grid_s <= hi)
    return grid_s, valid


def read_full_subvolume(f, iline_map, inline_axis, xl_axis, n_samples):
    out = np.full((len(inline_axis), len(xl_axis), n_samples), np.nan, dtype=np.float32)
    for i, il in enumerate(inline_axis):
        for j, xl in enumerate(xl_axis):
            idx = iline_map.get((int(il), int(xl)))
            if idx is not None:
                out[i, j] = f.trace[idx]
    return out


def extract_patch(mod, mask, anomaly_id, il_off, xl_off, il_center, xl_center, t_center_s, dt_s, n_time):
    il_lo, il_hi = il_center - il_off - IL_EXTENT // 2, il_center - il_off + IL_EXTENT // 2
    xl_lo, xl_hi = xl_center - xl_off - XL_EXTENT // 2, xl_center - xl_off + XL_EXTENT // 2
    if il_lo < 0 or il_hi > mod.shape[0] or xl_lo < 0 or xl_hi > mod.shape[1]:
        return None, 'il_xl_bounds'

    t_center_sample = t_center_s / dt_s
    t_half = (TIME_EXTENT_MS / 1000.0 / dt_s) / 2
    t_lo, t_hi = int(round(t_center_sample - t_half)), int(round(t_center_sample - t_half)) + int(round(2 * t_half))
    if t_lo < 0 or t_hi > n_time:
        return None, 'time_bounds'

    amplitude = np.nan_to_num(mod[il_lo:il_hi, xl_lo:xl_hi, t_lo:t_hi], nan=0.0).astype(np.float32)
    mask_3d = (mask[il_lo:il_hi, xl_lo:xl_hi, t_lo:t_hi] == anomaly_id)
    if not mask_3d.any():
        return None, 'no_mask_voxels_in_patch'
    return dict(amplitude=amplitude, mask=mask_3d.any(axis=2), mask_3d=mask_3d), None


def main():
    raw_dir = REPO / 'data_raw'
    segy_path = raw_dir / 'Seismic_data.sgy'
    output_dir = REPO / 'data' / 'F3_synthetic_dhi_dataset_p0_aziz_style'
    patches_dir = output_dir / 'patches'
    patches_dir.mkdir(parents=True, exist_ok=True)

    calib = json.loads((BUNDLE / 'calibration.json').read_text())
    wavelet = np.load(BUNDLE / 'wavelet.npy')

    # see module docstring: match his planner's own spacing to our patch size
    # instead of requesting-then-discarding.
    aziz_inject.MIN_SITE_SEPARATION = MIN_PATCH_SEPARATION
    # his own EDGE_MARGIN (~16, sized to his smaller footprints) lets sites
    # sit closer to the survey edge than our 96-wide extraction can use -
    # observed directly (13/24 train patches lost to il_xl_bounds). Matching
    # margin to our own patch half-width instead of his.
    aziz_inject.EDGE_MARGIN = IL_EXTENT // 2 + 2

    with (REPO / 'data/f3_trace_index.pkl').open('rb') as fh:
        index = pickle.load(fh)
    iline_map, inlines, xlines = index['iline_map'], index['inlines'], index['xlines']
    coords = build_coordinate_lookup(str(segy_path))
    # H3, not H1: dhi_lib's own preflight requires the horizon to sit within a
    # shallow depth window tied to our 462-sample traces ([608ms, 1236ms],
    # from reliable_time_window) - H1 (~1654ms) fails it outright, H3
    # (~1217ms mean) mostly fits (the ~10% that doesn't is excluded via
    # valid_mask below, not by switching horizon again).
    surface = load_horizon_surface(str(raw_dir / 'horizons/H3.xyz'), coords['ilxl_array'], coords['xy_array'])
    horizon = HorizonSurface(surface)

    rows = []
    skip_counts = {}
    example_id = 0

    with segyio.open(str(segy_path), ignore_geometry=True) as f:
        samples_ms = f.samples.astype(float)
        dt_s = (samples_ms[1] - samples_ms[0]) / 1000.0
        n_time = len(samples_ms)

        for split_name, (il_lo, il_hi) in SPLITS.items():
            inline_axis = np.array([il for il in inlines if il_lo <= il <= il_hi])
            xl_axis = xlines
            print(f'{split_name}: reading {len(inline_axis)}x{len(xl_axis)} sub-volume ...')
            volume = read_full_subvolume(f, iline_map, inline_axis, xl_axis, n_time)
            win_lo, win_hi = reliable_time_window(
                n_time, attr_names=None, patch_half_samples=SUPERVISED_PATCH_HALF_SAMPLES)
            depth_window_s = (win_lo * dt_s, (win_hi - 1) * dt_s)
            horizon_grid_s, valid_mask = build_horizon_grid_and_valid_mask(
                horizon, inline_axis, xl_axis, depth_window_s)

            rng = np.random.default_rng(SPLIT_SEEDS[split_name])
            catalog = plan_catalog(
                horizon_grid_s, rng=rng, nt=n_time, dt_s=dt_s,
                rc_gas=calib['rc_gas'], rc_brine=calib['rc_brine'],
                tuning_thickness_s=calib['tuning_thickness_s'],
                valid_mask=valid_mask, wavelet=wavelet, **CATALOG_PARAMS_BY_SPLIT[split_name],
            )
            print(f'{split_name}: planned {len(catalog)} anomalies')

            volume_filled = np.nan_to_num(volume, nan=0.0)
            mod, mask = apply_catalog(
                volume_filled, dt_s, horizon_grid_s, catalog, wavelet,
                tuning_thickness_s=calib['tuning_thickness_s'], apply_sag=True, polarity_mode='lateral',
            )
            report = qc_report(volume_filled, mod, mask, dt_s)
            print(f'{split_name}: qc gate_pass={report["gate_pass"]}')
            if not report['gate_pass']:
                raise RuntimeError(f'{split_name}: qc_report failed gate_pass: {report}')

            skip_counts[split_name] = {}
            for anomaly_id, spec in enumerate(catalog, start=1):
                il_c, xl_c = spec.site
                t_center_s = horizon_grid_s[il_c, xl_c]
                result, skip_reason = extract_patch(
                    mod, mask, anomaly_id, 0, 0, il_c, xl_c, t_center_s, dt_s, n_time)
                if result is None:
                    skip_counts[split_name][skip_reason] = skip_counts[split_name].get(skip_reason, 0) + 1
                    continue

                stack, channel_names = compute_attribute_stack(result['amplitude'], dt_s)
                fname = f'aziz_style_{example_id:04d}.npz'
                np.savez_compressed(patches_dir / fname, attribute_stack=stack.astype(np.float32),
                                     channel_names=np.array(channel_names),
                                     mask=result['mask'], mask_3d=result['mask_3d'])

                is_dhi = spec.kind in ('bright_spot', 'dim_spot', 'flat_spot_only')
                rows.append(dict(
                    is_dhi=is_dhi, kind=spec.kind, tier=(f'tier{spec.tier}' if spec.tier else spec.kind),
                    il_center=int(il_c + inline_axis[0]), xl_center=int(xl_c + xl_axis[0]),
                    thickness_s=spec.thickness_s, rc_gas=spec.rc_gas, rc_brine=spec.rc_brine,
                    center_time_ms=t_center_s * 1000.0, example_id=example_id,
                    split=split_name, patch_file=fname, catalog_anomaly_id=anomaly_id,
                ))
                example_id += 1

            print(f'{split_name}: {len([r for r in rows if r["split"] == split_name])} patches extracted, '
                  f'skip reasons {skip_counts[split_name]}')

    labels = pd.DataFrame(rows)
    labels['dataset_source'] = 'aziz_style'
    labels.to_parquet(output_dir / 'labels.parquet', index=False)
    labels.to_csv(output_dir / 'labels.csv', index=False)

    manifest = dict(
        source='external/aziz_dhi_lib catalog API, his calibration.json + wavelet.npy',
        horizon='H3', catalog_params=CATALOG_PARAMS_BY_SPLIT,
        min_patch_separation_traces=MIN_PATCH_SEPARATION,
        n_examples=len(labels),
        counts={f'{split}|{kind}': n for (split, kind), n in labels.groupby(['split', 'kind']).size().items()},
        skip_counts=skip_counts,
        note='Additive P0 diversity supplement (his generator convention). '
             'Concatenate with H1/H3/dim_spot for P1/P2 training, do not use alone.',
    )
    (output_dir / 'generation_manifest.json').write_text(
        json.dumps(manifest, indent=2, default=str))
    print(f'\n{len(labels)} aziz-style examples written to {output_dir}')
    print(json.dumps({k: v for k, v in manifest.items() if k != 'skip_counts'}, indent=2, default=str))


if __name__ == '__main__':
    main()
