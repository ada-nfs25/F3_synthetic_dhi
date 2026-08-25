#!/usr/bin/env python3
"""
Round 3 blind set (your side of the swap: you inject into your reserved
hold-out, Aziz detects with his frozen primary/secondary, you score).

Site budget is small and fixed, not sampled freely: the hold-out horizon's
usable pool (structural highs/lows genuinely clear of both your own 1208
training windows and Aziz's 3 training rectangles, under the corrected 3D
space x time criterion) was already computed and vetted interactively -
19 clear highs (12 flat-spot-feasible), 19 clear lows. Every scenario below
is pinned to one specific pre-vetted site via sample_*_scenario's `site=`
override, not drawn at random from the full horizon, so nothing here can
accidentally land outside the checked pool.

`no_conformance`'s flat_top_time_ms is jittered uniquely per example, not a
shared constant - that fixed-timestamp pattern is exactly what leaked class
correlation in the earlier reverse-leg export (7 patches all at exactly
1400.0ms, all negatives, caught only because Aziz asked about it).

Composition (32 total, all inside the vetted pool):
  tier1_subtle x3, tier2_approaching x3 (rest-highs pool)
  tier3_at_tuning x3, tier4_obvious x3 (flat-spot-reserved highs, 6 of 12)
  single_reflector x2, tuning x2, no_conformance x2 (rest-highs pool)
  = 18 of 19 highs used (7 plain + 6 reserved-flat + 5 leftover-flat)
  hard_negative_syncline x4, background x10 (lows) = 14 of 19 lows used
  dim_spot excluded - different injection mechanism, not worth integrating
  for one example in a one-shot generation (see COMPOSITION_FROM_HIGHS)

Writes two SEPARATE things, same convention as the reverse leg:
  - data/round3_blind_volume/blind_for_aziz/ - anonymised .sgy + index.json.
    SEND THIS, nothing else from this directory.
  - data/round3_blind_volume/private_answer_key.csv - real labels. KEEP
    PRIVATE, this is what scores his returned detections later.
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

from src.dhi_pipeline.horizons import (  # noqa: E402
    HorizonSurface, build_coordinate_lookup, find_structural_highs, find_structural_lows,
    load_horizon_surface,
)
from src.dhi_pipeline.scenarios import sample_hard_negative_scenario, sample_positive_scenario  # noqa: E402
from src.dhi_pipeline.segy_export import export_scenario_to_segy  # noqa: E402

SEED = 20260825
HOLD_OUT_HORIZON = 'H6'
DEPTH_CAP_MS = 1550.0

# 3D clearance boxes - same as the interactive check that vetted this pool
HALF = 48
AZIZ_RECTANGLES = [
    ('H3-deep',    (100, 399), (750, 1149), (800, 1445)),
    ('H4-mid',     (450, 699), (900, 1199), (410, 1175)),
    ('H5-shallow', (100, 349), (450, 749),  (430, 1115)),
]

COMPOSITION_FROM_HIGHS = {
    'tier1_subtle': 3, 'tier2_approaching': 3,           # non-flat-reserved
    'tier3_at_tuning': 3, 'tier4_obvious': 3,             # flat-spot-reserved
    'single_reflector': 2, 'tuning': 2, 'no_conformance': 2,
}
"""dim_spot deliberately excluded: it uses a different injection mechanism
(dim_spot.py's build_anomaly, not inject_dhi_anomaly_3d) that this script
doesn't integrate - for one example, not worth the risk of an undertested
integration in a one-shot generation. Declared reduced composition, same
principle already agreed with Aziz for the flat-spot allocation."""
N_SYNCLINE = 4
N_BACKGROUND = 10


def own_training_boxes():
    import glob
    boxes = []
    for path in glob.glob(str(REPO / 'data/F3_synthetic_dhi_dataset_p0_*/labels.parquet')):
        df = pd.read_parquet(path)
        for _, r in df.iterrows():
            if 'il_lo' in df.columns and pd.notna(r.get('il_lo')):
                il_lo, il_hi, xl_lo, xl_hi = r.il_lo, r.il_hi, r.xl_lo, r.xl_hi
            else:
                il_lo, il_hi, xl_lo, xl_hi = r.il_center - 48, r.il_center + 48, r.xl_center - 48, r.xl_center + 48
            t = r.center_time_ms
            boxes.append((il_lo, il_hi, xl_lo, xl_hi, t - 250, t + 250))
    return boxes


def is_3d_clear(il_c, xl_c, t_c, own_boxes, half_t=250):
    il_lo, il_hi = il_c - HALF, il_c + HALF
    xl_lo, xl_hi = xl_c - HALF, xl_c + HALF
    t_lo, t_hi = t_c - half_t, t_c + half_t
    if il_hi > 750 or il_lo < 100:
        return False
    for (b_il_lo, b_il_hi, b_xl_lo, b_xl_hi, b_t_lo, b_t_hi) in own_boxes:
        if not (il_hi < b_il_lo or il_lo > b_il_hi or xl_hi < b_xl_lo or xl_lo > b_xl_hi
                or t_hi < b_t_lo or t_lo > b_t_hi):
            return False
    for name, (r_il_lo, r_il_hi), (r_xl_lo, r_xl_hi), (r_t_lo, r_t_hi) in AZIZ_RECTANGLES:
        if not (il_hi < r_il_lo or il_lo > r_il_hi or xl_hi < r_xl_lo or xl_lo > r_xl_hi
                or t_hi < r_t_lo or t_lo > r_t_hi):
            return False
    return True


def flat_spot_capable(row, grid_s, inlines, xlines, wavelet, dt_s):
    sys.path.insert(0, str(REPO / 'external'))
    from aziz_dhi_lib.geometry import footprint_times
    from aziz_dhi_lib.primitives import flat_spot_feasible
    il_idx, xl_idx = list(inlines).index(int(row.inline)), list(xlines).index(int(row.crossline))
    n_il_a, n_xl_a = 20, 20
    il0 = int(np.clip(il_idx - n_il_a // 2, 0, len(inlines) - n_il_a))
    xl0 = int(np.clip(xl_idx - n_xl_a // 2, 0, len(xlines) - n_xl_a))
    conform = footprint_times(grid_s, il0, xl0, n_il_a, n_xl_a, "conform")
    return flat_spot_feasible(conform, wavelet, dt_s)


def build_vetted_pool(horizon, surface, inlines, xlines):
    own_boxes = own_training_boxes()
    highs = find_structural_highs(surface)
    lows = find_structural_lows(surface)
    highs = highs[highs.time_ms <= DEPTH_CAP_MS]
    lows = lows[lows.time_ms <= DEPTH_CAP_MS]
    highs = highs[highs.apply(lambda r: is_3d_clear(r.inline, r.crossline, r.time_ms, own_boxes), axis=1)]
    lows = lows[lows.apply(lambda r: is_3d_clear(r.inline, r.crossline, r.time_ms, own_boxes), axis=1)]

    wavelet_path = Path.home() / 'Desktop/BLIND TESTS/ROUND 2/share_nora_round2/wavelet.npy'
    wavelet = np.load(wavelet_path)
    grid_s = np.empty((len(inlines), len(xlines)))
    for i, il in enumerate(inlines):
        for j, xl in enumerate(xlines):
            grid_s[i, j] = horizon.time_at(int(il), int(xl)) / 1000.0
    highs = highs.copy()
    highs['flat_capable'] = highs.apply(
        lambda r: flat_spot_capable(r, grid_s, inlines, xlines, wavelet, 0.004), axis=1)
    return highs.reset_index(drop=True), lows.reset_index(drop=True)


def main():
    raw_dir = REPO / 'data_raw'
    segy_path = raw_dir / 'Seismic_data.sgy'
    manifest = json.loads(
        (REPO / 'data/F3_synthetic_dhi_dataset_p0_radius6_15/generation_manifest.json').read_text())
    velocity_mps, freq_hz = manifest['velocity_mps'], manifest['dominant_frequency_hz']

    with (REPO / 'data/f3_trace_index.pkl').open('rb') as fh:
        index = pickle.load(fh)
    iline_map, inlines, xlines = index['iline_map'], index['inlines'], index['xlines']
    coords = build_coordinate_lookup(str(segy_path))
    surface = load_horizon_surface(str(raw_dir / f'horizons/{HOLD_OUT_HORIZON}.xyz'),
                                    coords['ilxl_array'], coords['xy_array'])
    horizon = HorizonSurface(surface)

    highs, lows = build_vetted_pool(horizon, surface, inlines, xlines)
    n_flat = int(highs.flat_capable.sum())
    print(f'vetted pool: {len(highs)} highs ({n_flat} flat-capable), {len(lows)} lows')
    if n_flat < COMPOSITION_FROM_HIGHS['tier3_at_tuning'] + COMPOSITION_FROM_HIGHS['tier4_obvious']:
        raise RuntimeError('flat-capable pool shrank below what the composition needs - re-plan before generating')

    rng = np.random.default_rng(SEED)
    flat_highs = highs[highs.flat_capable].sample(frac=1.0, random_state=rng.integers(0, 2**31)).reset_index(drop=True)
    plain_highs = highs[~highs.flat_capable].sample(frac=1.0, random_state=rng.integers(0, 2**31)).reset_index(drop=True)
    lows_shuffled = lows.sample(frac=1.0, random_state=rng.integers(0, 2**31)).reset_index(drop=True)

    # Only tier3/tier4 strictly need flat-capable sites; reserve exactly what
    # they need (not the whole flat-capable pool) and pool everything else
    # (plain sites + the leftover flat-capable ones) for every other kind that
    # draws from highs - first attempt reserved the *entire* flat-capable pool
    # for tier3/4 alone and starved the other 6 kinds, which only had the
    # 7-site plain pool to share between 12 draws.
    n_flat_reserved = COMPOSITION_FROM_HIGHS['tier3_at_tuning'] + COMPOSITION_FROM_HIGHS['tier4_obvious']
    flat_for_tiers = flat_highs.iloc[:n_flat_reserved].reset_index(drop=True)
    rest_highs = pd.concat([plain_highs, flat_highs.iloc[n_flat_reserved:]], ignore_index=True) \
        .sample(frac=1.0, random_state=rng.integers(0, 2**31)).reset_index(drop=True)

    high_cursor_flat = high_cursor_rest = low_cursor = 0
    rows = []

    def next_flat_site():
        nonlocal high_cursor_flat
        site = flat_for_tiers.iloc[high_cursor_flat]
        high_cursor_flat += 1
        return site

    def next_plain_site():
        nonlocal high_cursor_rest
        site = rest_highs.iloc[high_cursor_rest]
        high_cursor_rest += 1
        return site

    def next_low_site():
        nonlocal low_cursor
        site = lows_shuffled.iloc[low_cursor]
        low_cursor += 1
        return site

    for tier in ['tier1_subtle', 'tier2_approaching']:
        for _ in range(COMPOSITION_FROM_HIGHS[tier]):
            kwargs, label = sample_positive_scenario(tier, rng, highs, velocity_mps, freq_hz, site=next_plain_site())
            rows.append((kwargs, label))
    for tier in ['tier3_at_tuning', 'tier4_obvious']:
        for _ in range(COMPOSITION_FROM_HIGHS[tier]):
            kwargs, label = sample_positive_scenario(tier, rng, highs, velocity_mps, freq_hz, site=next_flat_site())
            rows.append((kwargs, label))
    for _ in range(COMPOSITION_FROM_HIGHS['single_reflector']):
        kwargs, label = sample_hard_negative_scenario(
            'single_reflector', rng, highs, lows, velocity_mps, freq_hz, site=next_plain_site())
        rows.append((kwargs, label))
    for _ in range(COMPOSITION_FROM_HIGHS['tuning']):
        kwargs, label = sample_hard_negative_scenario(
            'tuning', rng, highs, lows, velocity_mps, freq_hz, site=next_plain_site())
        rows.append((kwargs, label))
    for _ in range(COMPOSITION_FROM_HIGHS['no_conformance']):
        site = next_plain_site()
        jitter_ms = float(rng.uniform(-40, 40))
        flat_time = float(horizon.time_at(int(site['inline']), int(site['crossline']))) + jitter_ms
        kwargs, label = sample_hard_negative_scenario(
            'no_conformance', rng, highs, lows, velocity_mps, freq_hz,
            flat_background_time_ms=flat_time, site=site)
        rows.append((kwargs, label))
    for _ in range(N_SYNCLINE):
        kwargs, label = sample_hard_negative_scenario(
            'syncline', rng, highs, lows, velocity_mps, freq_hz, site=next_low_site())
        rows.append((kwargs, label))

    # backgrounds: no injected anomaly, just a real patch at a remaining low
    for _ in range(N_BACKGROUND):
        site = next_low_site()
        rows.append((None, dict(is_dhi=False, kind='background', tier=None,
                                 il_center=site['inline'], xl_center=site['crossline'])))

    print(f'{len(rows)} scenarios built '
          f'(highs used: {high_cursor_flat} reserved-flat + {high_cursor_rest} rest-pool of '
          f'{len(flat_for_tiers)}+{len(rest_highs)}; lows used: {low_cursor} of {len(lows_shuffled)})')

    output_dir = REPO / 'data/round3_blind_volume/blind_for_aziz'
    output_dir.mkdir(parents=True, exist_ok=True)
    private_dir = REPO / 'data/round3_blind_volume'

    order = rng.permutation(len(rows))
    manifest_rows = []
    with segyio.open(str(segy_path), ignore_geometry=True) as f:
        for out_i, i in enumerate(order):
            kwargs, label = rows[i]
            blind_id = f'blind_round3_{out_i:04d}'
            out_path = output_dir / f'{blind_id}.sgy'
            if kwargs is None:
                # background: no injection - matches dataset.py's own background
                # kwargs convention exactly (thickness_m=0.0, NaN radii), not an
                # invented variant.
                center_time_ms = horizon.time_at(int(label['il_center']), int(label['xl_center']))
                bg_kwargs = dict(
                    velocity_mps=velocity_mps, freq_hz=freq_hz, thickness_m=0.0,
                    reflection_coefficient=0.0,
                    il_center=label['il_center'], xl_center=label['xl_center'],
                    il_radius=np.nan, xl_radius=np.nan, rotation_deg=0.0,
                )
                result = export_scenario_to_segy(bg_kwargs, None, f, iline_map, inlines, xlines, horizon, out_path)
            else:
                result = export_scenario_to_segy(kwargs, None, f, iline_map, inlines, xlines, horizon, out_path)
                from src.dhi_pipeline.dataset import _scenario_center_time
                center_time_ms = _scenario_center_time(kwargs, horizon)

            row = dict(label, blind_id=blind_id, center_time_ms=float(center_time_ms), exported=result is not None)
            manifest_rows.append(row)

    manifest_df = pd.DataFrame(manifest_rows)
    exported = manifest_df[manifest_df.exported]
    n_failed = len(manifest_df) - len(exported)
    if n_failed:
        print(f'{n_failed} example(s) failed to export (ran off survey edge)')

    n_unique_times = exported.center_time_ms.nunique()
    if n_unique_times != len(exported):
        print(f'WARNING: only {n_unique_times}/{len(exported)} unique center times - check for accidental repeats')

    index_doc = {'patches': [{'patch_id': r.blind_id, 'center_time_ms': r.center_time_ms}
                              for r in exported.itertuples()]}
    (output_dir / 'index.json').write_text(json.dumps(index_doc, indent=2) + '\n')
    exported.to_csv(private_dir / 'private_answer_key.csv', index=False)

    print(f'\n{len(exported)} patches exported to {output_dir}')
    print(f'unique center times: {n_unique_times}/{len(exported)}')
    print(f'wrote PRIVATE answer key to {private_dir / "private_answer_key.csv"} - do not send this')
    print(f'\nSend: everything in {output_dir} (the .sgy files + index.json), nothing else from {private_dir}')


if __name__ == '__main__':
    main()
