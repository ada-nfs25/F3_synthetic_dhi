"""
Patch + label dataset export.

Runs the stage-2/3 scenario samplers in bulk, extracts a fixed-size 3D patch
around each scenario (amplitude + envelope + instantaneous-phase channels,
plus a footprint mask for localization), and writes everything to disk as
one .npz per example plus a single labels table.

Spatial train/test partitioning: every example is injected onto the same
real F3 volume, so a random split would let train and test examples share
background noise fingerprints from nearby/overlapping locations. Splitting
by inline range instead means test-time locations are never seen at all
during training-set generation.
"""

import os

import numpy as np
import pandas as pd
import segyio

from .attributes import compute_attribute_stack
from .injection import inject_dhi_anomaly_3d, estimate_amplitude_scale
from .scenarios import sample_positive_scenario, sample_hard_negative_scenario, TIER_RANGES

HARD_NEGATIVE_KINDS = ['no_conformance', 'syncline', 'single_reflector', 'tuning']


def compute_footprint_mask(inline_axis, xl_axis, il_center, xl_center, il_radius, xl_radius, rotation_deg=0.0):
    """Boolean (n_inlines, n_xlines) mask, True inside the injected footprint ellipse - mirrors the
    distance calculation in `inject_dhi_anomaly_3d` exactly, so the mask matches what was actually injected."""
    theta = np.radians(rotation_deg)
    il_grid, xl_grid = np.meshgrid(inline_axis, xl_axis, indexing='ij')
    d_il, d_xl = il_grid - il_center, xl_grid - xl_center
    il_rot = d_il * np.cos(theta) + d_xl * np.sin(theta)
    xl_rot = -d_il * np.sin(theta) + d_xl * np.cos(theta)
    r = np.sqrt((il_rot / il_radius) ** 2 + (xl_rot / xl_radius) ** 2)
    return r <= 1.0


def _scenario_center_time(kwargs, horizon):
    if kwargs.get('flat_top_time_ms') is not None:
        return kwargs['flat_top_time_ms']
    return horizon.time_at(kwargs['il_center'], kwargs['xl_center']) + kwargs.get('horizon_time_offset_ms', 0.0)


def generate_example(kwargs, label, f, iline_map, inlines, xlines, horizon, dt_ms,
                      il_extent=160, xl_extent=160, time_extent_ms=500, reference_rc=0.05):
    """
    Build one dataset example: reads a fixed-size patch directly from SEG-Y,
    injects the given scenario, computes the attribute stack + footprint mask.

    Returns None if the patch would run off the survey's valid il/xl range
    (kept simple: skip rather than pad, so every example has identical shape).
    """
    il_center, xl_center = kwargs['il_center'], kwargs['xl_center']
    center_time_ms = _scenario_center_time(kwargs, horizon)

    il_lo, il_hi = int(il_center - il_extent // 2), int(il_center + il_extent // 2)
    xl_lo, xl_hi = int(xl_center - xl_extent // 2), int(xl_center + xl_extent // 2)
    if il_lo < inlines[0] or il_hi > inlines[-1] or xl_lo < xlines[0] or xl_hi > xlines[-1]:
        return None

    patch_inline_axis = np.arange(il_lo, il_hi)
    patch_xl_axis = np.arange(xl_lo, xl_hi)

    n_samples = f.samples.size
    raw = np.full((len(patch_inline_axis), len(patch_xl_axis), n_samples), np.nan)
    for i, il in enumerate(patch_inline_axis):
        for j, xl in enumerate(patch_xl_axis):
            idx = iline_map.get((int(il), int(xl)))
            if idx is not None:
                raw[i, j] = f.trace[idx]
    n_missing = int(np.isnan(raw).sum() // n_samples)

    full_time_axis_ms = f.samples.astype(float)
    t_mask = (full_time_axis_ms >= center_time_ms - time_extent_ms / 2) & \
             (full_time_axis_ms <= center_time_ms + time_extent_ms / 2)
    patch_time_axis_ms = full_time_axis_ms[t_mask]
    raw_patch = raw[:, :, t_mask]

    amp_scale = estimate_amplitude_scale(raw_patch, reference_rc=reference_rc)
    injected, twt_thickness_ms = inject_dhi_anomaly_3d(
        raw_patch, patch_time_axis_ms, patch_inline_axis, patch_xl_axis, horizon,
        amplitude_scale=amp_scale, **kwargs,
    )

    # Hilbert transform can't handle NaN gaps (real survey-edge missing traces) - fill with 0
    # before computing attributes; the mask below is unaffected (footprint is a geometric property).
    injected_filled = np.nan_to_num(injected, nan=0.0)
    attribute_stack = compute_attribute_stack(injected_filled)

    mask = compute_footprint_mask(patch_inline_axis, patch_xl_axis, il_center, xl_center,
                                   kwargs['il_radius'], kwargs['xl_radius'], kwargs.get('rotation_deg', 0.0))

    peak_amplitude = float(np.nanmax(np.abs(injected - raw_patch)))
    full_label = dict(label, il_lo=il_lo, il_hi=il_hi, xl_lo=xl_lo, xl_hi=xl_hi,
                       center_time_ms=center_time_ms, twt_thickness_ms=twt_thickness_ms,
                       n_missing_traces=n_missing, peak_amplitude=peak_amplitude,
                       il_radius=kwargs['il_radius'], xl_radius=kwargs['xl_radius'],
                       rotation_deg=kwargs.get('rotation_deg', 0.0))
    return dict(attribute_stack=attribute_stack, mask=mask, label=full_label)


def build_dataset(output_dir, segy_path, iline_map, inlines, xlines, horizon,
                   dt_ms, velocity_mps, freq_hz, train_inline_range, test_inline_range,
                   structural_highs, structural_lows, n_per_tier=3, n_hard_negatives_per_kind=3,
                   seed=0, il_extent=160, xl_extent=160, time_extent_ms=500):
    """
    Generate a full patch+label dataset: n_per_tier positive examples per
    severity tier, n_hard_negatives_per_kind per hard-negative type, for
    each of train/test (using only structural highs/lows within that
    split's inline range, so evaluation locations are never used to
    generate training examples).

    Writes one .npz per example to `{output_dir}/patches/`, and a combined
    labels table to `{output_dir}/labels.parquet`. Returns the labels DataFrame.
    """
    patches_dir = os.path.join(output_dir, 'patches')
    os.makedirs(patches_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    splits = {'train': train_inline_range, 'test': test_inline_range}
    rows = []
    example_id = 0

    with segyio.open(segy_path, ignore_geometry=True) as f:
        for split_name, (il_lo, il_hi) in splits.items():
            split_highs = structural_highs[structural_highs.inline.between(il_lo, il_hi)].reset_index(drop=True)
            split_lows = structural_lows[structural_lows.inline.between(il_lo, il_hi)].reset_index(drop=True)
            if len(split_highs) == 0:
                raise ValueError(f'no structural highs in split "{split_name}" inline range {il_lo}-{il_hi}')

            scenarios = []
            for tier in TIER_RANGES:
                for _ in range(n_per_tier):
                    scenarios.append(sample_positive_scenario(tier, rng, split_highs, velocity_mps, freq_hz))
            for kind in HARD_NEGATIVE_KINDS:
                if kind == 'syncline' and len(split_lows) == 0:
                    continue
                for _ in range(n_hard_negatives_per_kind):
                    scenarios.append(sample_hard_negative_scenario(
                        kind, rng, split_highs, split_lows, velocity_mps, freq_hz,
                        flat_background_time_ms=1400,
                    ))

            for kwargs, label in scenarios:
                result = generate_example(kwargs, label, f, iline_map, inlines, xlines, horizon, dt_ms,
                                           il_extent, xl_extent, time_extent_ms)
                if result is None:
                    continue  # patch ran off the survey edge - skip rather than pad

                fname = f'example_{example_id:04d}.npz'
                np.savez_compressed(os.path.join(patches_dir, fname),
                                     attribute_stack=result['attribute_stack'], mask=result['mask'])
                row = dict(result['label'], example_id=example_id, split=split_name, patch_file=fname)
                rows.append(row)
                example_id += 1

    labels = pd.DataFrame(rows)
    labels.to_parquet(os.path.join(output_dir, 'labels.parquet'), index=False)
    return labels
