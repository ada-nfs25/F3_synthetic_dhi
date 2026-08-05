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
import warnings

import numpy as np
import pandas as pd
import segyio

from .attributes import compute_attribute_stack
from .injection import inject_dhi_anomaly_3d, estimate_amplitude_scale, sag_time_shift_ms, V_GAS_SAND
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


def _read_subvolume(f, iline_map, inline_axis, xl_axis, n_samples):
    """
    Read a full (inline, crossline, time) sub-volume from SEG-Y in one pass,
    NaN where a trace is missing (F3's acquisition outline isn't a perfect
    rectangle) - F10. Same approach as utils/seismic_io.read_subvolume,
    duplicated here rather than imported, so dhi_pipeline doesn't pick up a
    dependency on the separate top-level utils package.
    """
    out = np.full((len(inline_axis), len(xl_axis), n_samples), np.nan)
    for i, il in enumerate(inline_axis):
        for j, xl in enumerate(xl_axis):
            idx = iline_map.get((int(il), int(xl)))
            if idx is not None:
                out[i, j] = f.trace[idx]
    return out


def generate_example(kwargs, label, cached_subvol, cache_inline_axis, cache_xl_axis, samples_ms,
                      horizon, dt_ms, il_extent=96, xl_extent=96, time_extent_ms=500, reference_rc=0.05):
    """
    Build one dataset example: slices a fixed-size patch out of a pre-cached
    sub-volume (F10 - see `_read_subvolume`/`build_dataset`, which reads the
    working sub-volume from SEG-Y once per split rather than per example),
    injects the given scenario, computes the attribute stack + footprint mask.

    cached_subvol: (n_cache_inlines, n_cache_xlines, n_samples) array already
        read from SEG-Y for this split - see build_dataset.
    cache_inline_axis/cache_xl_axis: the actual inline/crossline values
        cached_subvol's first two axes correspond to (consecutive integers),
        used to convert a patch's absolute il/xl range into an offset into
        cached_subvol.

    Returns (None, skip_reason) if the patch would run off the cached sub-volume's
    il/xl range, or off the survey's recorded time range (kept simple: skip rather
    than pad/truncate, so every example has identical shape) - skip_reason is
    'il_xl_bounds' or 'time_bounds', so a run with many skips is diagnosable (F9).
    Returns (result_dict, None) on success.
    """
    il_center, xl_center = kwargs['il_center'], kwargs['xl_center']
    center_time_ms = _scenario_center_time(kwargs, horizon)

    il_lo, il_hi = int(il_center - il_extent // 2), int(il_center + il_extent // 2)
    xl_lo, xl_hi = int(xl_center - xl_extent // 2), int(xl_center + xl_extent // 2)
    if il_lo < cache_inline_axis[0] or il_hi > cache_inline_axis[-1] or \
       xl_lo < cache_xl_axis[0] or xl_hi > cache_xl_axis[-1]:
        return None, 'il_xl_bounds'

    # F4: the footprint ellipse's own radius is its full extent (inject_dhi_anomaly_3d's
    # edge_taper_frac softens amplitude *inside* r<=1, it doesn't grow past il_radius/xl_radius),
    # so require the radius to clear the patch's half-extent with a small margin - otherwise the
    # footprint gets clipped by the patch edge and the localisation mask silently loses meaning
    # (see the review's F4, radius=80 against the old il_extent=160 covered 78% of the patch).
    if kwargs['il_radius'] >= il_extent / 2 or kwargs['xl_radius'] >= xl_extent / 2:
        return None, 'footprint_exceeds_patch'

    # same edge-skip logic as il/xl above, but for the time window - without this, a
    # center_time_ms near the start/end of the recording silently produced a shorter-than-
    # requested time_extent_ms window instead of being skipped. That was a real confound:
    # hard_negative_syncline was clipped 87.5% of the time vs 0% for no_conformance, since
    # synclines sit at structural lows (deeper -> later in the recording -> more likely clipped).
    t_lo_ms, t_hi_ms = center_time_ms - time_extent_ms / 2, center_time_ms + time_extent_ms / 2
    if t_lo_ms < samples_ms[0] or t_hi_ms > samples_ms[-1]:
        return None, 'time_bounds'

    patch_inline_axis = np.arange(il_lo, il_hi)
    patch_xl_axis = np.arange(xl_lo, xl_hi)

    # F10: slice directly out of the pre-cached sub-volume instead of re-reading
    # from SEG-Y per voxel - cache_inline_axis[0]/cache_xl_axis[0] are the cache's
    # own origin, so the patch's array offset is just relative to that.
    il_off = il_lo - cache_inline_axis[0]
    xl_off = xl_lo - cache_xl_axis[0]
    raw = cached_subvol[il_off:il_off + len(patch_inline_axis), xl_off:xl_off + len(patch_xl_axis), :]
    n_missing = int(np.isnan(raw).sum() // samples_ms.size)

    full_time_axis_ms = samples_ms
    t_mask = (full_time_axis_ms >= center_time_ms - time_extent_ms / 2) & \
             (full_time_axis_ms <= center_time_ms + time_extent_ms / 2)
    patch_time_axis_ms = full_time_axis_ms[t_mask]
    raw_patch = raw[:, :, t_mask]

    amp_scale = estimate_amplitude_scale(raw_patch, reference_rc=reference_rc)
    try:
        injected, twt_thickness_ms = inject_dhi_anomaly_3d(
            raw_patch, patch_time_axis_ms, patch_inline_axis, patch_xl_axis, horizon,
            amplitude_scale=amp_scale, **kwargs,
        )
    except ValueError as e:
        # F4's smaller footprints make two of inject_dhi_anomaly_3d's own validity checks
        # much more likely to fire on an otherwise-legitimate random scenario draw: a
        # shallow-dip site where the footprint's relief can't resolve a flat_spot/
        # polarity_reversal contact (F1's guard), or a footprint edge that happens to probe
        # a real horizon gap beyond max_lookup_distance (F6). Both are expected outcomes of
        # random sampling, not bugs - skip just this one scenario rather than let it crash
        # the whole build_dataset run (previously uncaught, so one bad draw lost the entire
        # batch). Anything else (e.g. a genuine flat_top_time_ms/flat_spot kwargs conflict)
        # still raises - that indicates a real construction bug, not sampling variance.
        if 'resolvable threshold' in str(e):
            return None, 'unresolvable_relief'
        if 'max_lookup_distance' in str(e):
            return None, 'horizon_lookup_gap'
        raise

    # Hilbert transform can't handle NaN gaps (real survey-edge missing traces) - fill with 0
    # before computing attributes; the mask below is unaffected (footprint is a geometric property).
    injected_filled = np.nan_to_num(injected, nan=0.0)
    attribute_stack, channel_names = compute_attribute_stack(injected_filled, dt_ms / 1000.0)
    attribute_stack = attribute_stack.astype(np.float32)

    mask = compute_footprint_mask(patch_inline_axis, patch_xl_axis, il_center, xl_center,
                                   kwargs['il_radius'], kwargs['xl_radius'], kwargs.get('rotation_deg', 0.0))

    peak_amplitude = float(np.nanmax(np.abs(injected - raw_patch)))
    # Ground truth for the sag/pull-down attribute (Nanda 2021) - mirrors inject_dhi_anomaly_3d's
    # own internal calculation rather than changing its return signature. Applies uniformly
    # across positives and hard negatives alike: sag depends on velocity contrast + thickness,
    # both shared by hard negatives per their design (see scenarios.py), not on structural realism.
    sag_shift_ms = (
        sag_time_shift_ms(kwargs['thickness_m'], kwargs.get('v_gas_mps') or V_GAS_SAND, kwargs['velocity_mps'])
        if kwargs.get('apply_sag', True) else 0.0
    )
    full_label = dict(label, il_lo=il_lo, il_hi=il_hi, xl_lo=xl_lo, xl_hi=xl_hi,
                       center_time_ms=center_time_ms, twt_thickness_ms=twt_thickness_ms,
                       n_missing_traces=n_missing, peak_amplitude=peak_amplitude,
                       il_radius=kwargs['il_radius'], xl_radius=kwargs['xl_radius'],
                       rotation_deg=kwargs.get('rotation_deg', 0.0), sag_shift_ms=sag_shift_ms)
    return dict(attribute_stack=attribute_stack, channel_names=channel_names, mask=mask, label=full_label), None


def build_dataset(output_dir, segy_path, iline_map, inlines, xlines, horizon,
                   dt_ms, velocity_mps, freq_hz, train_inline_range, test_inline_range,
                   structural_highs, structural_lows, n_per_tier=3, n_hard_negatives_per_kind=3,
                   seed=0, il_extent=96, xl_extent=96, time_extent_ms=500,
                   max_skip_rate=0.5):
    """
    Generate a full patch+label dataset: n_per_tier positive examples per
    severity tier, n_hard_negatives_per_kind per hard-negative type, for
    each of train/test (using only structural highs/lows within that
    split's inline range, so evaluation locations are never used to
    generate training examples).

    Writes one .npz per example to `{output_dir}/patches/`, and a combined
    labels table to `{output_dir}/labels.parquet`. Returns the labels DataFrame.

    F9: validates up front that `inlines`/`xlines` can hold at least one
    patch (a volume narrower than il_extent/xl_extent means every scenario
    would be silently skipped, and previously this function would finish
    "successfully" with an empty labels table and no error). Also raises if
    the run produces zero examples, and warns if more than `max_skip_rate`
    of scenarios were skipped - both report the skip-reason breakdown
    (il/xl bounds vs time bounds) so a partial run is diagnosable.

    F10: reads the working sub-volume for each split into memory once
    (`_read_subvolume`), rather than re-reading the same overlapping traces
    from SEG-Y for every example - see `generate_example`.
    """
    valid_il_band = len(inlines) - il_extent
    valid_xl_band = len(xlines) - xl_extent
    if valid_il_band <= 0 or valid_xl_band <= 0:
        raise ValueError(
            f'working volume too small to hold any patch: {len(inlines)} inlines x {len(xlines)} '
            f'crosslines available, but il_extent={il_extent}/xl_extent={xl_extent} need a valid '
            f'centre band > 0 (available: {valid_il_band} x {valid_xl_band}). At least '
            f'{il_extent + 1} inlines and {xl_extent + 1} crosslines are needed.'
        )

    patches_dir = os.path.join(output_dir, 'patches')
    os.makedirs(patches_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    splits = {'train': train_inline_range, 'test': test_inline_range}
    rows = []
    example_id = 0
    skip_counts = {}

    with segyio.open(segy_path, ignore_geometry=True) as f:
        samples_ms = f.samples.astype(float)
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

            # F10: read this split's working sub-volume once - the union of every
            # sampled scenario's patch, clipped to the survey's actual valid range -
            # rather than re-reading the same overlapping traces per example. Only
            # ~13 usable structural highs means patches overlap heavily across the
            # ~190 examples in a run, so this turns millions of repeated random
            # SEG-Y trace reads into one sequential pass.
            il_centers = [kwargs['il_center'] for kwargs, _ in scenarios]
            xl_centers = [kwargs['xl_center'] for kwargs, _ in scenarios]
            cache_il_lo = max(inlines[0], int(min(il_centers) - il_extent // 2))
            cache_il_hi = min(inlines[-1], int(max(il_centers) + il_extent // 2))
            cache_xl_lo = max(xlines[0], int(min(xl_centers) - xl_extent // 2))
            cache_xl_hi = min(xlines[-1], int(max(xl_centers) + xl_extent // 2))
            cache_inline_axis = np.arange(cache_il_lo, cache_il_hi + 1)
            cache_xl_axis = np.arange(cache_xl_lo, cache_xl_hi + 1)
            cached_subvol = _read_subvolume(f, iline_map, cache_inline_axis, cache_xl_axis, samples_ms.size)

            skip_counts[split_name] = {'il_xl_bounds': 0, 'time_bounds': 0, 'footprint_exceeds_patch': 0,
                                        'unresolvable_relief': 0, 'horizon_lookup_gap': 0}
            for kwargs, label in scenarios:
                result, skip_reason = generate_example(
                    kwargs, label, cached_subvol, cache_inline_axis, cache_xl_axis, samples_ms,
                    horizon, dt_ms, il_extent, xl_extent, time_extent_ms,
                )
                if result is None:
                    skip_counts[split_name][skip_reason] += 1  # F9: track *why*, not just *that*
                    continue

                fname = f'example_{example_id:04d}.npz'
                np.savez_compressed(os.path.join(patches_dir, fname),
                                     attribute_stack=result['attribute_stack'],
                                     channel_names=np.array(result['channel_names']),
                                     mask=result['mask'])
                row = dict(result['label'], example_id=example_id, split=split_name, patch_file=fname)
                rows.append(row)
                example_id += 1

    labels = pd.DataFrame(rows)

    # F9: a mostly- or fully-skipped run finishing "successfully" with an empty
    # or tiny labels table is exactly the silent failure this guards against.
    total_skipped = sum(sum(counts.values()) for counts in skip_counts.values())
    total_scenarios = example_id + total_skipped
    if len(labels) == 0:
        raise RuntimeError(f'build_dataset produced zero examples - every scenario was skipped. '
                            f'Skip reasons by split: {skip_counts}')
    skip_rate = total_skipped / total_scenarios if total_scenarios else 0.0
    if skip_rate > max_skip_rate:
        warnings.warn(f'{skip_rate:.0%} of scenarios were skipped ({example_id}/{total_scenarios} kept, '
                       f'threshold {max_skip_rate:.0%}). Skip reasons by split: {skip_counts}')

    labels.to_parquet(os.path.join(output_dir, 'labels.parquet'), index=False)
    return labels
