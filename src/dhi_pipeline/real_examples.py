"""
Real (non-synthetic) DHI examples from the F3 Demo project's own expert
interpretation - OpendTect PickSet files `Chimneys_yes.pck`/`Chimneys_no.pck`
(confirmed gas-chimney locations and confirmed non-chimney controls), matched
onto the trace grid and extracted as patches in the same on-disk format as
the synthetic dataset (dataset.build_dataset). Lets the detector be evaluated
against real, expert-labelled anomalies instead of only against injections
this project generated itself.
"""

import os

import numpy as np
import pandas as pd
import segyio
from scipy.spatial import cKDTree

from .attributes import compute_attribute_stack
from .dataset import _read_subvolume


def load_pickset(path):
    """
    Load an OpendTect PickSet Group file (`.pck`): a short header (format/
    group/date/'!'/ref/color/size/marker/connect lines) followed by
    whitespace-separated data rows. Two variants seen in this project's data:
    plain discrete picks (exactly x, y, time_s - e.g. Chimneys_yes/no.pck),
    and polygon-outline picksets with extra trailing per-vertex metadata
    columns (e.g. Shallow_Bright_Spot.pck: x, y, time_s, then 3 more numbers
    - vertex/connector flags, not coordinates). Only the first 3 fields are
    ever coordinates, so rows are read leniently (>=3 fields, extras
    ignored) rather than requiring an exact count. Header lines have <3
    fields or don't parse as floats, so they're skipped either way -
    robust to the exact header length and to which variant this file is.

    Returns DataFrame(x, y, time_ms).
    """
    rows = []
    with open(path) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                x, y, time_s = (float(p) for p in parts[:3])
            except ValueError:
                continue
            rows.append((x, y, time_s * 1000.0))
    return pd.DataFrame(rows, columns=['x', 'y', 'time_ms'])


def match_picks_to_grid(picks_df, ilxl_array, xy_array, max_distance=50.0):
    """
    Nearest-neighbour match picks (x, y) onto the survey's (inline,
    crossline) trace grid - same approach as horizons.load_horizon_surface.
    Picks farther than max_distance metres from any trace (survey-edge
    artefacts) are dropped rather than silently matched to a distant, wrong
    trace.
    """
    tree = cKDTree(xy_array)
    distances, indices = tree.query(picks_df[['x', 'y']].values)
    matched = pd.DataFrame({
        'inline': ilxl_array[indices, 0],
        'crossline': ilxl_array[indices, 1],
        'time_ms': picks_df['time_ms'].values,
        'distance': distances,
    })
    return matched[matched['distance'] <= max_distance].reset_index(drop=True)


def filter_to_patchable(matched_df, inlines, xlines, samples_ms,
                         il_extent=160, xl_extent=160, time_extent_ms=500):
    """
    Flag matched picks where a full (il_extent x xl_extent x time_extent_ms)
    patch centred on the pick fits inside the survey's actual recorded
    inline/crossline/time range - the same il_xl_bounds/time_bounds check
    dataset.generate_example uses for synthetic scenarios, so real patches
    get the same guaranteed shape instead of being silently clipped at
    survey edges.
    """
    il_lo = matched_df['inline'] - il_extent // 2
    il_hi = matched_df['inline'] + il_extent // 2
    xl_lo = matched_df['crossline'] - xl_extent // 2
    xl_hi = matched_df['crossline'] + xl_extent // 2
    t_lo = matched_df['time_ms'] - time_extent_ms / 2
    t_hi = matched_df['time_ms'] + time_extent_ms / 2

    patchable = (
        (il_lo >= inlines[0]) & (il_hi <= inlines[-1]) &
        (xl_lo >= xlines[0]) & (xl_hi <= xlines[-1]) &
        (t_lo >= samples_ms[0]) & (t_hi <= samples_ms[-1])
    )
    out = matched_df.copy()
    out['patchable'] = patchable
    return out


def extract_real_patch(il_center, xl_center, center_time_ms, cached_subvol,
                        cache_inline_axis, cache_xl_axis, samples_ms, dt_ms,
                        il_extent=160, xl_extent=160, time_extent_ms=500):
    """
    Slice a real (uninjected) patch out of a pre-cached sub-volume and
    compute its attribute stack - same shape/channel format as a synthetic
    example's attribute_stack. No footprint mask: there's nothing injected
    to mask, real examples carry their positive/negative label at the pick
    level instead (see build_real_example_set).
    """
    il_lo, il_hi = int(il_center - il_extent // 2), int(il_center + il_extent // 2)
    xl_lo, xl_hi = int(xl_center - xl_extent // 2), int(xl_center + xl_extent // 2)
    il_off = il_lo - cache_inline_axis[0]
    xl_off = xl_lo - cache_xl_axis[0]
    n_il, n_xl = il_hi - il_lo, xl_hi - xl_lo
    raw = cached_subvol[il_off:il_off + n_il, xl_off:xl_off + n_xl, :]

    t_mask = (samples_ms >= center_time_ms - time_extent_ms / 2) & \
             (samples_ms <= center_time_ms + time_extent_ms / 2)
    raw_patch = raw[:, :, t_mask]
    n_missing = int(np.isnan(raw_patch).sum() // raw_patch.shape[-1])

    # Hilbert transform can't handle NaN gaps (real survey-edge missing traces) - fill
    # with 0 before computing attributes, same as generate_example does for synthetics.
    raw_patch_filled = np.nan_to_num(raw_patch, nan=0.0)
    attribute_stack, channel_names = compute_attribute_stack(raw_patch_filled, dt_ms / 1000.0)

    return dict(attribute_stack=attribute_stack.astype(np.float32), channel_names=channel_names,
                il_lo=il_lo, il_hi=il_hi, xl_lo=xl_lo, xl_hi=xl_hi, n_missing_traces=n_missing)


def build_real_example_set(output_dir, segy_path, yes_pck_path, no_pck_path,
                            ilxl_array, xy_array, iline_map, inlines, xlines,
                            il_extent=160, xl_extent=160, time_extent_ms=500,
                            train_inline_range=(150, 345), test_inline_range=(365, 670)):
    """
    Build the real-anomaly evaluation set: matches Chimneys_yes/Chimneys_no
    OpendTect picks (F3 Demo project's own expert interpretation, not this
    project's synthetic injections) onto the trace grid, keeps only picks
    where a full patch fits inside the survey, extracts a real (uninjected)
    patch + attribute stack per pick, and writes one .npz per example plus a
    combined labels table - same on-disk shape as the synthetic dataset
    (dataset.build_dataset), so the detector notebook can load real and
    synthetic examples the same way.

    `inline_split` in the labels table is informational only (which side of
    the synthetic train/test inline split each real pick falls on) - it does
    NOT gate which real picks get built. The detector was never trained on
    any real pick, so there's no leakage concern restricting evaluation to
    the synthetic "test" band; every patchable real pick is built.

    Returns the labels DataFrame (patchable picks only - already written to
    disk alongside the .npz patches).

    Writes to `{output_dir}/labels_real.parquet`, not `labels.parquet` -
    dataset.build_dataset already owns that filename for the synthetic set,
    and this is commonly pointed at the same output_dir so the two example
    sets' patches/ live together; a shared filename would let one silently
    overwrite the other.
    """
    patches_dir = os.path.join(output_dir, 'patches')
    os.makedirs(patches_dir, exist_ok=True)

    picksets = {'real_chimney': (yes_pck_path, True), 'real_non_chimney': (no_pck_path, False)}
    matched_all = []
    for kind, (path, is_dhi) in picksets.items():
        picks = load_pickset(path)
        matched = match_picks_to_grid(picks, ilxl_array, xy_array)
        matched['is_dhi'] = is_dhi
        matched['kind'] = kind
        matched_all.append(matched)
    matched_all = pd.concat(matched_all, ignore_index=True)

    with segyio.open(segy_path, ignore_geometry=True) as f:
        samples_ms = f.samples.astype(float)
        dt_ms = float(samples_ms[1] - samples_ms[0])

        matched_all = filter_to_patchable(matched_all, inlines, xlines, samples_ms,
                                           il_extent, xl_extent, time_extent_ms)
        patchable = matched_all[matched_all['patchable']].reset_index(drop=True)
        if len(patchable) == 0:
            raise RuntimeError(
                f'no real picks are patchable: {len(matched_all)} matched picks, none with a full '
                f'{il_extent}x{xl_extent}x{time_extent_ms}ms patch fitting inside the survey range.'
            )

        # Same F10 approach as build_dataset: read the union bounding box once rather than
        # per-pick, clipped to the survey's actual valid range (already guaranteed to cover
        # every patchable pick's patch, since patchable itself required that).
        cache_il_lo = max(inlines[0], int(patchable['inline'].min() - il_extent // 2))
        cache_il_hi = min(inlines[-1], int(patchable['inline'].max() + il_extent // 2))
        cache_xl_lo = max(xlines[0], int(patchable['crossline'].min() - xl_extent // 2))
        cache_xl_hi = min(xlines[-1], int(patchable['crossline'].max() + xl_extent // 2))
        cache_inline_axis = np.arange(cache_il_lo, cache_il_hi + 1)
        cache_xl_axis = np.arange(cache_xl_lo, cache_xl_hi + 1)
        cached_subvol = _read_subvolume(f, iline_map, cache_inline_axis, cache_xl_axis, samples_ms.size)

        rows = []
        for example_id, pick in patchable.iterrows():
            result = extract_real_patch(
                pick['inline'], pick['crossline'], pick['time_ms'], cached_subvol,
                cache_inline_axis, cache_xl_axis, samples_ms, dt_ms,
                il_extent, xl_extent, time_extent_ms,
            )
            fname = f'real_{example_id:04d}.npz'
            np.savez_compressed(os.path.join(patches_dir, fname),
                                 attribute_stack=result['attribute_stack'],
                                 channel_names=np.array(result['channel_names']))

            train_lo, train_hi = train_inline_range
            test_lo, test_hi = test_inline_range
            if train_lo <= pick['inline'] <= train_hi:
                inline_split = 'train'
            elif test_lo <= pick['inline'] <= test_hi:
                inline_split = 'test'
            else:
                inline_split = 'neither'

            rows.append(dict(
                example_id=example_id, split='real', kind=pick['kind'], is_dhi=pick['is_dhi'],
                inline_split=inline_split, patch_file=fname, il_lo=result['il_lo'], il_hi=result['il_hi'],
                xl_lo=result['xl_lo'], xl_hi=result['xl_hi'],
                # il_center/xl_center (not center_inline/center_crossline) to match
                # dataset.build_dataset's synthetic labels schema - dhi_detector.ipynb's
                # site_id/grouping logic reads these column names directly.
                il_center=pick['inline'], xl_center=pick['crossline'], center_time_ms=pick['time_ms'],
                match_distance_m=pick['distance'], n_missing_traces=result['n_missing_traces'],
            ))

    labels = pd.DataFrame(rows)
    labels.to_parquet(os.path.join(output_dir, 'labels_real.parquet'), index=False)
    return labels
