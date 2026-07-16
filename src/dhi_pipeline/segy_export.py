"""
Export injected scenarios as standalone SEG-Y files.

Unlike the .npz patch dataset (built for model training), this writes a real,
loadable SEG-Y sub-volume per scenario - a real chunk of the F3 survey with one
injected DHI or hard negative, openable in normal seismic interpretation
software (OpendTect, Petrel, ...) so the result can be visually judged, not just
read back as an array.
"""

import numpy as np
import segyio

from .dataset import _scenario_center_time
from .injection import estimate_amplitude_scale, inject_dhi_anomaly_3d


def export_scenario_to_segy(kwargs, label, f, iline_map, inlines, xlines, horizon,
                             output_path, il_extent=160, xl_extent=160, reference_rc=0.05):
    """
    Inject one scenario onto a real sub-volume (full trace length, not the
    500ms window used for ML patches - more context for visual inspection)
    and write it out as a standalone SEG-Y file, copying real trace headers
    (inline/crossline/CDP X/Y/...) from the source survey.

    Returns the output path, or None if the sub-volume would run off the
    survey's valid il/xl range (same edge behaviour as generate_example).
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
    n_traces = len(patch_inline_axis) * len(patch_xl_axis)

    raw = np.full((len(patch_inline_axis), len(patch_xl_axis), n_samples), np.nan)
    trace_indices = np.full((len(patch_inline_axis), len(patch_xl_axis)), -1, dtype=int)
    for i, il in enumerate(patch_inline_axis):
        for j, xl in enumerate(patch_xl_axis):
            idx = iline_map.get((int(il), int(xl)))
            if idx is not None:
                raw[i, j] = f.trace[idx]
                trace_indices[i, j] = idx

    time_axis_ms = f.samples.astype(float)
    amp_scale = estimate_amplitude_scale(raw, reference_rc=reference_rc)
    injected, _ = inject_dhi_anomaly_3d(
        raw, time_axis_ms, patch_inline_axis, patch_xl_axis, horizon,
        amplitude_scale=amp_scale, **kwargs,
    )
    injected_filled = np.nan_to_num(injected, nan=0.0)

    spec = segyio.spec()
    spec.samples = f.samples
    spec.tracecount = n_traces
    spec.format = f.format
    spec.sorting = segyio.TraceSortingFormat.CROSSLINE_SORTING

    with segyio.create(str(output_path), spec) as dst:
        dst.bin = f.bin
        trace_i = 0
        for i, il in enumerate(patch_inline_axis):
            for j, xl in enumerate(patch_xl_axis):
                src_idx = trace_indices[i, j]
                if src_idx >= 0:
                    dst.header[trace_i] = f.header[src_idx]
                else:
                    # gap in the survey's acquisition outline - no real trace to copy headers
                    # from; still write correct inline/crossline so the geometry stays valid
                    dst.header[trace_i] = {
                        segyio.TraceField.INLINE_3D: int(il),
                        segyio.TraceField.CROSSLINE_3D: int(xl),
                    }
                dst.trace[trace_i] = injected_filled[i, j].astype(f.dtype)
                trace_i += 1

    return output_path
