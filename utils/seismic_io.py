# Implementation developed with AI (Claude Code) assistance - see AI_USAGE.md.
"""
Utility functions for reading F3 SEG-Y data using segyio. 
Handles the non-rectangular survey geometry via built trace index. 
"""
import pickle
from pathlib import Path

import segyio
import numpy as np

def read_inline(f, iline_map, xlines, il, n_samples):
    """
    Returns array of shape (n_xlines, n_samples), NaN where trace missing.
    """
    out = np.full((len(xlines), n_samples), np.nan)
    for j, xl in enumerate(xlines):
        idx = iline_map.get((il, xl))
        if idx is not None:
            out[j] = f.trace[idx]
    return out

def read_crossline(f, iline_map, inlines, xl, n_samples):
    """Returns array of shape (n_inlines, n_xlines), NaN where trace missing."""
    out = np.full((len(inlines), n_samples), np.nan)
    for j, il in enumerate(inlines):
        idx = iline_map.get((il, xl))
        if idx is not None:
            out[j] = f.trace[idx]
    return out

def read_timeslice(f, iline_map, inlines, xlines, time_idx):
    """Returns array of shape (n_inlines, n_xlines), NaN where trace missing."""
    out = np.full((len(inlines), len(xlines)), np.nan)
    for (il, xl), trace_idx in iline_map.items():
        il_idx = inlines.index(il)
        xl_idx = xlines.index(xl)
        out[il_idx][xl_idx] = f.trace[trace_idx][time_idx]
    return out

def read_subvolume(f, iline_map, inlines, xlines, n_samples):
    """
    Returns array of shape (n_inlines, n_xlines, n_samples), NaN where trace missing.
    `inlines`/`xlines` are the specific (already-bounded) lists defining the sub-volume.
    """
    out = np.full((len(inlines), len(xlines), n_samples), np.nan)
    for i, il in enumerate(inlines):
        for j, xl in enumerate(xlines):
            idx = iline_map.get((il, xl))
            if idx is not None:
                out[i, j] = f.trace[idx]
    return out

def build_trace_index(segy_path):
    """Build the F3 (inline, crossline) -> trace-number lookup from headers."""
    segy_path = Path(segy_path)

    if not segy_path.is_file():
        raise FileNotFoundError(f'SEG-Y file does not exist: {segy_path}')

    with segyio.open(str(segy_path), ignore_geometry=True) as segy:
        inline_headers = segy.attributes(
            segyio.TraceField.INLINE_3D
        )[:]
        crossline_headers = segy.attributes(
            segyio.TraceField.CROSSLINE_3D
        )[:]

    if len(inline_headers) != len(crossline_headers):
        raise RuntimeError(
            'Inline and crossline header arrays have different lengths'
        )
    if len(inline_headers) == 0:
        raise RuntimeError(f'SEG-Y contains no traces: {segy_path}')

    iline_map = {
        (int(inline), int(crossline)): trace_index
        for trace_index, (inline, crossline) in enumerate(
            zip(inline_headers, crossline_headers)
        )
    }
    inlines = sorted({inline for inline, _ in iline_map})
    xlines = sorted({crossline for _, crossline in iline_map})

    return {
        'iline_map': iline_map,
        'inlines': inlines,
        'xlines': xlines,
    }


def load_or_build_trace_index(
    segy_path,
    cache_path,
    force_rebuild=False,
):
    """Load the cached trace index, or build it automatically if absent."""
    cache_path = Path(cache_path)

    if cache_path.is_file() and not force_rebuild:
        with cache_path.open('rb') as handle:
            index = pickle.load(handle)

        required = {'iline_map', 'inlines', 'xlines'}
        missing = required - set(index)
        if missing:
            raise RuntimeError(
                f'Trace-index cache is missing keys: {sorted(missing)}'
            )

        return index

    index = build_trace_index(segy_path)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = cache_path.with_suffix(cache_path.suffix + '.tmp')

    with temporary_path.open('wb') as handle:
        pickle.dump(index, handle)

    temporary_path.replace(cache_path)
    return index