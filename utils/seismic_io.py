"""
Utility functions for reading F3 SEG-Y data using segyio. 
Handles the non-rectangular survey geometry via built trace index. 
"""

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