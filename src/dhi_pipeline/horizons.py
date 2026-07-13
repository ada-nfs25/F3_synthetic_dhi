"""
Real horizon geometry for structural conformance in DHI injection.

Loads Zenodo interpretation horizon picks (x, y, time_ms), matches them to
the survey's (inline, crossline) trace grid via nearest-neighbour lookup on
real-world coordinates (same approach as notebook 01's horizon-alignment
check), and exposes a queryable horizon-time surface. Used so injected
reservoirs follow real structural shape instead of sitting on an artificial
flat top.
"""

import os
import pickle

import numpy as np
import pandas as pd
import segyio
from scipy.spatial import cKDTree

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
COORDS_CACHE_PATH = os.path.normpath(os.path.join(_THIS_DIR, '..', '..', 'data', 'f3_coords.pkl'))


def build_coordinate_lookup(segy_path, cache_path=COORDS_CACHE_PATH, force_rebuild=False):
    """
    Scan SEG-Y headers once to get real-world (x, y) for every (inline,
    crossline) trace - needed to match horizon picks (given in x/y) onto the
    trace grid. Header-only (vectorised via segyio.attributes), so this is
    fast even across the full ~600k-trace survey; still cached to disk since
    there's no reason to repeat it every session.

    Returns {'ilxl_array': (N, 2) int array, 'xy_array': (N, 2) float array}.
    """
    if not force_rebuild and os.path.exists(cache_path):
        with open(cache_path, 'rb') as fh:
            return pickle.load(fh)

    with segyio.open(segy_path, ignore_geometry=True) as f:
        il_headers = f.attributes(segyio.TraceField.INLINE_3D)[:]
        xl_headers = f.attributes(segyio.TraceField.CROSSLINE_3D)[:]
        x_headers = f.attributes(segyio.TraceField.CDP_X)[:]
        y_headers = f.attributes(segyio.TraceField.CDP_Y)[:]
        scalar = abs(f.header[0][segyio.TraceField.SourceGroupScalar]) or 1

    ilxl_array = np.column_stack([il_headers, xl_headers]).astype(int)
    xy_array = np.column_stack([x_headers, y_headers]).astype(float) / scalar

    result = {'ilxl_array': ilxl_array, 'xy_array': xy_array}
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, 'wb') as fh:
        pickle.dump(result, fh)
    return result


def load_horizon_surface(horizon_path, ilxl_array, xy_array, max_distance=50.0):
    """
    Load a horizon .xyz pick file and match every pick to its nearest trace
    via a KD-tree on real-world coordinates.

    Returns a DataFrame(inline, crossline, time_ms, distance). Picks farther
    than `max_distance` metres from any trace (survey-edge artifacts - see
    notebook 01, where the max observed mismatch was ~975m vs a ~3.5m mean)
    are dropped rather than silently matched to a distant, wrong trace.
    """
    picks = pd.read_csv(horizon_path, sep=r'\s+', header=None, names=['x', 'y', 'time_ms'])
    tree = cKDTree(xy_array)
    distances, indices = tree.query(picks[['x', 'y']].values)

    surface = pd.DataFrame({
        'inline': ilxl_array[indices, 0],
        'crossline': ilxl_array[indices, 1],
        'time_ms': picks['time_ms'].values,
        'distance': distances,
    })
    return surface[surface['distance'] <= max_distance].reset_index(drop=True)


class HorizonSurface:
    """Queryable (inline, crossline) -> time_ms lookup for a matched horizon."""

    def __init__(self, surface_df):
        self.lookup = {
            (int(row.inline), int(row.crossline)): row.time_ms
            for row in surface_df.itertuples()
        }
        self._ilxl_array = surface_df[['inline', 'crossline']].values
        self._tree = cKDTree(self._ilxl_array)
        self._times = surface_df['time_ms'].values

    def time_at(self, il, xl):
        """Exact match if this (il, xl) was picked, else nearest picked location's time (gaps happen near survey edges)."""
        exact = self.lookup.get((il, xl))
        if exact is not None:
            return exact
        _, idx = self._tree.query([il, xl])
        return self._times[idx]
