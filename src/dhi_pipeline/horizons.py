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
from scipy.ndimage import minimum_filter, uniform_filter
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


def _find_structural_extrema(horizon_df, window, min_relief_ms, edge_margin, kind):
    """Shared logic for find_structural_highs / find_structural_lows."""
    grid = horizon_df.pivot_table(index='inline', columns='crossline', values='time_ms')
    grid_inlines = grid.index.values
    grid_xlines = grid.columns.values
    values = grid.values

    sign = 1 if kind == 'high' else -1  # highs = local minima in time; lows (synclines) = local maxima
    fill_value = np.nanmax(sign * values) * sign
    filled = np.where(np.isnan(values), fill_value, values)
    scored = sign * filled

    local_extreme = minimum_filter(scored, size=window, mode='nearest')
    local_mean = uniform_filter(scored, size=window * 3, mode='nearest')
    is_extreme = (scored == local_extreme) & (local_mean - scored >= min_relief_ms) & ~np.isnan(values)

    is_extreme[:edge_margin] = False
    is_extreme[-edge_margin:] = False
    is_extreme[:, :edge_margin] = False
    is_extreme[:, -edge_margin:] = False

    rows, cols = np.where(is_extreme)
    candidates = pd.DataFrame({
        'inline': grid_inlines[rows],
        'crossline': grid_xlines[cols],
        'time_ms': filled[rows, cols],
        'relief_ms': (local_mean - scored)[rows, cols],
    }).sort_values('relief_ms', ascending=False).reset_index(drop=True)

    # non-max suppression: nearby points on the same plateau shouldn't all show up as separate candidates
    kept_rows, kept_coords = [], []
    for _, row in candidates.iterrows():
        coord = np.array([row['inline'], row['crossline']])
        if all(np.linalg.norm(coord - k) > window for k in kept_coords):
            kept_rows.append(row)
            kept_coords.append(coord)
    return pd.DataFrame(kept_rows).reset_index(drop=True)


def find_structural_highs(horizon_df, window=25, min_relief_ms=10.0, edge_margin=40):
    """
    Local structural highs (shallowest points - local minima in time) on a
    matched horizon surface: candidate trap locations for positive DHI
    examples, since structural closures are where hydrocarbons actually
    accumulate.

    window: neighbourhood size (trace units) used to test for a local extremum.
    min_relief_ms: minimum time difference from the local window average, to
        filter out noise-level bumps that aren't real structural closures.
    edge_margin: excludes candidates within this many traces of the matched
        surface's edge, leaving room for a footprint + taper.

    Returns DataFrame(inline, crossline, time_ms, relief_ms), most prominent first.
    """
    return _find_structural_extrema(horizon_df, window, min_relief_ms, edge_margin, kind='high')


def find_structural_lows(horizon_df, window=25, min_relief_ms=10.0, edge_margin=40):
    """
    Local structural lows (deepest points - synclines) on a matched horizon
    surface. Useful for a physically-motivated hard negative: a bright,
    doublet-shaped, conformant reflection sitting in a syncline rather than
    an anticline is not a valid hydrocarbon trap (fluids don't accumulate
    there under normal buoyancy-driven trapping) despite an amplitude/
    polarity signature identical to a true positive - see Nanda (2021)'s
    point about flat spots needing the right structural position (cited in
    the notebook's research notes).
    """
    return _find_structural_extrema(horizon_df, window, min_relief_ms, edge_margin, kind='low')
