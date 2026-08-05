"""Shared fixtures for the injection.py regression tests (F1/F2/F3, blind-test integrity).

Uses small synthetic grids and duck-typed fake horizons rather than real F3 data - these
tests check the injection *mechanics* (event placement, sign, timing) against known-by-
construction ground truth, not anything about the real survey.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest


class DippingHorizon:
    """time_at(il, xl) = base_time_ms + dip_ms_per_trace * (the chosen axis's value).

    Mirrors the review's own reproduction setup (a horizon dipping at a constant
    rate along one axis) without needing a real HorizonSurface/SEG-Y survey.
    """
    def __init__(self, base_time_ms=900.0, dip_ms_per_trace=1.0, dip_axis='il'):
        self.base_time_ms = base_time_ms
        self.dip_ms_per_trace = dip_ms_per_trace
        self.dip_axis = dip_axis

    def time_at(self, il, xl):
        pos = il if self.dip_axis == 'il' else xl
        return self.base_time_ms + self.dip_ms_per_trace * pos


class FlatHorizon:
    """Constant top_time_ms everywhere - zero structural relief."""
    def __init__(self, time_ms=900.0):
        self.time_ms = time_ms

    def time_at(self, il, xl):
        return self.time_ms


@pytest.fixture
def dt_ms():
    return 4.0


@pytest.fixture
def small_grid(dt_ms):
    """A small (inline, crossline, time) grid - big enough to hold a real footprint,
    small enough that inject_dhi_anomaly_3d's per-voxel loop stays fast in tests."""
    n_il, n_xl, n_t = 40, 40, 300
    inline_axis = np.arange(n_il)
    xl_axis = np.arange(n_xl)
    time_axis_ms = np.arange(n_t) * dt_ms
    volume = np.zeros((n_il, n_xl, n_t))
    return volume, time_axis_ms, inline_axis, xl_axis


@pytest.fixture
def base_injection_kwargs():
    """A physically reasonable, tier4-like parameter set - real enough to exercise
    flat_spot/polarity_reversal/sag together, matching how the real pipeline uses them."""
    return dict(
        thickness_m=14.0, velocity_mps=2000.0, reflection_coefficient=-0.1545,
        freq_hz=30.0, il_center=20, xl_center=20, il_radius=15, xl_radius=15,
        amplitude_scale=1.0,
    )
