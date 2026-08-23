# Implementation developed with AI (Claude Code) assistance - see AI_USAGE.md.
"""Round 2, P1 item 1: the lag-swept doublet must actually see a separation
that a fixed lag-1 check misses - the whole point of the change (Aziz's
ROUND2_PLAN.md: a fixed 1-sample lag can't see tier5_resolved's ~17-23ms
reflector separations)."""
import numpy as np
import pytest

from src.dhi_pipeline.ratio_features import compute_doublet_autocorrelation


def _widely_separated_doublet(separation_samples, n_time=21, spatial=3):
    """Uniform (il, xl, t) volume: a +1 sample then a -1 sample `separation_samples`
    apart, identical at every spatial position - clean, deterministic doublet."""
    trace = np.zeros(n_time)
    mid = n_time // 2
    trace[mid] = 1.0
    trace[mid + separation_samples] = -1.0
    return np.tile(trace, (spatial, spatial, 1))


def test_lag1_default_reproduces_v1_and_misses_wide_separation():
    amplitude = _widely_separated_doublet(separation_samples=5)
    feature = compute_doublet_autocorrelation(amplitude, spatial_radius=1, time_halfwidth=10, lags=(1,))
    # adjacent samples around two isolated impulses 5 apart don't anti-correlate at lag 1
    assert feature == pytest.approx(1.0)


def test_swept_lags_detect_the_wide_separation():
    amplitude = _widely_separated_doublet(separation_samples=5)
    lag1_only = compute_doublet_autocorrelation(amplitude, spatial_radius=1, time_halfwidth=10, lags=(1,))
    swept = compute_doublet_autocorrelation(amplitude, spatial_radius=1, time_halfwidth=10, lags=range(1, 6))
    assert swept > lag1_only + 0.3  # clearly stronger doublet signal once lag 5 is checked
    assert swept == pytest.approx(1.5)


def test_swept_lags_reduce_to_lag1_for_a_true_1_sample_doublet():
    amplitude = _widely_separated_doublet(separation_samples=1)
    lag1_only = compute_doublet_autocorrelation(amplitude, spatial_radius=1, time_halfwidth=10, lags=(1,))
    swept = compute_doublet_autocorrelation(amplitude, spatial_radius=1, time_halfwidth=10, lags=range(1, 6))
    assert swept == pytest.approx(lag1_only)
