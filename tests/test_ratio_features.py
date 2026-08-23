# Implementation developed with AI (Claude Code) assistance - see AI_USAGE.md.
"""Round 2, P1 item 1: the lag-swept doublet must actually see a separation
that a fixed lag-1 check misses - the whole point of the change (Aziz's
ROUND2_PLAN.md: a fixed 1-sample lag can't see tier5_resolved's ~17-23ms
reflector separations)."""
import numpy as np
import pytest

from src.dhi_pipeline.attributes import compute_attribute_stack
from src.dhi_pipeline.ratio_features import compute_doublet_autocorrelation, ratio_features_from_stack


def _v1_ratio_features_from_stack(stack, channel_names):
    """Verbatim copy of irp-nfs25/notebooks/dhi_xgb_detector.ipynb's
    ratio_features_from_stack (== scripts/run_blind_predictions.py's copy) -
    the frozen v1 reference, kept only in this test so ratio_features_from_
    stack(doublet_lags=(1,)) can be checked against it directly."""
    n_samples = stack.shape[-1]
    dt_ms = 4.0
    mid = n_samples // 2
    near_top_half = int(round(60 / dt_ms))
    windows = {'near_top': (max(mid - near_top_half, 0), min(mid + near_top_half, n_samples))}

    features = {}
    for ch in ['envelope', 'sweetness', 'inst_freq', 'band_ratio']:
        arr = stack[channel_names.index(ch)]
        for window_name, (lo, hi) in windows.items():
            window = arr[..., lo:hi]
            for stat_name, stat_fn in [('mean', np.mean), ('p90', lambda a: np.percentile(a, 90)),
                                        ('maxabs', lambda a: np.max(np.abs(a)))]:
                features[f'{ch}_{stat_name}_{window_name}_ratio'] = stat_fn(window) / (stat_fn(arr) + 1e-10)

    amplitude = stack[channel_names.index('amplitude')]
    ni, nx, nt = amplitude.shape
    mi, mx, mt = ni // 2, nx // 2, nt // 2
    central = amplitude[mi - 5:mi + 6, mx - 5:mx + 6, mt - 15:mt + 16]
    traces = central.reshape(-1, 31)
    traces = traces - np.median(traces, axis=1, keepdims=True)
    scale = np.sqrt(np.sum(traces ** 2, axis=1)) + 1e-10
    acf_lag1 = np.sum(traces[:, :-1] * traces[:, 1:], axis=1) / scale ** 2
    features['doublet_autocorrelation'] = 1.0 - np.median(acf_lag1)

    centre_vals = amplitude[mi - 2:mi + 3, mx - 2:mx + 3, mt]
    features['signed_polarity'] = np.sum(centre_vals) / (np.sum(np.abs(centre_vals)) + 1e-10)
    return features


def test_v2_lag1_reproduces_v1_feature_values_exactly():
    rng = np.random.default_rng(0)
    amplitude = rng.normal(size=(96, 96, 125)).astype(np.float32)
    stack, channel_names = compute_attribute_stack(amplitude, dt_s=0.004)

    v1 = _v1_ratio_features_from_stack(stack, channel_names)
    v2 = ratio_features_from_stack(stack, channel_names, doublet_lags=(1,))

    assert v1.keys() == v2.keys()
    for key in v1:
        assert v1[key] == pytest.approx(v2[key], abs=1e-9), key


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
