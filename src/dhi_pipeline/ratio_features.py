# Implementation developed with AI (Claude Code) assistance - see AI_USAGE.md.
"""
Window/whole-patch ratio features built on top of an attribute stack
(src/dhi_pipeline/attributes.py's compute_attribute_stack output) - the
scalar summary features that actually feed the XGBoost classifier, as
opposed to the per-pixel channels themselves.

F4/C4 (regional-bias fix): these are ratios, not absolute values, so they
don't carry the survey's arbitrary amplitude gain the way a raw envelope or
RMS reading would - see attributes.py's module docstring for why that
distinction mattered.

Round 2, P1 item 1 (Aziz, ROUND2_PLAN.md): the frozen v1 model's
doublet_autocorrelation (scripts/run_blind_predictions.py,
scripts/inhouse_replication_thick_dim.py) checks only lag-1 sample
autocorrelation. That's a hard constant tied to v1's training, so this
module does NOT modify those scripts - doing so would feed the frozen model
a differently-defined feature than it was trained on. compute_doublet_
autocorrelation below is the v2-candidate: it sweeps a range of lags and
keeps the strongest anti-correlation found, matching a real reflector
separation at whatever thickness produced it, not just a 1-sample gap. It's
a superset of the old behaviour (lags=(1,) reproduces v1's feature exactly -
see the docstring and tests/test_ratio_features.py).
"""

import numpy as np

FEATURE_CHANNELS = ['envelope', 'sweetness', 'inst_freq', 'band_ratio']
NEAR_TOP_HALF_MS = 60
DOUBLET_SPATIAL_RADIUS = 5
DOUBLET_TIME_HALFWIDTH = 15
SIGNED_POLARITY_SPATIAL_RADIUS = 2
V2_DOUBLET_LAGS = range(1, 6)
"""~4-20ms at 4ms dt - see compute_doublet_autocorrelation's docstring."""


def compute_doublet_autocorrelation(amplitude, spatial_radius=5, time_halfwidth=15, lags=(1,)):
    """
    Doublet cue: 1.0 minus the least negative (i.e. strongest anti-
    correlated) median lag-k autocorrelation found across `lags`, over a
    central (2*spatial_radius+1) x (2*spatial_radius+1) x (2*time_halfwidth+1)
    window of `amplitude` (an (il, xl, t) array). A genuine doublet
    (trough-peak or peak-trough reflector pair) anti-correlates strongly at
    the lag matching its separation; two independent reflectors don't.

    lags=(1,) (default) reproduces the frozen v1 model's fixed-lag feature
    exactly - only ever checks a 1-sample (4ms) separation. Round 2, P1 item
    1 (Aziz): that can't see a thicker bed's wider doublet separation -
    tier5_resolved's 18-26m thickness range corresponds to roughly 17-23ms
    (4-6 samples at 4ms dt) two-way-time separation between the top/base
    reflections. Pass lags=range(1, 6) (~4-20ms) to sweep across that instead
    - this is a genuinely new feature definition, meant for evaluating and
    retraining a v2 model against the P0-diversified dataset, not for
    feeding the frozen v1 model (see module docstring).
    """
    ni, nx, nt = amplitude.shape
    mi, mx, mt = ni // 2, nx // 2, nt // 2
    central = amplitude[mi - spatial_radius:mi + spatial_radius + 1,
                         mx - spatial_radius:mx + spatial_radius + 1,
                         mt - time_halfwidth:mt + time_halfwidth + 1]
    traces = central.reshape(-1, 2 * time_halfwidth + 1)
    traces = traces - np.median(traces, axis=1, keepdims=True)
    scale = np.sqrt(np.sum(traces ** 2, axis=1)) + 1e-10

    best_acf = None
    for lag in lags:
        acf_lag = np.sum(traces[:, :-lag] * traces[:, lag:], axis=1) / scale ** 2
        median_acf = np.median(acf_lag)
        if best_acf is None or median_acf < best_acf:
            best_acf = median_acf

    return 1.0 - best_acf


def ratio_features_from_stack(stack, channel_names, doublet_lags=V2_DOUBLET_LAGS):
    """
    The v2 candidate feature set: same 14 features as the frozen v1 model
    (irp-nfs25/notebooks/dhi_xgb_detector.ipynb's ratio_features_from_stack -
    12 window/whole-patch ratios over FEATURE_CHANNELS + doublet_
    autocorrelation + signed_polarity), with two changes carried in from the
    stack itself rather than from this function:

    - band_ratio (one of FEATURE_CHANNELS) is whatever channel_names/stack
      already contain, so if `stack` came from the current
      attributes.compute_attribute_stack, it's already the P1 item 2
      survey-adaptive version, not the old fixed-15/45Hz one - nothing here
      needs to know which.
    - doublet_autocorrelation defaults to the swept-lag P1 item 1 version
      (V2_DOUBLET_LAGS) instead of v1's fixed lag=1. Pass doublet_lags=(1,)
      to reproduce v1's feature exactly for a side-by-side ablation.

    Kept in this module rather than duplicated per-script (unlike v1's
    scattered copies - see this module's docstring) since this is the
    single feature definition v2 training and any future v2 scoring should
    both call.
    """
    n_samples = stack.shape[-1]
    dt_ms = 4.0
    mid = n_samples // 2

    near_top_half = int(round(NEAR_TOP_HALF_MS / dt_ms))
    windows = {'near_top': (max(mid - near_top_half, 0), min(mid + near_top_half, n_samples))}

    features = {}
    for ch in FEATURE_CHANNELS:
        arr = stack[channel_names.index(ch)]
        for window_name, (lo, hi) in windows.items():
            window = arr[..., lo:hi]
            for stat_name, stat_fn in [('mean', np.mean), ('p90', lambda a: np.percentile(a, 90)),
                                        ('maxabs', lambda a: np.max(np.abs(a)))]:
                window_stat = stat_fn(window)
                whole_stat = stat_fn(arr)
                features[f'{ch}_{stat_name}_{window_name}_ratio'] = window_stat / (whole_stat + 1e-10)

    amplitude = stack[channel_names.index('amplitude')]
    features['doublet_autocorrelation'] = compute_doublet_autocorrelation(
        amplitude, spatial_radius=DOUBLET_SPATIAL_RADIUS, time_halfwidth=DOUBLET_TIME_HALFWIDTH,
        lags=doublet_lags,
    )

    ni, nx, nt = amplitude.shape
    mi, mx, mt = ni // 2, nx // 2, nt // 2
    centre_vals = amplitude[mi - SIGNED_POLARITY_SPATIAL_RADIUS:mi + SIGNED_POLARITY_SPATIAL_RADIUS + 1,
                             mx - SIGNED_POLARITY_SPATIAL_RADIUS:mx + SIGNED_POLARITY_SPATIAL_RADIUS + 1, mt]
    features['signed_polarity'] = np.sum(centre_vals) / (np.sum(np.abs(centre_vals)) + 1e-10)

    return features
