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
