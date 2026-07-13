"""
Randomised scenario sampling for DHI dataset generation.

Stage 2 (positive diversity): continuous thickness/RC within each severity
tier's regime rather than four fixed points, randomised footprint size/
orientation, independent flat-spot/polarity-reversal inclusion, sampled
across many real structural highs rather than one hand-picked location.

Stage 3 (hard negatives): the same amplitude/polarity machinery, deliberately
placed without genuine trap geometry - so a detector trained on this has to
learn "conformance + doublet", not "bright = DHI" (see Nanda 2021's warning
that bright/flat spots can come from volcanics, coal, overpressured sand, or
tuning - not just hydrocarbons).

Each sampler returns a plain dict of keyword arguments ready to pass to
`inject_dhi_anomaly_3d`, plus a `label` dict describing what was generated
(for the eventual patch+label dataset export).
"""

import numpy as np

from .injection import RC_GAS_SAND

# Continuous ranges per tier, centred on the fixed SEVERITY_TIERS values but
# with real spread - a trained detector should see a continuum near each
# regime, not four exact repeated points.
TIER_RANGES = {
    'tier1_subtle': dict(thickness_range=(1.5, 3.0), rc_frac_range=(0.30, 0.50),
                          flat_spot_prob=0.05, polarity_reversal_prob=0.05),
    'tier2_approaching': dict(thickness_range=(4.0, 6.0), rc_frac_range=(0.55, 0.75),
                               flat_spot_prob=0.15, polarity_reversal_prob=0.15),
    'tier3_at_tuning': dict(thickness_range=(6.0, 8.0), rc_frac_range=(0.75, 0.95),
                             flat_spot_prob=0.30, polarity_reversal_prob=0.30),
    'tier4_obvious': dict(thickness_range=(11.0, 18.0), rc_frac_range=(0.90, 1.00),
                           flat_spot_prob=0.70, polarity_reversal_prob=0.70),
}

FOOTPRINT_RADIUS_RANGE = (30, 80)  # traces, both il and xl radius sampled independently


def _sample_footprint(rng):
    return dict(
        il_radius=rng.uniform(*FOOTPRINT_RADIUS_RANGE),
        xl_radius=rng.uniform(*FOOTPRINT_RADIUS_RANGE),
        rotation_deg=rng.uniform(0, 180),
    )


def sample_positive_scenario(tier_name, rng, structural_highs, velocity_mps, freq_hz):
    """
    Sample a randomised positive-example scenario for a given severity tier,
    at a randomly chosen real structural high.

    structural_highs: DataFrame from `horizons.find_structural_highs`.
    Returns (injection_kwargs, label).
    """
    ranges = TIER_RANGES[tier_name]
    site = structural_highs.iloc[rng.integers(len(structural_highs))]

    thickness_m = rng.uniform(*ranges['thickness_range'])
    rc_frac = rng.uniform(*ranges['rc_frac_range'])
    flat_spot = rng.random() < ranges['flat_spot_prob']
    polarity_reversal = rng.random() < ranges['polarity_reversal_prob']

    kwargs = dict(
        velocity_mps=velocity_mps, freq_hz=freq_hz,
        thickness_m=thickness_m, reflection_coefficient=RC_GAS_SAND * rc_frac,
        il_center=site['inline'], xl_center=site['crossline'],
        flat_spot=flat_spot, polarity_reversal=polarity_reversal,
        **_sample_footprint(rng),
    )
    label = dict(
        is_dhi=True, kind=tier_name, tier=tier_name,
        il_center=site['inline'], xl_center=site['crossline'],
        thickness_m=thickness_m, reflection_coefficient=kwargs['reflection_coefficient'],
        flat_spot=flat_spot, polarity_reversal=polarity_reversal,
    )
    return kwargs, label


def sample_hard_negative_scenario(kind, rng, structural_highs, structural_lows,
                                   velocity_mps, freq_hz, flat_background_time_ms=None):
    """
    Sample a randomised hard-negative scenario: same amplitude/polarity
    machinery as a positive example, deliberately placed without genuine
    trap geometry.

    kind:
      'no_conformance'  - flat top time (ignores the horizon entirely),
                           same as the very first, geometrically-naive
                           version of this pipeline. Tests whether a
                           detector keys off amplitude alone.
      'syncline'        - sits on a real horizon (conformant shape) but at
                           a structural LOW rather than a high - physically
                           non-prospective despite looking geometrically
                           legitimate at a glance.
      'single_reflector'- a single isolated bright event (no base reflector)
                           on a real structural high - mimics a volcanic
                           flow top, coal bed, or unconformity rather than
                           a genuine reservoir wedge.

    Uses a random tier's amplitude/thickness regime so hard negatives span
    the same severity range as positives, rather than always being maximally
    obvious. Returns (injection_kwargs, label).
    """
    tier_name = rng.choice(list(TIER_RANGES.keys()))
    ranges = TIER_RANGES[tier_name]
    thickness_m = rng.uniform(*ranges['thickness_range'])
    rc_frac = rng.uniform(*ranges['rc_frac_range'])

    base_kwargs = dict(
        velocity_mps=velocity_mps, freq_hz=freq_hz,
        thickness_m=thickness_m, reflection_coefficient=RC_GAS_SAND * rc_frac,
        **_sample_footprint(rng),
    )
    label = dict(is_dhi=False, kind=f'hard_negative_{kind}', tier=tier_name,
                 thickness_m=thickness_m, reflection_coefficient=base_kwargs['reflection_coefficient'])

    if kind == 'no_conformance':
        if flat_background_time_ms is None:
            raise ValueError("kind='no_conformance' needs flat_background_time_ms")
        site = structural_highs.iloc[rng.integers(len(structural_highs))]  # location still needs an il/xl centre
        kwargs = dict(base_kwargs, il_center=site['inline'], xl_center=site['crossline'],
                      flat_top_time_ms=flat_background_time_ms)
        label.update(il_center=site['inline'], xl_center=site['crossline'])

    elif kind == 'syncline':
        site = structural_lows.iloc[rng.integers(len(structural_lows))]
        kwargs = dict(base_kwargs, il_center=site['inline'], xl_center=site['crossline'])
        label.update(il_center=site['inline'], xl_center=site['crossline'])

    elif kind == 'single_reflector':
        site = structural_highs.iloc[rng.integers(len(structural_highs))]
        kwargs = dict(base_kwargs, il_center=site['inline'], xl_center=site['crossline'],
                      single_reflector=True)
        label.update(il_center=site['inline'], xl_center=site['crossline'])

    else:
        raise ValueError(f'unknown hard-negative kind: {kind}')

    return kwargs, label
