# Implementation developed with AI (Claude Code) assistance - see AI_USAGE.md.
"""F2 regression: polarity_reversal must not null the anomaly's centre, must give both
legs (gas/brine) a real, non-zero amplitude, and must genuinely flip sign across the real
fluid contact - not the footprint's geometric centre. See
colleague-pipeline-review-2026-07-28.md F2 / fixes-2026-07-29.md F2.
"""
import numpy as np

from src.dhi_pipeline.injection import inject_dhi_anomaly_3d, RC_WATER_SAND
from tests.conftest import DippingHorizon


def test_footprint_centre_trace_is_not_nulled(small_grid, base_injection_kwargs):
    """The old bug flipped RC sign linearly across the footprint's geometric centre,
    which was exactly zero there. Build a case where the real contact (derived from
    structural relief) lands at that same geometric centre by construction - a linear
    dip symmetric about il_center puts the contact there with the default contact_frac=0.5
    - and confirm the centre trace is NOT a null."""
    volume, time_axis_ms, inline_axis, xl_axis = small_grid
    horizon = DippingHorizon(base_time_ms=900.0, dip_ms_per_trace=1.0, dip_axis='il')

    out, _ = inject_dhi_anomaly_3d(
        volume, time_axis_ms, inline_axis, xl_axis, horizon,
        flat_spot=False, polarity_reversal=True, **base_injection_kwargs,
    )

    il_center, xl_center = base_injection_kwargs['il_center'], base_injection_kwargs['xl_center']
    centre_peak = np.max(np.abs(out[il_center, xl_center, :]))
    assert centre_peak > 1e-6, 'footprint centre trace should not be a null'


def test_both_legs_have_real_amplitude(small_grid, base_injection_kwargs):
    """Neither the gas leg (updip) nor the brine leg (downdip) should be near-zero -
    a real fluid contact has genuine, non-zero reflectivity on both sides."""
    volume, time_axis_ms, inline_axis, xl_axis = small_grid
    horizon = DippingHorizon(base_time_ms=900.0, dip_ms_per_trace=1.0, dip_axis='il')

    out, _ = inject_dhi_anomaly_3d(
        volume, time_axis_ms, inline_axis, xl_axis, horizon,
        flat_spot=False, polarity_reversal=True, **base_injection_kwargs,
    )

    il_center = base_injection_kwargs['il_center']
    xl_center = base_injection_kwargs['xl_center']
    updip_peak = np.max(np.abs(out[il_center - 8, xl_center, :]))
    downdip_peak = np.max(np.abs(out[il_center + 8, xl_center, :]))

    # both legs should be a real fraction of the full (non-reversed) gas-leg amplitude,
    # not vanishingly small - loose bound, just ruling out an accidental near-null
    reference_rc = abs(base_injection_kwargs['reflection_coefficient'])
    min_expected = 0.1 * reference_rc * base_injection_kwargs['amplitude_scale']
    assert updip_peak > min_expected, f'gas leg looks nulled: {updip_peak}'
    assert downdip_peak > min_expected, f'brine leg looks nulled: {downdip_peak}'


def test_sign_flips_across_the_contact(small_grid, base_injection_kwargs):
    """The reflection coefficient must genuinely flip sign between the gas leg (updip)
    and the brine leg (downdip). Uses a much thicker bed than the other tests purely so
    the top and base reflectors are cleanly separated in time - otherwise wavelet
    interference between them can muddy a sign reading taken directly off the trace
    (the review hit exactly this and isolated the RC formula directly instead)."""
    volume, time_axis_ms, inline_axis, xl_axis = small_grid
    horizon = DippingHorizon(base_time_ms=900.0, dip_ms_per_trace=1.0, dip_axis='il')

    kwargs = dict(base_injection_kwargs, thickness_m=50.0)
    out, _ = inject_dhi_anomaly_3d(
        volume, time_axis_ms, inline_axis, xl_axis, horizon,
        flat_spot=False, polarity_reversal=True, **kwargs,
    )

    il_center, xl_center = kwargs['il_center'], kwargs['xl_center']
    updip_il, downdip_il = il_center - 8, il_center + 8
    top_time_updip = horizon.time_at(updip_il, xl_center)
    top_time_downdip = horizon.time_at(downdip_il, xl_center)

    # read the sign right at each trace's own top-reflector arrival, not just the peak -
    # with top/base well separated this is unambiguously the top event's polarity
    updip_sign = np.sign(out[updip_il, xl_center, np.argmin(np.abs(time_axis_ms - top_time_updip))])
    downdip_sign = np.sign(out[downdip_il, xl_center, np.argmin(np.abs(time_axis_ms - top_time_downdip))])

    assert updip_sign != downdip_sign, 'RC should flip sign across the contact'
    # gas leg (updip) should carry the (negative) reflection_coefficient's sign,
    # brine leg (downdip) should carry RC_WATER_SAND's (positive) sign
    assert updip_sign == np.sign(kwargs['reflection_coefficient'])
    assert downdip_sign == np.sign(RC_WATER_SAND)
