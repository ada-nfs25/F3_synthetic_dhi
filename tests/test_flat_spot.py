# Implementation developed with AI (Claude Code) assistance - see AI_USAGE.md.
"""F1 regression: the flat spot must be flat (an absolute contact time, not an offset
below the dipping top), present only updip of the contact, and refuse to place an
unresolvable one. See colleague-pipeline-review-2026-07-28.md F1 / fixes-2026-07-29.md F1.
"""
import numpy as np
import pytest

from src.dhi_pipeline.injection import inject_dhi_anomaly_3d, model_wedge_response
from tests.conftest import DippingHorizon, FlatHorizon


def _inject(volume, time_axis_ms, inline_axis, xl_axis, horizon, kwargs, flat_spot):
    out, _ = inject_dhi_anomaly_3d(
        volume, time_axis_ms, inline_axis, xl_axis, horizon,
        flat_spot=flat_spot, polarity_reversal=False, **kwargs,
    )
    return out


def test_flat_spot_event_time_constant_across_dipping_traces(small_grid, base_injection_kwargs):
    """On a dipping horizon, the flat-spot event must land at the same absolute time on
    every trace - that discordance with the dipping top is the entire diagnostic value of
    a flat spot. Isolate the event by differencing against a flat_spot=False twin (same
    review methodology), on several traces near the footprint's shallow (updip) end."""
    volume, time_axis_ms, inline_axis, xl_axis = small_grid
    horizon = DippingHorizon(base_time_ms=900.0, dip_ms_per_trace=1.0, dip_axis='il')

    with_flat = _inject(volume, time_axis_ms, inline_axis, xl_axis, horizon, base_injection_kwargs, True)
    without_flat = _inject(volume, time_axis_ms, inline_axis, xl_axis, horizon, base_injection_kwargs, False)
    diff = with_flat - without_flat

    il_center, xl_center = base_injection_kwargs['il_center'], base_injection_kwargs['xl_center']
    il_radius = base_injection_kwargs['il_radius']
    updip_ils = [il_center - il_radius + 2, il_center - il_radius + 5, il_center - il_radius + 8]

    event_times_ms = []
    for il in updip_ils:
        trace_diff = diff[il, xl_center, :]
        assert np.max(np.abs(trace_diff)) > 1e-8, f'expected a flat-spot event on trace il={il}'
        event_times_ms.append(time_axis_ms[np.argmax(np.abs(trace_diff))])

    assert event_times_ms[0] == event_times_ms[1] == event_times_ms[2], (
        f'flat-spot event time should be identical across traces, got {event_times_ms}'
    )


def test_flat_spot_absent_downdip_present_updip(small_grid, base_injection_kwargs):
    """Downdip of the contact there is no gas, so no flat-spot event at all; updip there
    must be one."""
    volume, time_axis_ms, inline_axis, xl_axis = small_grid
    horizon = DippingHorizon(base_time_ms=900.0, dip_ms_per_trace=1.0, dip_axis='il')

    with_flat = _inject(volume, time_axis_ms, inline_axis, xl_axis, horizon, base_injection_kwargs, True)
    without_flat = _inject(volume, time_axis_ms, inline_axis, xl_axis, horizon, base_injection_kwargs, False)
    diff = with_flat - without_flat

    il_center, xl_center = base_injection_kwargs['il_center'], base_injection_kwargs['xl_center']
    il_radius = base_injection_kwargs['il_radius']
    updip_il = il_center - il_radius + 2       # near the crest (shallowest) - clearly updip
    downdip_il = il_center + il_radius - 2     # near the deepest point - clearly downdip

    assert np.max(np.abs(diff[updip_il, xl_center, :])) > 1e-8, 'expected a flat-spot event updip'
    assert np.max(np.abs(diff[downdip_il, xl_center, :])) < 1e-8, 'expected no flat-spot event downdip'


def test_flat_spot_false_matches_plain_two_reflector_wedge(small_grid, base_injection_kwargs):
    """flat_spot=False must not leave any trace of a third event, even on a trace that
    would be updip of the contact if flat_spot were True - compare directly against a
    plain top+base wedge with no contact concept at all. Picks a trace well inside the
    footprint (not near the edge) so the edge taper doesn't need to be modelled here too -
    that's a separate mechanism, covered by its own test elsewhere."""
    volume, time_axis_ms, inline_axis, xl_axis = small_grid
    horizon = DippingHorizon(base_time_ms=900.0, dip_ms_per_trace=1.0, dip_axis='il')

    without_flat = _inject(volume, time_axis_ms, inline_axis, xl_axis, horizon, base_injection_kwargs, False)

    il_center, xl_center = base_injection_kwargs['il_center'], base_injection_kwargs['xl_center']
    updip_il = il_center - 5  # updip (shallower) but well inside the footprint, no edge taper
    top_time_ms = horizon.time_at(updip_il, xl_center)
    dt_ms = time_axis_ms[1] - time_axis_ms[0]

    wedge, wedge_t, _ = model_wedge_response(
        top_time_ms, base_injection_kwargs['thickness_m'], base_injection_kwargs['velocity_mps'],
        base_injection_kwargs['reflection_coefficient'], base_injection_kwargs['freq_hz'], dt_ms,
        contact_time_ms=None, flat_spot_rc=None,
    )
    expected = np.zeros_like(time_axis_ms)
    in_range = (time_axis_ms >= wedge_t[0]) & (time_axis_ms <= wedge_t[-1])
    expected[in_range] = np.interp(time_axis_ms[in_range], wedge_t, wedge)

    np.testing.assert_allclose(without_flat[updip_il, xl_center, :], expected, atol=1e-8)


def test_insufficient_relief_raises(small_grid, base_injection_kwargs):
    """A footprint sitting on a flat horizon has zero structural relief - nowhere near
    enough to resolve a contact as a separable event. Must fail loudly, not silently
    emit a meaningless flat spot."""
    volume, time_axis_ms, inline_axis, xl_axis = small_grid
    horizon = FlatHorizon(time_ms=900.0)

    with pytest.raises(ValueError, match='resolvable threshold'):
        inject_dhi_anomaly_3d(
            volume, time_axis_ms, inline_axis, xl_axis, horizon,
            flat_spot=True, polarity_reversal=False, **base_injection_kwargs,
        )
