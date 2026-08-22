# Implementation developed with AI (Claude Code) assistance - see AI_USAGE.md.
"""
Dim-spot positive examples: genuine host-reflectivity attenuation, not an
additive wedge event. Round 2, P0 item 3 (Aziz, ROUND2_PLAN.md): "brightness-
correlated features ... need counter-examples where the anomaly is an
amplitude decrease."

Our own injection.py has no such primitive - every anomaly there is additive
(a wedge, a flat spot, a single reflector). Rather than reimplement the
attenuation physics, this borrows Aziz's dim_spot mechanic
(external/aziz_dhi_lib/primitives.py::build_anomaly, kind='dim_spot': scales
existing host reflectivity down by a tier-dependent factor) with OUR OWN
calibration (RC constants, measured tuning thickness, Ricker wavelet) - the
same split validated in the in-house replication
(scripts/inhouse_replication_thick_dim.py).

Kept as a SEPARATE module rather than folded into dataset.py's
generate_example, so the existing, already-validated positive/hard-negative
path is untouched - this is purely additive.

Important: aziz_dhi_lib's build_anomaly indexes its `volume` argument's time
axis from sample 0 = t_s 0.0 (it's designed for a full, zero-origin survey
volume). Our patches are cropped to a local time window, so horizon times
must be rebased to the patch's own origin or every event lands outside the
patch's sample range and silently no-ops - see local_horizon_grid_s below.
This is the same bug the in-house replication script hit and fixed.
"""
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / 'external') not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / 'external'))
from aziz_dhi_lib.primitives import build_anomaly  # noqa: E402

from .attributes import compute_attribute_stack  # noqa: E402
from .injection import ricker_wavelet  # noqa: E402

MASK_RELATIVE_THRESHOLD = 0.05
"""Same relative-threshold convention aziz_dhi_lib.inject.apply_catalog uses
for its own ground-truth masks (`strong = |delta| > 0.05 * |delta|.max()`) -
reused here rather than invented fresh, since dim_spot's delta is an
attenuation (host amplitude shrinks toward zero) rather than an additive
event, so the "replay injection against a zero background" trick
dataset.py's own masks use doesn't apply: attenuating zero leaves zero."""


def local_horizon_grid_s(horizon, inline_axis, xl_axis, patch_time_origin_ms):
    grid_ms = np.array([[horizon.time_at(int(il), int(xl)) for xl in xl_axis] for il in inline_axis])
    return (grid_ms - patch_time_origin_ms) / 1000.0


def generate_dim_spot_example(tier, il_center, xl_center, cached_subvol, cache_inline_axis,
                               cache_xl_axis, samples_ms, horizon, dt_ms, freq_hz, rc_gas, rc_brine,
                               tuning_thickness_s, v_gas_mps, velocity_mps, rng,
                               il_extent=96, xl_extent=96, time_extent_ms=500, size_ilxl=(20, 20)):
    """Build one dim_spot example - mirrors dataset.generate_example's
    contract (same return shape, same skip-reason convention) so it can be
    written into the same patches/ + labels table layout, but for a
    fundamentally different (attenuation, not additive) anomaly mechanism.

    Returns (None, skip_reason) or (result_dict, None), same as
    dataset.generate_example.
    """
    center_time_ms = horizon.time_at(il_center, xl_center)
    il_lo, il_hi = int(il_center - il_extent // 2), int(il_center + il_extent // 2)
    xl_lo, xl_hi = int(xl_center - xl_extent // 2), int(xl_center + xl_extent // 2)
    if il_lo < cache_inline_axis[0] or il_hi > cache_inline_axis[-1] or \
       xl_lo < cache_xl_axis[0] or xl_hi > cache_xl_axis[-1]:
        return None, 'il_xl_bounds'

    t_lo_ms, t_hi_ms = center_time_ms - time_extent_ms / 2, center_time_ms + time_extent_ms / 2
    if t_lo_ms < samples_ms[0] or t_hi_ms > samples_ms[-1]:
        return None, 'time_bounds'

    patch_inline_axis = np.arange(il_lo, il_hi)
    patch_xl_axis = np.arange(xl_lo, xl_hi)
    il_off = il_lo - cache_inline_axis[0]
    xl_off = xl_lo - cache_xl_axis[0]
    raw = cached_subvol[il_off:il_off + len(patch_inline_axis), xl_off:xl_off + len(patch_xl_axis), :]

    t_mask = (samples_ms >= t_lo_ms) & (samples_ms <= t_hi_ms)
    patch_time_axis_ms = samples_ms[t_mask]
    raw_patch = np.nan_to_num(raw[:, :, t_mask], nan=0.0).astype(np.float32)

    dt_s = dt_ms / 1000.0
    wavelet = ricker_wavelet(freq_hz, dt_s, length_ms=120)
    horizon_grid_s = local_horizon_grid_s(horizon, patch_inline_axis, patch_xl_axis, patch_time_axis_ms[0])
    local_site = (raw_patch.shape[0] // 2, raw_patch.shape[1] // 2)

    delta = build_anomaly(
        kind='dim_spot', tier=tier, volume=raw_patch, dt_s=dt_s,
        horizon_twt_s=horizon_grid_s, site=local_site, size_ilxl=size_ilxl,
        rc_gas=rc_gas, rc_brine=rc_brine, thickness_s=tuning_thickness_s,
        wavelet=wavelet, rng=rng, tuning_thickness_s=tuning_thickness_s,
        apply_sag=True, v_gas_mps=v_gas_mps, v_background_mps=velocity_mps,
    )
    injected = raw_patch + delta

    attribute_stack, channel_names = compute_attribute_stack(injected, dt_s)
    attribute_stack = attribute_stack.astype(np.float32)

    strong = np.abs(delta) > MASK_RELATIVE_THRESHOLD * np.abs(delta).max()
    mask_3d = strong
    mask = strong.any(axis=2)
    if not mask.any():
        return None, 'dim_spot_no_mask_voxels'

    peak_amplitude = float(np.nanmax(np.abs(delta)))
    label = dict(
        is_dhi=True, kind='dim_spot', tier=f'dim_spot_tier{tier}',
        il_center=il_center, xl_center=xl_center, thickness_m=None,
        reflection_coefficient=None, flat_spot=False, polarity_reversal=False,
        il_lo=il_lo, il_hi=il_hi, xl_lo=xl_lo, xl_hi=xl_hi,
        center_time_ms=center_time_ms, twt_thickness_ms=None,
        n_missing_traces=int(np.isnan(raw).sum() // samples_ms.size),
        peak_amplitude=peak_amplitude, il_radius=size_ilxl[0] / 2, xl_radius=size_ilxl[1] / 2,
        rotation_deg=0.0, sag_shift_ms=0.0,
    )
    return dict(attribute_stack=attribute_stack, channel_names=channel_names, mask=mask,
                mask_3d=mask_3d, label=label), None
