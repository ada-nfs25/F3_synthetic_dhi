"""
Synthetic DHI (bright-spot) injection.

Models a reservoir as a two-reflector wedge (top + base) convolved with a
zero-phase Ricker wavelet - the standard wedge-model approach for
reproducing tuning behaviour (e.g. Kallweit & Wood 1982).

The reflectivity series is built on a fine time grid before convolution,
then resampled down to the trace's own sample interval. Building directly
on the coarse trace grid would make beds thinner than one sample interval
(true for tier 1 here - see synthetic_dhi_generation.ipynb) land their
top/base spikes on the same sample and cancel to zero instead of tuning.
"""

import numpy as np


def ricker_wavelet(freq_hz, dt_s, length_ms=120):
    """Zero-phase Ricker wavelet at a given peak/dominant frequency."""
    half_len_s = (length_ms / 2) / 1000.0
    t = np.arange(-half_len_s, half_len_s + dt_s, dt_s)
    arg = (np.pi * freq_hz * t) ** 2
    return (1 - 2 * arg) * np.exp(-arg)


def thickness_to_twt_ms(thickness_m, velocity_mps):
    """True (one-way) thickness -> two-way travel-time thickness, in ms."""
    return 2000 * thickness_m / velocity_mps


def _reflectivity_series(time_axis_ms, spikes):
    """spikes: list of (time_ms, amplitude). Nearest-sample placement."""
    r = np.zeros_like(time_axis_ms, dtype=float)
    for t_ms, amp in spikes:
        idx = np.argmin(np.abs(time_axis_ms - t_ms))
        r[idx] += amp
    return r


def _fine_wedge(top_time_ms, thickness_m, velocity_mps, reflection_coefficient, freq_hz,
                 flat_spot_offset_ms=None, flat_spot_rc=None, include_base=True,
                 dt_fine_ms=0.1, pad_ms=150):
    """Reflectivity + wavelet convolution on a fine time grid (shared core)."""
    twt_thickness_ms = thickness_to_twt_ms(thickness_m, velocity_mps)
    base_time_ms = top_time_ms + twt_thickness_ms

    fine_t = np.arange(top_time_ms - pad_ms, top_time_ms + twt_thickness_ms + pad_ms, dt_fine_ms)
    spikes = [(top_time_ms, reflection_coefficient)]
    if include_base:
        spikes.append((base_time_ms, -reflection_coefficient))
    if flat_spot_offset_ms is not None:
        spikes.append((base_time_ms + flat_spot_offset_ms, flat_spot_rc))
    r_fine = _reflectivity_series(fine_t, spikes)

    wavelet_fine = ricker_wavelet(freq_hz, dt_fine_ms / 1000.0, length_ms=120)
    wedge_fine = np.convolve(r_fine, wavelet_fine, mode='same')

    return wedge_fine, fine_t, twt_thickness_ms


def wedge_peak_amplitude(thickness_m, velocity_mps, reflection_coefficient, freq_hz,
                          top_time_ms=700, flat_spot_offset_ms=None, flat_spot_rc=None,
                          dt_fine_ms=0.1, pad_ms=150):
    """
    Peak |amplitude| of the wedge response, measured on the fine time grid -
    i.e. the true continuous-time interference amplitude, not the amplitude
    read off a discretely-sampled trace.

    This is the right quantity for validating tuning behaviour (e.g. "does
    peak amplitude occur near the calibrated tuning thickness?"). Measuring
    on a resampled coarse trace grid instead mixes in sample-phase-dependent
    amplitude loss - the coarse grid's fixed 4ms samples don't generally land
    exactly on the wedge's true peak, and that gap grows/shrinks continuously
    as the sweep varies thickness, which distorts the tuning curve's shape
    and shifts its apparent peak location.
    """
    wedge_fine, _, _ = _fine_wedge(top_time_ms, thickness_m, velocity_mps, reflection_coefficient,
                                    freq_hz, flat_spot_offset_ms, flat_spot_rc,
                                    dt_fine_ms=dt_fine_ms, pad_ms=pad_ms)
    return np.max(np.abs(wedge_fine))


def model_wedge_response(top_time_ms, thickness_m, velocity_mps, reflection_coefficient,
                          freq_hz, dt_ms, flat_spot_offset_ms=None, flat_spot_rc=None,
                          include_base=True, dt_fine_ms=0.1, pad_ms=150):
    """
    Band-limited seismic response of a reservoir wedge, resampled onto the
    trace's own sample interval (dt_ms) - use this for actually injecting
    onto a real trace. For validating tuning behaviour in the abstract, use
    `wedge_peak_amplitude` instead (see its docstring for why).

    include_base=False drops the base-of-reservoir reflector, leaving a
    single isolated event - mimics a single-interface look-alike (volcanic
    flow top, coal bed, unconformity) rather than a genuine reservoir wedge.

    Returns (wedge_trace, time_axis_ms for that trace, twt_thickness_ms).
    """
    wedge_fine, fine_t, twt_thickness_ms = _fine_wedge(
        top_time_ms, thickness_m, velocity_mps, reflection_coefficient, freq_hz,
        flat_spot_offset_ms, flat_spot_rc, include_base, dt_fine_ms, pad_ms,
    )
    coarse_t = np.arange(top_time_ms - pad_ms, top_time_ms + twt_thickness_ms + pad_ms, dt_ms)
    wedge_coarse = np.interp(coarse_t, fine_t, wedge_fine)

    return wedge_coarse, coarse_t, twt_thickness_ms


def estimate_amplitude_scale(background_patch, reference_rc=0.05):
    """
    Convert a dimensionless reflection coefficient into this dataset's raw
    trace-amplitude units (counts per unit RC).

    Real SEG-Y amplitude isn't physically calibrated to reflection
    coefficient - it carries whatever arbitrary processing/gain was applied
    upstream (F3's raw amplitudes run into the tens of thousands, RC is
    bounded in [-1, 1]). Without that processing history, there's no exact
    conversion; this estimates one by treating the RMS amplitude of a
    representative background patch (a mix of many ordinary, unremarkable
    stratal reflections, not one strong isolated event) as corresponding to
    a "typical" reflection coefficient (`reference_rc`, default 0.05 - a
    plausible value for everyday shale/sand contrasts, well below a
    hydrocarbon-sand contrast like RC_GAS_SAND ~ -0.18). This is an
    approximation, not a measurement - flag it as such wherever it's used.

    Uses nanmean: sub-volumes drawn from arbitrary survey locations can
    contain missing traces (F3's acquisition outline isn't a perfect
    rectangle), and a plain mean would let a handful of NaN traces poison
    the entire calibration.
    """
    rms_amplitude = np.sqrt(np.nanmean(background_patch ** 2))
    return rms_amplitude / reference_rc


def inject_dhi_anomaly(patch, time_axis_ms, xl_axis, top_time_ms, thickness_m, velocity_mps,
                        reflection_coefficient, freq_hz, xl_extent=None, taper_traces=5,
                        flat_spot=False, flat_spot_offset_ms=15, polarity_reversal=False,
                        amplitude_scale=1.0):
    """
    Add a synthetic reservoir-wedge response onto a background patch.

    patch: (n_traces, n_samples) raw amplitude - copied, not mutated.
    reflection_coefficient: signed RC at the reservoir top; negative = bright/
        gas-sand per Nanda (2021), positive = weak water-sand case.
    xl_extent: (xl_lo, xl_hi) lateral footprint of the reservoir; defaults to
        the full width of `patch`.
    polarity_reversal: linearly flips RC sign across xl_extent (updip gas ->
        downdip water), producing the polarity-reversal signature from Nanda.
    flat_spot: adds a positive-polarity flat reflector `flat_spot_offset_ms`
        below the reservoir base, representing the fluid contact.
    taper_traces: number of traces at each edge of xl_extent to cosine-taper,
        so the injected patch doesn't have a hard-edged lateral boundary.
    amplitude_scale: counts-per-unit-RC conversion factor (see
        `estimate_amplitude_scale`) - default 1.0 leaves the wedge in raw RC
        units, which is invisible next to real trace amplitudes; pass a
        dataset-calibrated scale to actually inject onto real data.
    """
    out = patch.copy()
    dt_ms = time_axis_ms[1] - time_axis_ms[0]
    xl_lo, xl_hi = xl_extent if xl_extent is not None else (xl_axis[0], xl_axis[-1])
    footprint = np.where((xl_axis >= xl_lo) & (xl_axis <= xl_hi))[0]

    twt_thickness_ms = thickness_to_twt_ms(thickness_m, velocity_mps)

    for i, xl_idx in enumerate(footprint):
        rc = reflection_coefficient
        if polarity_reversal:
            frac = i / max(len(footprint) - 1, 1)          # 0 at xl_lo -> 1 at xl_hi
            rc = reflection_coefficient * (1 - 2 * frac)     # sign flips halfway across

        wedge, wedge_t, _ = model_wedge_response(
            top_time_ms, thickness_m, velocity_mps, rc, freq_hz, dt_ms,
            flat_spot_offset_ms=flat_spot_offset_ms if flat_spot else None,
            flat_spot_rc=abs(reflection_coefficient) if flat_spot else None,
        )

        weight = 1.0
        if taper_traces > 0:
            edge_dist = min(i, len(footprint) - 1 - i)
            if edge_dist < taper_traces:
                weight = 0.5 * (1 - np.cos(np.pi * (edge_dist + 1) / (taper_traces + 1)))

        in_range = (time_axis_ms >= wedge_t[0]) & (time_axis_ms <= wedge_t[-1])
        out[xl_idx, in_range] += weight * amplitude_scale * np.interp(time_axis_ms[in_range], wedge_t, wedge)

    return out, twt_thickness_ms


def inject_dhi_anomaly_3d(volume, time_axis_ms, inline_axis, xl_axis, horizon_surface,
                           thickness_m, velocity_mps, reflection_coefficient, freq_hz,
                           il_center, xl_center, il_radius, xl_radius, rotation_deg=0.0,
                           amplitude_scale=1.0, edge_taper_frac=0.2, flat_spot=False,
                           flat_spot_offset_ms=15, polarity_reversal=False,
                           horizon_time_offset_ms=0.0, single_reflector=False,
                           flat_top_time_ms=None):
    """
    Add a synthetic reservoir-wedge response onto a 3D background volume,
    following a real horizon surface for structural conformance rather than
    sitting at a constant, artificial top time (see `inject_dhi_anomaly` for
    the flat single-inline version this generalises).

    volume: (n_inlines, n_xlines, n_samples) raw amplitude - copied, not mutated.
    horizon_surface: a `HorizonSurface` (src/dhi_pipeline/horizons.py) giving
        real top_time_ms at any (il, xl) - the reservoir top follows its shape.
    il_center/xl_center/il_radius/xl_radius: footprint is an ellipse in
        (inline, crossline) space, centred at (il_center, xl_center).
    rotation_deg: rotates the ellipse's axes relative to the inline/crossline
        grid (0 = il_radius along inline, xl_radius along crossline).
    edge_taper_frac: fraction of the footprint's outer radius (by normalised
        elliptical distance) that cosine-tapers to zero, so the footprint
        doesn't have a hard-edged boundary.
    polarity_reversal: linearly flips RC sign across the footprint's local
        (rotated) crossline-like axis (updip gas -> downdip water), as in
        Nanda (2021).
    horizon_time_offset_ms: constant shift applied to the horizon's time at
        every point in the footprint - lets a scenario sit structurally
        *below* a horizon (e.g. a syncline hard negative, see scenarios.py)
        while still following its real shape. 0 = sit directly on the horizon.
    single_reflector: drop the base-of-reservoir event, leaving one isolated
        reflector - a non-conformant look-alike (volcanic/coal/unconformity)
        rather than a genuine reservoir wedge.
    flat_top_time_ms: if given, overrides the horizon entirely and uses this
        constant time everywhere in the footprint - the "no structural
        conformance at all" hard-negative case.
    (other params as `inject_dhi_anomaly`)

    Returns (out, twt_thickness_ms).
    """
    out = volume.copy()
    dt_ms = time_axis_ms[1] - time_axis_ms[0]
    twt_thickness_ms = thickness_to_twt_ms(thickness_m, velocity_mps)
    theta = np.radians(rotation_deg)

    for i, il in enumerate(inline_axis):
        for j, xl in enumerate(xl_axis):
            d_il, d_xl = il - il_center, xl - xl_center
            il_rot = d_il * np.cos(theta) + d_xl * np.sin(theta)
            xl_rot = -d_il * np.sin(theta) + d_xl * np.cos(theta)
            r = np.sqrt((il_rot / il_radius) ** 2 + (xl_rot / xl_radius) ** 2)
            if r > 1.0:
                continue

            rc = reflection_coefficient
            if polarity_reversal:
                frac = (xl_rot + xl_radius) / (2 * xl_radius)
                rc = reflection_coefficient * (1 - 2 * np.clip(frac, 0, 1))

            if flat_top_time_ms is not None:
                top_time_ms = flat_top_time_ms
            else:
                top_time_ms = horizon_surface.time_at(il, xl) + horizon_time_offset_ms

            wedge, wedge_t, _ = model_wedge_response(
                top_time_ms, thickness_m, velocity_mps, rc, freq_hz, dt_ms,
                flat_spot_offset_ms=flat_spot_offset_ms if flat_spot else None,
                flat_spot_rc=abs(reflection_coefficient) if flat_spot else None,
                include_base=not single_reflector,
            )

            weight = 1.0
            if r > 1 - edge_taper_frac:
                weight = 0.5 * (1 + np.cos(np.pi * (r - (1 - edge_taper_frac)) / edge_taper_frac))

            in_range = (time_axis_ms >= wedge_t[0]) & (time_axis_ms <= wedge_t[-1])
            out[i, j, in_range] += weight * amplitude_scale * np.interp(time_axis_ms[in_range], wedge_t, wedge)

    return out, twt_thickness_ms


# --- Petrophysics -> reflection coefficient (Nanda 2021, see notebook research cell) ---
# Water sand: Vp=2300 m/s, rho=2.2 g/cc -> "impedance" 2300*2.2=5060
# Gas sand:   Vp=1600 m/s, rho=2.1 g/cc -> "impedance" 1600*2.1=3360
# (units cancel in the RC ratio, so V*rho in any consistent units is fine)
# Overlying shale impedance isn't given in Nanda's qualitative example, so
# 4800 is *assumed* here, chosen to reproduce the described pattern: weak
# positive contrast for the brine sand, strong negative for the gas sand.
Z_SHALE_ASSUMED = 4800
Z_WATER_SAND = 2300 * 2.2
Z_GAS_SAND = 1600 * 2.1

RC_WATER_SAND = (Z_WATER_SAND - Z_SHALE_ASSUMED) / (Z_WATER_SAND + Z_SHALE_ASSUMED)
RC_GAS_SAND = (Z_GAS_SAND - Z_SHALE_ASSUMED) / (Z_GAS_SAND + Z_SHALE_ASSUMED)

# Severity tiers: thickness relative to tuning thickness is the primary axis;
# reflection_coefficient is scaled as a fraction of the full gas-sand
# contrast, treated as a saturation/net-to-gross proxy - full contrast is
# reserved for the "textbook" tier 4 case.
#
# Thicknesses are anchored on the *empirically measured* tuning thickness of
# this wedge model (~7.07m via wedge_peak_amplitude sweep in the notebook),
# not the analytic quarter-wavelength estimate V/(4f) (~9.07m for this
# survey). The two differ by ~22% because a Ricker wavelet's spectral-peak
# frequency (what both the data calibration and this wavelet are built from)
# doesn't correspond to the "frequency" the quarter-wavelength rule of thumb
# implicitly assumes - a known property of Ricker wavelets specifically
# (Kallweit & Wood 1982). The empirical value is what actually governs
# interference in this injection pipeline, so tiers are built on that.
SEVERITY_TIERS = {
    'tier1_subtle': dict(
        thickness_m=2.0,
        reflection_coefficient=RC_GAS_SAND * 0.40,
        flat_spot=False,
        polarity_reversal=False,
    ),
    'tier2_approaching': dict(
        thickness_m=5.0,
        reflection_coefficient=RC_GAS_SAND * 0.65,
        flat_spot=False,
        polarity_reversal=False,
    ),
    'tier3_at_tuning': dict(
        thickness_m=7.0,
        reflection_coefficient=RC_GAS_SAND * 0.85,
        flat_spot=False,
        polarity_reversal=False,
    ),
    'tier4_obvious': dict(
        thickness_m=14.0,
        reflection_coefficient=RC_GAS_SAND * 1.00,
        flat_spot=True,
        polarity_reversal=True,
    ),
}
