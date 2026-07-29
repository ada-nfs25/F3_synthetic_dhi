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

`inject_dhi_anomaly_3d` also models the 'sag'/pull-down effect (Nanda 2021):
the velocity drop within the gas-filled reservoir delays reflectors beneath
it, via `sag_time_shift_ms`. Disable with apply_sag=False if not wanted.
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


def sag_time_shift_ms(thickness_m, v_gas_mps, v_background_mps):
    """
    Extra two-way travel-time delay on reflectors below a gas-filled
    reservoir, from the velocity drop within its thickness - Nanda (2021)'s
    'sag'/pull-down effect. v_gas_mps < v_background_mps gives a positive
    shift (later arrival, i.e. a downward 'sag' in time).
    """
    return 2000 * thickness_m * (1.0 / v_gas_mps - 1.0 / v_background_mps)


def _reflectivity_series(time_axis_ms, spikes):
    """spikes: list of (time_ms, amplitude). Nearest-sample placement."""
    r = np.zeros_like(time_axis_ms, dtype=float)
    for t_ms, amp in spikes:
        idx = np.argmin(np.abs(time_axis_ms - t_ms))
        r[idx] += amp
    return r


def _fine_wedge(top_time_ms, thickness_m, velocity_mps, reflection_coefficient, freq_hz,
                 contact_time_ms=None, flat_spot_rc=None, include_base=True,
                 dt_fine_ms=0.1, pad_ms=150):
    """Reflectivity + wavelet convolution on a fine time grid (shared core).

    contact_time_ms: absolute two-way time of the fluid contact (flat spot) -
    the same value regardless of top_time_ms, which is what makes it "flat"
    against a dipping top (replaces the old flat_spot_offset_ms, which was
    measured from base_time_ms and so dipped with the structure - see F1).
    Only emitted when top_time_ms < contact_time_ms (updip / gas leg);
    downdip of the contact there is no gas, so no flat-spot event at all.
    """
    twt_thickness_ms = thickness_to_twt_ms(thickness_m, velocity_mps)
    base_time_ms = top_time_ms + twt_thickness_ms

    # extend the fine grid to cover the contact too, if it sits deeper than
    # the reservoir base + pad - otherwise a contact far below the base would
    # fall outside fine_t and its spike would silently vanish
    fine_t_hi = top_time_ms + twt_thickness_ms + pad_ms
    if contact_time_ms is not None:
        fine_t_hi = max(fine_t_hi, contact_time_ms + pad_ms)
    fine_t = np.arange(top_time_ms - pad_ms, fine_t_hi, dt_fine_ms)

    spikes = [(top_time_ms, reflection_coefficient)]
    if include_base:
        spikes.append((base_time_ms, -reflection_coefficient))
    if contact_time_ms is not None and top_time_ms < contact_time_ms:
        spikes.append((contact_time_ms, flat_spot_rc))
    r_fine = _reflectivity_series(fine_t, spikes)

    wavelet_fine = ricker_wavelet(freq_hz, dt_fine_ms / 1000.0, length_ms=120)
    wedge_fine = np.convolve(r_fine, wavelet_fine, mode='same')

    return wedge_fine, fine_t, twt_thickness_ms


def wedge_peak_amplitude(thickness_m, velocity_mps, reflection_coefficient, freq_hz,
                          top_time_ms=700, contact_time_ms=None, flat_spot_rc=None,
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
                                    freq_hz, contact_time_ms, flat_spot_rc,
                                    dt_fine_ms=dt_fine_ms, pad_ms=pad_ms)
    return np.max(np.abs(wedge_fine))


def model_wedge_response(top_time_ms, thickness_m, velocity_mps, reflection_coefficient,
                          freq_hz, dt_ms, contact_time_ms=None, flat_spot_rc=None,
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
        contact_time_ms, flat_spot_rc, include_base, dt_fine_ms, pad_ms,
    )
    # kept in sync with _fine_wedge's fine_t extension above - otherwise the
    # flat-spot event could exist in wedge_fine but fall outside coarse_t and
    # never actually get injected into the trace
    coarse_t_hi = top_time_ms + twt_thickness_ms + pad_ms
    if contact_time_ms is not None:
        coarse_t_hi = max(coarse_t_hi, contact_time_ms + pad_ms)
    coarse_t = np.arange(top_time_ms - pad_ms, coarse_t_hi, dt_ms)
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
                        flat_spot=False, contact_time_ms=None, polarity_reversal=False,
                        water_leg_scale=1.0, polarity_blend_frac=0.03, amplitude_scale=1.0):
    """
    Add a synthetic reservoir-wedge response onto a background patch.

    patch: (n_traces, n_samples) raw amplitude - copied, not mutated.
    reflection_coefficient: signed RC at the reservoir top; negative = bright/
        gas-sand per Nanda (2021), positive = weak water-sand case.
    xl_extent: (xl_lo, xl_hi) lateral footprint of the reservoir; defaults to
        the full width of `patch`.
    polarity_reversal: smoothly blends RC from `reflection_coefficient` (gas
        leg) to `RC_WATER_SAND * water_leg_scale` (brine leg, weaker and
        opposite-signed - never zero) across the footprint's lateral centre.
        Unlike the 3D version there's no real horizon here (top_time_ms is a
        single constant), so the transition is anchored to the footprint's
        geometric centre in lieu of a real fluid-contact depth - see F2.
        `polarity_blend_frac` sets the transition width as a fraction of the
        footprint's trace count (default 0.03, i.e. a handful of traces
        either side of centre - "~2-3 traces" per the review, not a hard step).
    flat_spot: adds a positive-polarity flat reflector at the absolute time
        `contact_time_ms` (required if flat_spot=True) - the fluid contact.
        Unlike the 3D version, this function has no horizon/structural-relief
        concept (top_time_ms is a single constant for the whole patch), so
        the contact time must be supplied directly rather than derived.
    taper_traces: number of traces at each edge of xl_extent to cosine-taper,
        so the injected patch doesn't have a hard-edged lateral boundary.
    amplitude_scale: counts-per-unit-RC conversion factor (see
        `estimate_amplitude_scale`) - default 1.0 leaves the wedge in raw RC
        units, which is invisible next to real trace amplitudes; pass a
        dataset-calibrated scale to actually inject onto real data.
    """
    if flat_spot and contact_time_ms is None:
        raise ValueError('flat_spot=True requires contact_time_ms to be set')

    out = patch.copy()
    dt_ms = time_axis_ms[1] - time_axis_ms[0]
    xl_lo, xl_hi = xl_extent if xl_extent is not None else (xl_axis[0], xl_axis[-1])
    footprint = np.where((xl_axis >= xl_lo) & (xl_axis <= xl_hi))[0]

    twt_thickness_ms = thickness_to_twt_ms(thickness_m, velocity_mps)
    blend_width = max(polarity_blend_frac, 1e-6)

    for i, xl_idx in enumerate(footprint):
        rc = reflection_coefficient
        if polarity_reversal:
            # F2: same non-zero-leg + smooth-blend fix as the 3D version, but
            # anchored to lateral position (frac) since there's no real depth
            # here - previously rc crossed exactly zero at frac=0.5 (footprint
            # centre), same bug as the 3D version just in a different axis.
            frac = i / max(len(footprint) - 1, 1)          # 0 at xl_lo -> 1 at xl_hi
            blend_frac = 0.5 * (1 + np.tanh((frac - 0.5) / blend_width))
            rc = reflection_coefficient * (1 - blend_frac) + (RC_WATER_SAND * water_leg_scale) * blend_frac

        wedge, wedge_t, _ = model_wedge_response(
            top_time_ms, thickness_m, velocity_mps, rc, freq_hz, dt_ms,
            contact_time_ms=contact_time_ms if flat_spot else None,
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
                           contact_frac=0.5, polarity_reversal=False, water_leg_scale=1.0,
                           horizon_time_offset_ms=0.0, single_reflector=False,
                           flat_top_time_ms=None, apply_sag=True, v_gas_mps=None):
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
    polarity_reversal: drives RC from the same `contact_time_ms` as flat_spot
        (F2) - bright gas-sand RC updip of the contact, a real (weaker,
        opposite-signed) brine-sand RC downdip, blended smoothly across the
        contact rather than a hard step. Previously this flipped sign
        linearly across the footprint's geometric centre regardless of where
        the real contact was, which zeroed out exactly at the centre (where
        the edge taper was also strongest) - see the review, F2.
    flat_spot: adds a fluid-contact reflector at a single absolute time
        (`contact_time_ms`, derived below from the footprint's own structural
        relief), present only updip of the contact - see F1. Requires a real
        horizon (flat_top_time_ms must be None), since the contact is placed
        relative to the footprint's structural relief. Note flat_spot and
        polarity_reversal both key off the same contact_time_ms - enabling
        either one triggers the relief derivation below, since a polarity
        reversal is physically the same fluid contact as the flat spot,
        just expressed as an amplitude/sign change rather than an extra event.
    contact_frac: where the contact sits within the footprint's relief, as a
        fraction from the crest (0 = at the crest itself, 1 = at the deepest
        point in the footprint). Default 0.5 puts it roughly halfway down,
        so part of the footprint is updip (gas-filled) and part downdip
        (water leg).
    water_leg_scale: scales RC_WATER_SAND (the real, measured brine-sand
        reflection coefficient - see the constants below) for the downdip
        leg when polarity_reversal=True. Default 1.0 uses the full measured
        contrast; RC_WATER_SAND itself is not tier-scaled the way the gas
        side is, since a brine leg doesn't have a "severity".
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
    apply_sag: model the 'sag'/pull-down effect (Nanda 2021) - the velocity
        drop within a gas-filled reservoir delays reflectors beneath it.
        `velocity_mps` is treated as the real background velocity at that
        location (it's normally interpolated from the real interval-velocity
        log, see synthetic_dhi_generation.ipynb); `v_gas_mps` is the
        anomalously slower velocity inside the gas zone itself. Shift is
        tapered by the same footprint edge weight as the amplitude, so it
        goes smoothly to zero rather than cutting off abruptly at the
        footprint boundary.
    v_gas_mps: gas-sand velocity used for the sag calculation above; defaults
        to V_GAS_SAND (Nanda 2021's illustrative Vp=1600m/s, the same value
        `Z_GAS_SAND` below is built from).
    (other params as `inject_dhi_anomaly`)

    Returns (out, twt_thickness_ms).
    """
    if v_gas_mps is None:
        v_gas_mps = V_GAS_SAND

    out = volume.copy()
    dt_ms = time_axis_ms[1] - time_axis_ms[0]
    twt_thickness_ms = thickness_to_twt_ms(thickness_m, velocity_mps)
    sag_shift_ms = sag_time_shift_ms(thickness_m, v_gas_mps, velocity_mps) if apply_sag else 0.0
    theta = np.radians(rotation_deg)

    # F1/F2: derive the contact from the footprint's own structural relief,
    # once, before the per-voxel loop. Both flat_spot (an extra reflector at
    # the contact) and polarity_reversal (a sign change across it) are the
    # same physical fluid contact, so either one needs contact_time_ms - a
    # real absolute time has to come from somewhere, not an arbitrary offset
    # below the (dipping) top or a flip at the footprint's geometric centre.
    contact_time_ms = None
    polarity_blend_ms = None
    if flat_spot or polarity_reversal:
        if flat_top_time_ms is not None:
            raise ValueError('flat_spot/polarity_reversal need a real horizon '
                              '(flat_top_time_ms must be None) - there is no structural relief '
                              'to place a contact against when the top time is a flat '
                              'artificial constant')
        footprint_top_times = []
        for i, il in enumerate(inline_axis):
            for j, xl in enumerate(xl_axis):
                d_il, d_xl = il - il_center, xl - xl_center
                il_rot = d_il * np.cos(theta) + d_xl * np.sin(theta)
                xl_rot = -d_il * np.sin(theta) + d_xl * np.cos(theta)
                r = np.sqrt((il_rot / il_radius) ** 2 + (xl_rot / xl_radius) ** 2)
                if r <= 1.0:
                    footprint_top_times.append(horizon_surface.time_at(il, xl) + horizon_time_offset_ms)

        crest_time_ms = min(footprint_top_times)
        relief_ms = max(footprint_top_times) - crest_time_ms
        min_resolvable_relief_ms = 500.0 / freq_hz  # half a Ricker cycle - below this the
        # contact sits inside the same wavelet as the reservoir top and isn't a separable event
        if relief_ms < min_resolvable_relief_ms:
            raise ValueError(
                f'flat_spot/polarity_reversal but footprint relief ({relief_ms:.1f}ms) is below '
                f'the resolvable threshold ({min_resolvable_relief_ms:.1f}ms at {freq_hz}Hz) - '
                'the contact would sit inside the same wavelet as the reservoir top and produce '
                'no observable flat spot or polarity change. Use a larger/differently-placed '
                'footprint, a lower freq_hz, or disable both for this scenario.'
            )
        contact_time_ms = crest_time_ms + contact_frac * relief_ms

        # F2: width (in ms) of the smooth transition across the contact, for
        # polarity_reversal - "~2-3 traces" translated into time via the
        # footprint's own average dip rate (total relief over roughly its
        # diameter along the dip direction), so the blend scales with how
        # steeply this particular footprint dips rather than a fixed ms value.
        dip_rate_ms_per_trace = relief_ms / (2 * max(il_radius, xl_radius))
        polarity_blend_ms = 2.5 * dip_rate_ms_per_trace

    for i, il in enumerate(inline_axis):
        for j, xl in enumerate(xl_axis):
            d_il, d_xl = il - il_center, xl - xl_center
            il_rot = d_il * np.cos(theta) + d_xl * np.sin(theta)
            xl_rot = -d_il * np.sin(theta) + d_xl * np.cos(theta)
            r = np.sqrt((il_rot / il_radius) ** 2 + (xl_rot / xl_radius) ** 2)
            if r > 1.0:
                continue

            if flat_top_time_ms is not None:
                top_time_ms = flat_top_time_ms
            else:
                top_time_ms = horizon_surface.time_at(il, xl) + horizon_time_offset_ms

            rc = reflection_coefficient
            if polarity_reversal:
                # F2: keyed off the real contact (structural depth), not lateral
                # position in the footprint - a smooth tanh blend across
                # delta_t_ms=0 so neither leg is ever exactly zero and there's
                # no hard step, unlike the old geometric-centre zero-crossing.
                delta_t_ms = top_time_ms - contact_time_ms
                blend_frac = 0.5 * (1 + np.tanh(delta_t_ms / polarity_blend_ms))
                rc = reflection_coefficient * (1 - blend_frac) + (RC_WATER_SAND * water_leg_scale) * blend_frac

            wedge, wedge_t, _ = model_wedge_response(
                top_time_ms, thickness_m, velocity_mps, rc, freq_hz, dt_ms,
                contact_time_ms=contact_time_ms if flat_spot else None,
                flat_spot_rc=abs(reflection_coefficient) if flat_spot else None,
                include_base=not single_reflector,
            )

            weight = 1.0
            if r > 1 - edge_taper_frac:
                weight = 0.5 * (1 + np.cos(np.pi * (r - (1 - edge_taper_frac)) / edge_taper_frac))

            in_range = (time_axis_ms >= wedge_t[0]) & (time_axis_ms <= wedge_t[-1])
            out[i, j, in_range] += weight * amplitude_scale * np.interp(time_axis_ms[in_range], wedge_t, wedge)

            if apply_sag and sag_shift_ms != 0:
                tapered_shift_ms = weight * sag_shift_ms
                below_mask = time_axis_ms > wedge_t[-1]
                if np.any(below_mask):
                    out[i, j, below_mask] = np.interp(
                        time_axis_ms[below_mask] - tapered_shift_ms, time_axis_ms, out[i, j, :]
                    )

    return out, twt_thickness_ms


# --- Petrophysics -> reflection coefficient ---
# Z_SHALE_REAL / Z_WATER_SAND_REAL are measured, not assumed: pooled from real density
# (RHOB), sonic (DT) and gamma-ray (GR) logs across all 4 F3 wells (F02-1, F03-2, F03-4,
# F06-1) - see the "Checking these numbers against real F3 well logs" cell in
# synthetic_dhi_generation.ipynb for the derivation (including two real bugs caught along
# the way: coal contamination in a naive GR-only sand cutoff, and a units mismatch between
# LAS sources). Sand facies = low GR *and* plausible clastic density (excludes coal);
# shale = high GR. Values are the mean impedance (Vp[m/s] * RHOB[g/cc]) within each facies,
# pooled across all 4 wells.
Z_SHALE_REAL = 4588
Z_WATER_SAND_REAL = 5599
# No real gas-sand penetration exists in these wells (F3 Demo has no commercial discovery),
# so gas sand keeps Nanda (2021)'s illustrative Vp=1600m/s, rho=2.1g/cc -> "impedance" 3360
# (units cancel in the RC ratio, so V*rho in any consistent units is fine) - paired below
# with the real shale impedance rather than an assumed one. V_GAS_SAND is also
# sag_time_shift_ms()'s default gas velocity above - the same illustrative velocity drop
# that produces the bright-spot impedance contrast is what produces the pull-down delay
# beneath it.
V_GAS_SAND = 1600  # m/s
Z_GAS_SAND = V_GAS_SAND * 2.1

RC_WATER_SAND = (Z_WATER_SAND_REAL - Z_SHALE_REAL) / (Z_WATER_SAND_REAL + Z_SHALE_REAL)
RC_GAS_SAND = (Z_GAS_SAND - Z_SHALE_REAL) / (Z_GAS_SAND + Z_SHALE_REAL)

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
