# Implementation developed with AI (Claude Code) assistance - see AI_USAGE.md.
"""
Export injected scenarios as standalone SEG-Y files.

Unlike the .npz patch dataset (built for model training), this writes a real,
loadable SEG-Y sub-volume per scenario - a real chunk of the F3 survey with one
injected DHI or hard negative, openable in normal seismic interpretation
software (OpendTect, Petrel, ...) so the result can be visually judged, not just
read back as an array.

`export_blind_exchange_batch` builds on this for the actual blind-test hand-off
(C5 in the runbook): the .npz patch dataset can't be sent as-is, since each
patch is already cropped exactly onto the injection site and carries the
footprint mask - handing that over would give away the answer before the other
side's detector even runs. This exports real, unlabelled SEG-Y sub-volumes
instead, shuffled and renamed to anonymous IDs so neither the filenames nor
their order leak anything, with the real answer key kept in a separate,
private manifest that never leaves this side.
"""

import numpy as np
import pandas as pd
import segyio

from .dataset import _scenario_center_time
from .injection import estimate_amplitude_scale, inject_dhi_anomaly_3d


def export_scenario_to_segy(kwargs, label, f, iline_map, inlines, xlines, horizon,
                             output_path, il_extent=96, xl_extent=96, reference_rc=0.05):
    """
    Inject one scenario onto a real sub-volume (full trace length, not the
    500ms window used for ML patches - more context for visual inspection)
    and write it out as a standalone SEG-Y file, copying real trace headers
    (inline/crossline/CDP X/Y/...) from the source survey.

    Returns the output path, or None if the sub-volume would run off the
    survey's valid il/xl range (same edge behaviour as generate_example).
    """
    il_center, xl_center = kwargs['il_center'], kwargs['xl_center']
    center_time_ms = _scenario_center_time(kwargs, horizon)

    il_lo, il_hi = int(il_center - il_extent // 2), int(il_center + il_extent // 2)
    xl_lo, xl_hi = int(xl_center - xl_extent // 2), int(xl_center + xl_extent // 2)
    if il_lo < inlines[0] or il_hi > inlines[-1] or xl_lo < xlines[0] or xl_hi > xlines[-1]:
        return None

    patch_inline_axis = np.arange(il_lo, il_hi)
    patch_xl_axis = np.arange(xl_lo, xl_hi)
    n_samples = f.samples.size
    n_traces = len(patch_inline_axis) * len(patch_xl_axis)

    raw = np.full((len(patch_inline_axis), len(patch_xl_axis), n_samples), np.nan)
    trace_indices = np.full((len(patch_inline_axis), len(patch_xl_axis)), -1, dtype=int)
    for i, il in enumerate(patch_inline_axis):
        for j, xl in enumerate(patch_xl_axis):
            idx = iline_map.get((int(il), int(xl)))
            if idx is not None:
                raw[i, j] = f.trace[idx]
                trace_indices[i, j] = idx

    time_axis_ms = f.samples.astype(float)
    amp_scale = estimate_amplitude_scale(raw, reference_rc=reference_rc)
    injected, _ = inject_dhi_anomaly_3d(
        raw, time_axis_ms, patch_inline_axis, patch_xl_axis, horizon,
        amplitude_scale=amp_scale, **kwargs,
    )
    injected_filled = np.nan_to_num(injected, nan=0.0)

    spec = segyio.spec()
    spec.samples = f.samples
    spec.tracecount = n_traces
    # Schema v1.1: always write IEEE float (SEG-Y format 5), not the source
    # survey's format (int16, format 3). int16 quantises amplitude on export,
    # an uncontrolled difference between blind-exchange datasets since the
    # other side's own export is float - subtle-tier deltas and taper skirts
    # get rounded to integer counts on this side only. Free to remove since
    # amplitudes here are in the thousands (see estimate_amplitude_scale) -
    # agreed with Aziz, see colleague-notes-2026-08-09.md.
    spec.format = segyio.SegySampleFormat.IEEE_FLOAT_4_BYTE
    spec.sorting = segyio.TraceSortingFormat.CROSSLINE_SORTING

    with segyio.create(str(output_path), spec) as dst:
        # dst.bin = f.bin copies the *whole* source binary header wholesale,
        # including its Format field (int16, code 3) - silently reverting the
        # spec.format = IEEE_FLOAT_4_BYTE set above. Traces are still written
        # as float32 either way, so the header would then lie about the
        # on-disk format: segyio computes trace length from the declared
        # format's byte width, so a file with a float-sized trace but an
        # int16-declared header fails to reopen ("trace count inconsistent
        # with file size"). Re-assert the format field after the bulk copy.
        dst.bin = f.bin
        dst.bin[segyio.BinField.Format] = segyio.SegySampleFormat.IEEE_FLOAT_4_BYTE
        trace_i = 0
        for i, il in enumerate(patch_inline_axis):
            for j, xl in enumerate(patch_xl_axis):
                src_idx = trace_indices[i, j]
                if src_idx >= 0:
                    dst.header[trace_i] = f.header[src_idx]
                else:
                    # gap in the survey's acquisition outline - no real trace to copy headers
                    # from; still write correct inline/crossline so the geometry stays valid
                    dst.header[trace_i] = {
                        segyio.TraceField.INLINE_3D: int(il),
                        segyio.TraceField.CROSSLINE_3D: int(xl),
                    }
                dst.trace[trace_i] = injected_filled[i, j].astype(np.float32)
                trace_i += 1

    return output_path


def label_row_to_injection_kwargs(row, velocity_mps, freq_hz):
    """
    Reconstruct the kwargs export_scenario_to_segy/inject_dhi_anomaly_3d needs
    from one row of labels.parquet - the row itself only stores the resulting
    label fields, not the original call, so a couple of fields need inferring
    from `kind` rather than reading directly:

    - `single_reflector` isn't stored as its own column; it's implied by
      kind == 'hard_negative_single_reflector'.
    - `flat_top_time_ms` isn't stored either; for kind == 'hard_negative_
      no_conformance' it equals the stored center_time_ms exactly (see
      dataset.py::_scenario_center_time - that's *why* center_time_ms is
      already correct for those rows), otherwise it's None (follow the horizon).
    - `velocity_mps`/`freq_hz` are survey-level calibration constants (see
      C1), not per-example fields, so they're passed in rather than read from
      the row.
    - `horizon_time_offset_ms`/`apply_sag`/`v_gas_mps` are never overridden by
      either scenario sampler in this codebase, so their defaults are correct
      and don't need reconstructing here.
    """
    return dict(
        thickness_m=row['thickness_m'], velocity_mps=velocity_mps,
        reflection_coefficient=row['reflection_coefficient'], freq_hz=freq_hz,
        il_center=int(row['il_center']), xl_center=int(row['xl_center']),
        il_radius=row['il_radius'], xl_radius=row['xl_radius'], rotation_deg=row['rotation_deg'],
        flat_spot=bool(row['flat_spot']) if pd.notna(row['flat_spot']) else False,
        polarity_reversal=bool(row['polarity_reversal']) if pd.notna(row['polarity_reversal']) else False,
        single_reflector=(row['kind'] == 'hard_negative_single_reflector'),
        flat_top_time_ms=row['center_time_ms'] if row['kind'] == 'hard_negative_no_conformance' else None,
    )


def export_blind_exchange_batch(labels, f, iline_map, inlines, xlines, horizon,
                                 velocity_mps, freq_hz, output_dir, seed=0,
                                 il_extent=96, xl_extent=96, reference_rc=0.05):
    """
    Export every example in `labels` as an anonymised, unlabelled SEG-Y file
    for the C5 blind exchange - safe to hand to the other side, unlike the
    .npz patch dataset (pre-cropped onto the injection site, mask included).

    Shuffles the examples (seeded, so reproducible on this side only) and
    assigns sequential anonymous IDs (blind_0000, blind_0001, ...) instead of
    the real example_id - so neither the filename nor the hand-off order
    reveals anything about which examples are positives, which tier, etc.

    Writes the .sgy files to `output_dir` (send this to the other side - no
    other file from this directory) and returns a private manifest DataFrame
    (blind_id -> every real label field, including is_dhi/tier/kind/mask
    geometry) - keep this on this side only, it's the answer key needed to
    score the other side's guesses later. Does not write the manifest to disk
    itself - the caller decides where to save it privately.
    """
    shuffled = labels.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    shuffled['blind_id'] = [f'blind_{i:04d}' for i in range(len(shuffled))]

    manifest_rows = []
    for _, row in shuffled.iterrows():
        kwargs = label_row_to_injection_kwargs(row, velocity_mps, freq_hz)
        out_path = f"{output_dir}/{row['blind_id']}.sgy"
        result = export_scenario_to_segy(kwargs, None, f, iline_map, inlines, xlines, horizon,
                                          out_path, il_extent=il_extent, xl_extent=xl_extent,
                                          reference_rc=reference_rc)
        manifest_rows.append(dict(row, exported=result is not None))

    manifest = pd.DataFrame(manifest_rows)
    n_failed = (~manifest['exported']).sum()
    if n_failed:
        import warnings
        warnings.warn(f'{n_failed}/{len(manifest)} examples could not be exported '
                       f'(ran off the survey edge) - excluded from the exchange batch.')
    return manifest
