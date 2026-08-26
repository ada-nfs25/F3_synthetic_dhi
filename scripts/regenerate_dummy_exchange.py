#!/usr/bin/env python3
# Implementation developed with AI (Claude Code) assistance - see AI_USAGE.md.
"""Regenerate the dummy C5 blind-exchange file + detection record.

The original blind_0000.sgy/blind_0000_detection.json predate two later
fixes and were never refreshed: the schema v1.1 IEEE-float export (still
int16) and the axis_convention literal (had "map-view", schema requires
exactly "map_view"). This regenerates both from the current P0 dataset and
current segy_export.py so the dummy artifacts actually match what the code
and SCHEMA.md now specify, before the real exchange.
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import segyio

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.dhi_pipeline.horizons import HorizonSurface, build_coordinate_lookup, load_horizon_surface  # noqa: E402
from src.dhi_pipeline.segy_export import export_scenario_to_segy, label_row_to_injection_kwargs  # noqa: E402


def main():
    # calibration constants (dt/velocity/dominant frequency) are already pinned
    # for this dataset in its own manifest - reuse them rather than re-deriving
    # from the raw SEG-Y, so this dummy file matches the real P0 dataset exactly
    dataset_dir = REPO / 'data/F3_synthetic_dhi_dataset_p0_radius6_15'
    manifest = json.loads((dataset_dir / 'generation_manifest.json').read_text())
    dt_ms, velocity_mps, freq_hz = manifest['dt_ms'], manifest['velocity_mps'], manifest['dominant_frequency_hz']

    # any tier3_at_tuning row will do - this file only exists to prove the
    # exchange *format* round-trips before the real exchange, per the "DUMMY
    # detection" note baked into the JSON below, so the specific example
    # doesn't need to match the original commit's choice
    labels = pd.read_csv(dataset_dir / 'labels.csv')
    row = labels[labels['tier'] == 'tier3_at_tuning'].iloc[0]

    raw = REPO / 'data_raw'
    segy_path = raw / 'Seismic_data.sgy'
    with (REPO / 'data/f3_trace_index.pkl').open('rb') as fh:
        index = pickle.load(fh)
    iline_map, inlines, xlines = index['iline_map'], index['inlines'], index['xlines']

    coords = build_coordinate_lookup(str(segy_path))
    surface = load_horizon_surface(str(raw / 'horizons/H1.xyz'), coords['ilxl_array'], coords['xy_array'])
    horizon = HorizonSurface(surface)

    output_dir = REPO / 'data/dummy_exchange'
    output_path = output_dir / 'blind_0000.sgy'

    # reconstructs the original inject_dhi_anomaly_3d() call from the label
    # row alone, same as a real blind-exchange export would - label rows don't
    # store the original kwargs directly, see label_row_to_injection_kwargs's
    # own docstring for which fields have to be inferred from `kind`
    kwargs = label_row_to_injection_kwargs(row, velocity_mps, freq_hz)
    with segyio.open(str(segy_path), ignore_geometry=True) as f:
        result = export_scenario_to_segy(
            kwargs, None, f, iline_map, inlines, xlines, horizon, output_path,
        )
    if result is None:
        raise RuntimeError('chosen example ran off the survey edge, pick a different row')

    # il_extent/xl_extent/il_lo/xl_lo mirror export_scenario_to_segy's own
    # patch geometry (see that function) - recomputed here rather than
    # returned by it, since its return value is just the output path
    il_extent = xl_extent = 96
    il_lo = int(row['il_center']) - il_extent // 2
    xl_lo = int(row['xl_center']) - xl_extent // 2

    # reopen the file just written to read back its actual sample count/start
    # time, rather than assuming them, so patch_dimensions below can't drift
    # out of sync with what export_scenario_to_segy actually wrote
    with segyio.open(str(output_path), ignore_geometry=True) as f:
        n_samples = f.samples.size
        patch_origin_time_ms = float(f.samples[0])

    # illustrative detector output in the agreed schema v1.1 shape - see
    # data/dummy_exchange/SCHEMA.md for field definitions. is_dhi/predicted_tier/
    # confidence/predicted_time_ms/localisation_mask are placeholder values (this
    # is a format check, not a real detector run); patch_origin/patch_dimensions
    # describe the actual file just written, and axis_convention must match
    # SCHEMA.md's literal exactly ("map_view" with an underscore, not a hyphen -
    # the bug being fixed here)
    detection = {
        'schema_version': '1.1',
        'note': 'DUMMY detection - values are illustrative placeholders for format-testing (C5 protocol), not a real model output',
        'detector_side': 'nora',
        'blind_id': 'blind_0000',
        'patch_origin': {'il': il_lo, 'xl': xl_lo, 'time_ms': patch_origin_time_ms},
        'patch_dimensions': {'n_il': il_extent, 'n_xl': xl_extent, 'n_samples': n_samples, 'dt_ms': dt_ms},
        'is_dhi': True,
        'predicted_tier': 'tier3_at_tuning',
        'confidence': 0.82,
        'predicted_time_ms': float(row['center_time_ms']) - patch_origin_time_ms,
        'localisation_mask': {
            'axis_convention': 'map_view (il, xl), indices local to this patch (0,0) = (il_origin, xl_origin)',
            'shape': [il_extent, xl_extent],
            'mask': np.zeros((il_extent, xl_extent), dtype=int).tolist(),
        },
    }
    (output_dir / 'blind_0000_detection.json').write_text(json.dumps(detection, indent=2) + '\n')

    with segyio.open(str(output_path), ignore_geometry=True) as f:
        print('format:', f.format)
    print('axis_convention:', detection['localisation_mask']['axis_convention'])
    print('wrote', output_path)
    print('wrote', output_dir / 'blind_0000_detection.json')


if __name__ == '__main__':
    main()
