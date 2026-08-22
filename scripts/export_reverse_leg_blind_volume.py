#!/usr/bin/env python3
"""Export the blind volume for the reverse leg (Aziz's ask #3, ROUND2_PLAN.md's
"what to send back" list) - Aziz runs his detector on this, no answers revealed
on this side, doesn't wait for v2.

~40 examples, test-split only (held out from our own model too, so this stays
comparable to our own model's held-out recall on the same examples later).
Composition matches round 1's exact counts (round1_evaluation.json), using
our own tier1-4 system in place of Aziz's bright_spot/dim_spot/flat_spot_only
split (which we have no equivalent for): 4 per severity tier (16 positives),
1 per hard-negative kind (4), 20 backgrounds = 40.

Writes two SEPARATE things:
  - data/round2_reverse_leg/blind_for_aziz/  - the anonymised .sgy files +
    index.json (patch_id -> center_time_ms, mirroring the index.json Aziz sent
    us in round 1). SEND THIS, nothing else from this directory.
  - data/round2_reverse_leg/private_answer_key.csv - the real labels behind
    each blind_id. KEEP THIS PRIVATE - it's what lets us score his detections
    later, and sending it would defeat the blind test.
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
from src.dhi_pipeline.segy_export import export_blind_exchange_batch  # noqa: E402

SEED = 20260822
N_PER_TIER = 4
N_PER_HARD_NEGATIVE = 1
N_BACKGROUND = 20
TIERS = ['tier1_subtle', 'tier2_approaching', 'tier3_at_tuning', 'tier4_obvious']
HARD_NEGATIVES = ['hard_negative_no_conformance', 'hard_negative_syncline',
                   'hard_negative_single_reflector', 'hard_negative_tuning']


def select_examples(labels, rng):
    test = labels[labels.split == 'test']
    parts = []
    for kind in TIERS:
        pool = test[test.kind == kind]
        parts.append(pool.sample(n=min(N_PER_TIER, len(pool)), random_state=rng.integers(0, 2**31)))
    for kind in HARD_NEGATIVES:
        pool = test[test.kind == kind]
        n = min(N_PER_HARD_NEGATIVE, len(pool))
        parts.append(pool.sample(n=n, random_state=rng.integers(0, 2**31)))
        if n < N_PER_HARD_NEGATIVE:
            print(f'  note: only {n}/{N_PER_HARD_NEGATIVE} available for {kind} in test split')
    pool = test[test.kind == 'background']
    parts.append(pool.sample(n=min(N_BACKGROUND, len(pool)), random_state=rng.integers(0, 2**31)))
    return pd.concat(parts, ignore_index=True)


def main():
    raw_dir = REPO / 'data_raw'
    segy_path = raw_dir / 'Seismic_data.sgy'
    dataset_dir = REPO / 'data/F3_synthetic_dhi_dataset_p0_radius6_15'
    manifest = json.loads((dataset_dir / 'generation_manifest.json').read_text())
    dt_ms, velocity_mps, freq_hz = manifest['dt_ms'], manifest['velocity_mps'], manifest['dominant_frequency_hz']

    labels = pd.read_parquet(dataset_dir / 'labels.parquet')
    rng = np.random.default_rng(SEED)
    selected = select_examples(labels, rng)
    print(f'selected {len(selected)} examples from the test split:')
    print(selected.groupby('kind').size())

    with (REPO / 'data/f3_trace_index.pkl').open('rb') as fh:
        index = pickle.load(fh)
    iline_map, inlines, xlines = index['iline_map'], index['inlines'], index['xlines']

    coords = build_coordinate_lookup(str(segy_path))
    surface = load_horizon_surface(str(raw_dir / 'horizons/H1.xyz'), coords['ilxl_array'], coords['xy_array'])
    horizon = HorizonSurface(surface)

    output_dir = REPO / 'data/round2_reverse_leg/blind_for_aziz'
    output_dir.mkdir(parents=True, exist_ok=True)
    private_dir = REPO / 'data/round2_reverse_leg'

    with segyio.open(str(segy_path), ignore_geometry=True) as f:
        manifest_df = export_blind_exchange_batch(
            selected, f, iline_map, inlines, xlines, horizon,
            velocity_mps, freq_hz, str(output_dir), seed=SEED,
        )

    exported = manifest_df[manifest_df.exported]
    n_failed = len(manifest_df) - len(exported)
    if n_failed:
        print(f'{n_failed} example(s) ran off the survey edge and were not exported')

    index_doc = {
        'patches': [
            {'patch_id': row.blind_id, 'center_time_ms': float(row.center_time_ms)}
            for row in exported.itertuples()
        ]
    }
    (output_dir / 'index.json').write_text(json.dumps(index_doc, indent=2) + '\n')

    exported.to_csv(private_dir / 'private_answer_key.csv', index=False)

    print(f'\n{len(exported)} patches exported to {output_dir}')
    print(f'wrote {output_dir / "index.json"}')
    print(f'wrote PRIVATE answer key to {private_dir / "private_answer_key.csv"} - do not send this')
    print(f'\nSend: everything in {output_dir} (the .sgy files + index.json), nothing else from {private_dir}')


if __name__ == '__main__':
    main()
