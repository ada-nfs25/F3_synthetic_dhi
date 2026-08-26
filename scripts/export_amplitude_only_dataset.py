#!/usr/bin/env python3
# Implementation developed with AI (Claude Code) assistance - see AI_USAGE.md.
"""
Export the combined P0 dataset (H1 + H3 + dim_spot + aziz_style, 1208
examples) as amplitude-only patches - raw seismic amplitude + ground truth
(mask/mask_3d), dropping the 7 derived attribute channels.

Our .npz files normally store the full 8-channel attribute stack
(amplitude, envelope, inst_phase, inst_freq, sweetness, band_ratio, rms,
local_var) - already-computed derived channels, not raw data. That's fine
for training directly here, but 11.1x larger than amplitude alone (measured
directly: 31.7MB -> 2.9MB per patch, ~36GB -> ~3.2GB for the full combined
set) - worth avoiding whenever the full stack isn't actually needed:
sharing the dataset externally, or
packaging a smaller deliverable footprint generally.

mask/mask_3d (ground truth, not a computed feature) are always kept.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

DATASETS = {
    'H1': 'F3_synthetic_dhi_dataset_p0_radius6_15',
    'H3': 'F3_synthetic_dhi_dataset_p0_h3_shallow',
    'dim_spot': 'F3_synthetic_dhi_dataset_p0_dim_spot',
    'aziz_style': 'F3_synthetic_dhi_dataset_p0_aziz_style',
}
LABELS_PATH = REPO / 'data' / 'v2_features' / 'v2_combined_features.parquet'
OUT_DIR = REPO / 'data' / 'F3_synthetic_dhi_dataset_p0_combined_amplitude_only'


def main():
    patches_out = OUT_DIR / 'patches'
    patches_out.mkdir(parents=True, exist_ok=True)

    labels = pd.read_parquet(LABELS_PATH)

    written = 0
    for _, row in labels.iterrows():
        src_dir = DATASETS[row['dataset_source']]
        src_path = REPO / 'data' / src_dir / 'patches' / row['patch_file']
        data = np.load(src_path)
        channel_names = list(data['channel_names'])
        amplitude = data['attribute_stack'][channel_names.index('amplitude')].astype(np.float32)

        out_name = f"{row['global_id'].replace(':', '_')}.npz"
        np.savez_compressed(patches_out / out_name, amplitude=amplitude,
                             mask=data['mask'], mask_3d=data['mask_3d'])
        written += 1
        if written % 200 == 0:
            print(f'  {written}/{len(labels)} exported')

    labels_out = labels.copy()
    labels_out['amplitude_only_npz'] = labels['global_id'].apply(lambda g: f"{g.replace(':', '_')}.npz")
    labels_out.to_parquet(OUT_DIR / 'labels.parquet', index=False)
    labels_out.to_csv(OUT_DIR / 'labels.csv', index=False)

    print(f'\n{written} amplitude-only patches written to {patches_out}')
    total_bytes = sum(f.stat().st_size for f in patches_out.glob('*.npz'))
    print(f'total size: {total_bytes / 1e9:.2f}GB')


if __name__ == '__main__':
    main()
