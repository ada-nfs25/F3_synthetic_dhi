#!/usr/bin/env python3
# Implementation developed with AI (Claude Code) assistance - see AI_USAGE.md.
"""Build the regeneratable F3 SEG-Y trace-index cache."""

import argparse
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from utils.seismic_io import load_or_build_trace_index  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--segy-path',
        type=Path,
        default=REPO / 'data_raw' / 'Seismic_data.sgy',
        help='input F3 SEG-Y file',
    )
    parser.add_argument(
        '--output-path',
        type=Path,
        default=REPO / 'data' / 'f3_trace_index.pkl',
        help='output pickle cache',
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='rebuild the cache even if it already exists',
    )
    args = parser.parse_args()

    existed_before = args.output_path.is_file()

    index = load_or_build_trace_index(
        segy_path=args.segy_path,
        cache_path=args.output_path,
        force_rebuild=args.force,
    )

    action = (
        'rebuilt'
        if args.force
        else 'loaded'
        if existed_before
        else 'built'
    )

    print(f'Trace index {action}: {args.output_path}')
    print(f'Unique trace positions: {len(index["iline_map"])}')
    print(
        f'Inlines: {index["inlines"][0]} to '
        f'{index["inlines"][-1]} '
        f'({len(index["inlines"])} unique)'
    )
    print(
        f'Crosslines: {index["xlines"][0]} to '
        f'{index["xlines"][-1]} '
        f'({len(index["xlines"])} unique)'
    )


if __name__ == '__main__':
    main()
