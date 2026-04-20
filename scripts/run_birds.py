"""Run the BirdNET bird analysis stage from an existing inventory."""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from audio_ecology.analysis.birdnet import (
    get_birdnet_output_dir,
    run_birdnet_analysis,
)
from audio_ecology.config import load_config
from audio_ecology.logging_config import configure_pipeline_logging
from audio_ecology.profiling import ProfileRecorder

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = SCRIPT_DIR / 'config_files' / 'config.yaml'


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Run BirdNET bird analysis from an existing inventory.'
    )
    parser.add_argument(
        'config_path',
        nargs='?',
        default=DEFAULT_CONFIG_PATH,
        type=Path,
        help='Path to the pipeline config YAML.',
    )
    parser.add_argument(
        '--inventory-stem',
        default='audio_inventory',
        help='Inventory file stem in the configured output directory.',
    )
    parser.add_argument(
        '--log-level',
        default='INFO',
        help='Logging level: INFO or DEBUG.',
    )
    parser.add_argument(
        '--overwrite-checkpoints',
        action='store_true',
        help='Re-run files even when BirdNET checkpoints already exist.',
    )
    parser.add_argument(
        '--no-profile',
        action='store_true',
        help='Disable profiling report output.',
    )
    return parser.parse_args()


def main() -> None:
    """Run BirdNET bird analysis from an existing inventory."""
    args = parse_args()
    config = load_config(args.config_path.resolve())
    log_file_path = configure_pipeline_logging(
        config=config,
        level=args.log_level,
        run_name='birds',
    )
    profiler = ProfileRecorder(
        output_dir=config.output_dir,
        run_name='birdnet_existing_inventory',
        enabled=not args.no_profile,
    )

    inventory_path = config.output_dir / f'{args.inventory_stem}.parquet'
    if not inventory_path.exists():
        raise FileNotFoundError(
            f'Inventory parquet not found: {inventory_path}. '
            'Run scripts/run_inventory.py first.'
        )

    with profiler.profile('read_inventory'):
        inventory_df = pl.read_parquet(inventory_path)
    with profiler.profile('birdnet_analysis'):
        detections_df = run_birdnet_analysis(
            config=config,
            inventory_df=inventory_df,
            overwrite_checkpoints=args.overwrite_checkpoints,
        )
    profile_paths = profiler.write()

    print(
        f'Wrote {detections_df.height} BirdNET detections to '
        f'{get_birdnet_output_dir(config)}'
    )
    if profile_paths is not None:
        print(f'Wrote profile reports to {profile_paths[0].parent}')
    if log_file_path is not None:
        print(f'Wrote log to {log_file_path}')


if __name__ == '__main__':
    main()
