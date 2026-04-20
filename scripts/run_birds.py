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
from audio_ecology.logging_config import configure_logging

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
    return parser.parse_args()


def main() -> None:
    """Run BirdNET bird analysis from an existing inventory."""
    args = parse_args()
    configure_logging(args.log_level)
    config = load_config(args.config_path.resolve())

    inventory_path = config.output_dir / f'{args.inventory_stem}.parquet'
    if not inventory_path.exists():
        raise FileNotFoundError(
            f'Inventory parquet not found: {inventory_path}. '
            'Run scripts/run_inventory.py first.'
        )

    inventory_df = pl.read_parquet(inventory_path)
    detections_df = run_birdnet_analysis(
        config=config,
        inventory_df=inventory_df,
    )

    print(
        f'Wrote {detections_df.height} BirdNET detections to '
        f'{get_birdnet_output_dir(config)}'
    )


if __name__ == '__main__':
    main()
