"""Run the inventory stage only."""

from __future__ import annotations

import argparse
from pathlib import Path

from audio_ecology.config import load_config
from audio_ecology.ingest.inventory import (
    build_inventory_records,
    records_to_polars,
    write_inventory_outputs,
)
from audio_ecology.logging_config import configure_logging

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = SCRIPT_DIR / 'config_files' / 'config.yaml'


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Build the audio inventory only.'
    )
    parser.add_argument(
        'config_path',
        nargs='?',
        default=DEFAULT_CONFIG_PATH,
        type=Path,
        help='Path to the pipeline config YAML.',
    )
    parser.add_argument(
        '--stem',
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
    """Run the inventory stage only."""
    args = parse_args()
    configure_logging(args.log_level)
    config = load_config(args.config_path.resolve())

    records = build_inventory_records(config)
    inventory_df = records_to_polars(records)
    parquet_path, csv_path = write_inventory_outputs(
        inventory_df=inventory_df,
        output_dir=config.output_dir,
        stem=args.stem,
        write_csv=config.outputs.write_csv,
    )

    message = f'Wrote inventory with {inventory_df.height} files to {parquet_path}'
    if csv_path is not None:
        message = f'{message} and {csv_path}'
    print(message)


if __name__ == '__main__':
    main()
