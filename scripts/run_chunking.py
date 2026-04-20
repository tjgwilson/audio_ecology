"""Run the chunking stage from an existing inventory."""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from audio_ecology.config import load_config
from audio_ecology.ingest.chunking import (
    build_chunk_records,
    write_chunk_wavs,
)
from audio_ecology.ingest.inventory import (
    chunk_records_to_polars,
    write_chunk_inventory_outputs,
)
from audio_ecology.logging_config import configure_logging
from audio_ecology.models import AudioFileRecord

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = SCRIPT_DIR / 'config_files' / 'config.yaml'


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Build chunk records from an existing inventory.'
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
        '--chunk-stem',
        default='audio_chunks',
        help='Chunk inventory file stem in the configured output directory.',
    )
    parser.add_argument(
        '--log-level',
        default='INFO',
        help='Logging level: INFO or DEBUG.',
    )
    return parser.parse_args()


def records_from_inventory_df(inventory_df: pl.DataFrame) -> list[AudioFileRecord]:
    """Convert inventory rows back into AudioFileRecord objects."""
    return [
        AudioFileRecord.model_validate(row)
        for row in inventory_df.iter_rows(named=True)
    ]


def main() -> None:
    """Run the chunking stage from an existing inventory."""
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
    records = records_from_inventory_df(inventory_df)
    chunking_config = config.chunking.model_copy(update={'enabled': True})
    chunk_records = build_chunk_records(
        records=records,
        chunking_config=chunking_config,
        analysis_targets=chunking_config.analysis_targets,
    )
    chunk_records = write_chunk_wavs(
        chunk_records=chunk_records,
        chunking_config=chunking_config,
        default_output_dir=config.output_dir,
    )
    chunk_df = chunk_records_to_polars(chunk_records)
    parquet_path, csv_path = write_chunk_inventory_outputs(
        chunk_df=chunk_df,
        output_dir=config.output_dir,
        stem=args.chunk_stem,
        write_csv=config.outputs.write_csv,
    )

    message = (
        f'Wrote chunk inventory with {chunk_df.height} chunks to {parquet_path}'
    )
    if csv_path is not None:
        message = f'{message} and {csv_path}'
    print(message)


if __name__ == '__main__':
    main()
