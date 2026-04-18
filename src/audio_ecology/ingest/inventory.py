"""Inventory building and Polars output utilities."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from audio_ecology.config import PipelineConfig
from audio_ecology.ingest.discovery import discover_wav_files
from audio_ecology.ingest.metadata import build_audio_file_record
from audio_ecology.models import AudioFileRecord


def build_inventory_records(config: PipelineConfig) -> list[AudioFileRecord]:
    """Build inventory records for all WAV files in a directory tree.

    :param config: Pipeline configuration.
    :return: List of inventory records.
    """
    wav_files = discover_wav_files(config.input_dir)
    return [
        build_audio_file_record(file_path, config)
        for file_path in wav_files
    ]


def records_to_polars(records: list[AudioFileRecord]) -> pl.DataFrame:
    """Convert inventory records to a Polars DataFrame.

    :param records: Inventory records.
    :return: Polars DataFrame.
    """
    rows = []
    for record in records:
        row = record.model_dump(mode='json')
        row['file_path'] = str(record.file_path)
        rows.append(row)

    return pl.DataFrame(rows)


def write_inventory_outputs(
    inventory_df: pl.DataFrame,
    output_dir: Path,
    stem: str = 'audio_inventory',
) -> tuple[Path, Path]:
    """Write inventory outputs to Parquet and CSV.

    :param inventory_df: Inventory DataFrame.
    :param output_dir: Output directory.
    :param stem: Base file name stem.
    :return: Paths to Parquet and CSV outputs.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = output_dir / f'{stem}.parquet'
    csv_path = output_dir / f'{stem}.csv'

    inventory_df.write_parquet(parquet_path)
    inventory_df.write_csv(csv_path)

    return parquet_path, csv_path


def build_and_write_inventory(
    config: PipelineConfig,
    stem: str = 'audio_inventory',
) -> pl.DataFrame:
    """Build inventory records and write them to disk.

    :param config: Pipeline configuration.
    :param stem: Base file name stem.
    :return: Inventory Polars DataFrame.
    """
    records = build_inventory_records(config)
    inventory_df = records_to_polars(records)
    write_inventory_outputs(
        inventory_df=inventory_df,
        output_dir=config.output_dir,
        stem=stem,
    )
    return inventory_df