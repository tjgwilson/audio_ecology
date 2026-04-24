"""Inventory building and Polars output utilities."""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl
from polars import DataFrame

from audio_ecology.config import PipelineConfig
from audio_ecology.ingest.chunking import (
    build_chunk_records,
    write_chunk_wavs,
)
from audio_ecology.ingest.discovery import discover_wav_files
from audio_ecology.ingest.metadata import build_audio_file_record
from audio_ecology.models import AudioChunkRecord, AudioFileRecord

logger = logging.getLogger(__name__)


def build_inventory_records(config: PipelineConfig) -> list[AudioFileRecord]:
    """Build inventory records for all WAV files in a directory tree.

    :param config: Pipeline configuration.
    :return: List of inventory records.
    """
    wav_files = discover_wav_files(config.input_dir)
    logger.info('Building inventory records for %d WAV files', len(wav_files))
    records = []
    for index, file_path in enumerate(wav_files, start=1):
        logger.debug(
            'Building inventory record %d/%d for %s',
            index,
            len(wav_files),
            file_path,
        )
        records.append(build_audio_file_record(file_path, config))

    logger.info('Built %d inventory records', len(records))
    return records


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


def chunk_records_to_polars(records: list[AudioChunkRecord]) -> pl.DataFrame:
    """Convert chunk records to a Polars DataFrame.

    :param records: Inventory Records.
    :return: Polars DataFrame.
    """
    if not records:
        return pl.DataFrame(
            schema={
                'parent_file_path': pl.Utf8,
                'parent_file_name': pl.Utf8,
                'chunk_file_path': pl.Utf8,
                'device_id': pl.Utf8,
                'device_label': pl.Utf8,
                'deployment_id': pl.Utf8,
                'habitat_label': pl.Utf8,
                'chunk_index': pl.Int64,
                'chunk_start_s': pl.Float64,
                'chunk_end_s': pl.Float64,
                'chunk_duration_s': pl.Float64,
                'timestamp': pl.Utf8,
                'latitude': pl.Float64,
                'longitude': pl.Float64,
                'sample_rate_hz': pl.Int64,
                'analysis_targets': pl.List(pl.Utf8),
            }
        )

    rows = []
    for record in records:
        row = record.model_dump(mode='json')
        row['parent_file_path'] = str(record.parent_file_path)
        row['chunk_file_path'] = (
            str(record.chunk_file_path)
            if record.chunk_file_path is not None
            else None
        )
        rows.append(row)

    return pl.DataFrame(rows)


def write_inventory_outputs(
    inventory_df: pl.DataFrame,
    output_dir: Path,
    stem: str = 'audio_inventory',
    write_csv: bool = False,
) -> tuple[Path, Path | None]:
    """Write inventory outputs.

    :param inventory_df: Inventory DataFrame.
    :param output_dir: Output directory.
    :param stem: Base file name stem.
    :param write_csv: Whether to also write a CSV copy.
    :return: Paths to Parquet and optional CSV outputs.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = output_dir / f'{stem}.parquet'
    csv_path = output_dir / f'{stem}.csv' if write_csv else None

    logger.info('Writing inventory parquet to %s', parquet_path)
    inventory_df.write_parquet(parquet_path)
    if csv_path is not None:
        logger.info('Writing inventory CSV to %s', csv_path)
        inventory_df.write_csv(csv_path)
    logger.info('Wrote inventory outputs with %d rows', inventory_df.height)

    return parquet_path, csv_path


def write_chunk_inventory_outputs(
    chunk_df: pl.DataFrame,
    output_dir: Path,
    stem: str = 'audio_chunks',
    write_csv: bool = False,
) -> tuple[Path, Path | None]:
    """Write chunk inventory outputs.

    :param chunk_df: Chunk DataFrame.
    :param output_dir: Output directory.
    :param stem: Base file name stem.
    :param write_csv: Whether to also write a CSV copy.
    :return: Paths to Parquet and optional CSV outputs.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = output_dir / f'{stem}.parquet'
    csv_path = output_dir / f'{stem}.csv' if write_csv else None

    logger.info('Writing chunk inventory parquet to %s', parquet_path)
    chunk_df.write_parquet(parquet_path)

    if csv_path is not None:
        logger.info('Writing chunk inventory CSV to %s', csv_path)
        chunk_df_for_csv = chunk_df.with_columns(
            pl.col('analysis_targets').list.join(';').alias('analysis_targets')
        )
        chunk_df_for_csv.write_csv(csv_path)
    logger.info('Wrote chunk inventory outputs with %d rows', chunk_df.height)

    return parquet_path, csv_path


def build_and_write_inventory(
    config: PipelineConfig,
    stem: str = 'audio_inventory',
) -> DataFrame:
    """Build inventory records and write them to disk.

    :param config: Pipeline configuration.
    :param stem: Base file name stem.
    :return: Inventory Polars DataFrame.
    """
    inventory_df, _ = build_and_write_inventory_with_chunks(
        config=config,
        stem=stem,
    )
    return inventory_df


def build_and_write_inventory_with_chunks(
    config: PipelineConfig,
    stem: str = 'audio_inventory',
) -> tuple[DataFrame, DataFrame | None]:
    """Build inventory and optional chunk records, then write them to disk.

    :param config: Pipeline configuration.
    :param stem: Base file name stem.
    :return: Inventory DataFrame and optional chunk DataFrame.
    """
    logger.info('Starting inventory build for %s', config.input_dir)
    records = build_inventory_records(config)
    inventory_df = records_to_polars(records)
    write_inventory_outputs(
        inventory_df=inventory_df,
        output_dir=config.output_dir,
        stem=stem,
        write_csv=config.outputs.write_csv,
    )

    chunk_df: pl.DataFrame | None = None
    if config.chunking.enabled:
        logger.info('Chunking is enabled')
        chunk_records = build_chunk_records(
            records=records,
            chunking_config=config.chunking,
            analysis_targets=config.chunking.analysis_targets,
        )

        chunk_records = write_chunk_wavs(
            chunk_records=chunk_records,
            chunking_config=config.chunking,
            default_output_dir=config.output_dir,
        )

        if config.chunking.write_chunk_inventory:
            chunk_df = chunk_records_to_polars(chunk_records)
            write_chunk_inventory_outputs(
                chunk_df=chunk_df,
                output_dir=config.output_dir,
                stem='audio_chunks',
                write_csv=config.outputs.write_csv,
            )
    else:
        logger.info('Chunking is disabled')

    logger.info('Finished inventory build')
    return inventory_df, chunk_df
