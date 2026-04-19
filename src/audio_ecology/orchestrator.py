"""Pipeline orchestration utilities."""

from __future__ import annotations

from collections.abc import Mapping
import logging

import polars as pl
from polars import DataFrame

from audio_ecology.config import PipelineConfig
from audio_ecology.ingest.inventory import build_and_write_inventory_with_chunks

logger = logging.getLogger(__name__)


def summarise_inventory(inventory_df: pl.DataFrame) -> dict[str, object]:
    """Build a small summary of the audio inventory.

    :param inventory_df: Inventory DataFrame.
    :return: Summary statistics.
    """
    if inventory_df.is_empty():
        return {
            'n_files': 0,
            'n_unreadable_wav': 0,
            'n_missing_timestamp': 0,
            'n_missing_location': 0,
            'n_missing_temperature': 0,
            'n_with_guano': 0,
            'device_ids': [],
            'sample_rate_counts': {},
        }

    sample_rate_counts_df = (
        inventory_df.group_by('sample_rate_hz')
        .len()
        .sort('sample_rate_hz')
    )

    sample_rate_counts = {
        str(row['sample_rate_hz']): row['len']
        for row in sample_rate_counts_df.iter_rows(named=True)
    }

    device_ids = sorted(
        {
            value
            for value in inventory_df.get_column('device_id').to_list()
            if value is not None
        }
    )

    return {
        'n_files': inventory_df.height,
        'n_unreadable_wav': inventory_df.filter(
            ~pl.col('readable_wav')
        ).height,
        'n_missing_timestamp': inventory_df.filter(
            pl.col('timestamp').is_null()
        ).height,
        'n_missing_location': inventory_df.filter(
            pl.col('latitude').is_null() | pl.col('longitude').is_null()
        ).height,
        'n_missing_temperature': inventory_df.filter(
            pl.col('temperature_int_c').is_null()
        ).height,
        'n_with_guano': inventory_df.filter(
            pl.col('guano_present')
        ).height,
        'device_ids': device_ids,
        'sample_rate_counts': sample_rate_counts,
    }


def run_inventory_pipeline(
    config: PipelineConfig,
    stem: str = 'audio_inventory',
) -> tuple[DataFrame, DataFrame | None, dict[str, object]]:
    """Run the inventory pipeline and return data plus summary.

    :param config: Pipeline configuration.
    :param stem: Base output file stem.
    :return: Inventory DataFrame, optional chunk DataFrame, and summary.
    """
    logger.info('Running inventory pipeline')
    inventory_df, chunk_df = build_and_write_inventory_with_chunks(
        config=config,
        stem=stem,
    )
    summary = summarise_inventory(inventory_df)
    logger.info(
        'Inventory pipeline complete: %s files, %s chunk rows',
        summary['n_files'],
        0 if chunk_df is None else chunk_df.height,
    )
    return inventory_df, chunk_df, summary


def format_inventory_summary(summary: Mapping[str, object]) -> str:
    """Format an inventory summary for console output.

    :param summary: Summary dictionary.
    :return: Human-readable summary text.
    """
    lines = [
        f"Files: {summary['n_files']}",
        f"Unreadable WAVs: {summary['n_unreadable_wav']}",
        f"Missing timestamp: {summary['n_missing_timestamp']}",
        f"Missing location: {summary['n_missing_location']}",
        f"Missing internal temperature: {summary['n_missing_temperature']}",
        f"With GUANO: {summary['n_with_guano']}",
        f"Device IDs: {', '.join(summary['device_ids']) or 'None'}",
        'Sample rates:',
    ]

    sample_rate_counts = summary['sample_rate_counts']
    if isinstance(sample_rate_counts, dict) and sample_rate_counts:
        for sample_rate_hz, count in sample_rate_counts.items():
            lines.append(f'  {sample_rate_hz}: {count}')
    else:
        lines.append('  None')

    return '\n'.join(lines)
