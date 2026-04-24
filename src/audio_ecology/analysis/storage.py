"""Shared storage helpers for model outputs and checkpoints."""

from __future__ import annotations

from datetime import datetime
import logging
from pathlib import Path

import polars as pl

logger = logging.getLogger(__name__)

ANALYSIS_BACKEND_PARTITION = 'analysis_backend'
DETECTIONS_DIRNAME = 'detections'
CHECKPOINTS_DIRNAME = 'checkpoints'
DETECTIONS_STEM = 'detections'


def backend_partition_name(analysis_backend: str) -> str:
    """Return the hive-style partition directory name for a backend."""
    return f'{ANALYSIS_BACKEND_PARTITION}={analysis_backend}'


def get_detection_root_dir(output_dir: Path) -> Path:
    """Return the root directory for canonical detection datasets."""
    return output_dir / DETECTIONS_DIRNAME


def get_detection_dataset_dir(
    output_dir: Path,
    analysis_backend: str,
) -> Path:
    """Return the dataset directory for one backend's detections."""
    return get_detection_root_dir(output_dir) / backend_partition_name(
        analysis_backend
    )


def get_checkpoint_root_dir(output_dir: Path) -> Path:
    """Return the root directory for analysis checkpoints."""
    return output_dir / CHECKPOINTS_DIRNAME


def get_checkpoint_backend_dir(
    output_dir: Path,
    analysis_backend: str,
) -> Path:
    """Return the checkpoint directory for one backend."""
    return get_checkpoint_root_dir(output_dir) / backend_partition_name(
        analysis_backend
    )


def _partition_date_from_row(detection_row: dict[str, object]) -> str:
    """Return a YYYY-MM-DD partition key for one detection row."""
    for field_name in ('detection_timestamp', 'timestamp'):
        timestamp_value = detection_row.get(field_name)
        if timestamp_value is None:
            continue

        timestamp_text = str(timestamp_value)
        try:
            return datetime.fromisoformat(timestamp_text).date().isoformat()
        except ValueError:
            logger.debug(
                'Could not parse detection timestamp %s for partitioning',
                timestamp_text,
            )

    return 'unknown'


def get_date_partition_dir(dataset_dir: Path, partition_date: str) -> Path:
    """Return the hive-style partition directory for one date."""
    if partition_date == 'unknown':
        return dataset_dir / 'date=unknown'

    year, month, day = partition_date.split('-')
    return dataset_dir / f'year={year}' / f'month={month}' / f'day={day}'


def load_detection_dataframe(
    detections_path: Path,
    schema: dict[str, pl.DataType],
) -> pl.DataFrame:
    """Load detections from either a parquet file or a partitioned dataset."""
    if detections_path.is_file():
        return pl.read_parquet(detections_path)

    if detections_path.is_dir():
        parquet_paths = sorted(detections_path.rglob('*.parquet'))
        if not parquet_paths:
            logger.info('No parquet files found in detection dataset %s', detections_path)
            return pl.DataFrame(schema=schema)
        return pl.read_parquet([str(path) for path in parquet_paths])

    raise FileNotFoundError(f'Detections parquet not found: {detections_path}')


def write_detection_dataset(
    detections_df: pl.DataFrame,
    dataset_dir: Path,
    stem: str = DETECTIONS_STEM,
) -> Path:
    """Write detections to a date-partitioned parquet dataset."""
    logger.info('Writing detections parquet dataset to %s', dataset_dir)
    if detections_df.is_empty():
        dataset_dir.mkdir(parents=True, exist_ok=True)
        return dataset_dir

    detections_by_date: dict[str, list[dict[str, object]]] = {}
    for detection_row in detections_df.iter_rows(named=True):
        partition_date = _partition_date_from_row(detection_row)
        detections_by_date.setdefault(partition_date, []).append(detection_row)

    for partition_date, partition_rows in sorted(detections_by_date.items()):
        partition_dir = get_date_partition_dir(dataset_dir, partition_date)
        partition_dir.mkdir(parents=True, exist_ok=True)
        partition_path = partition_dir / f'{stem}.parquet'
        partition_df = pl.DataFrame(
            partition_rows,
            schema=detections_df.schema,
        )
        logger.info(
            'Writing %d detections for %s to %s',
            partition_df.height,
            partition_date,
            partition_path,
        )
        partition_df.write_parquet(partition_path)

    return dataset_dir
