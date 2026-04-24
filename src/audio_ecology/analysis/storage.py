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
    """Return the hive-style partition directory name for a backend.

    :param analysis_backend: Canonical backend identifier such as ``birdnet``.
    :return: Partition directory name like ``analysis_backend=birdnet``.
    """
    return f'{ANALYSIS_BACKEND_PARTITION}={analysis_backend}'


def get_detection_root_dir(output_dir: Path) -> Path:
    """Return the root directory for canonical detection datasets.

    :param output_dir: Site-level processed output directory.
    :return: Root directory for shared detection datasets.
    """
    return output_dir / DETECTIONS_DIRNAME


def get_detection_dataset_dir(
    output_dir: Path,
    analysis_backend: str,
) -> Path:
    """Return the dataset directory for one backend's detections.

    :param output_dir: Site-level processed output directory.
    :param analysis_backend: Canonical backend identifier.
    :return: Backend-specific shared detection dataset directory.
    """
    return get_detection_root_dir(output_dir) / backend_partition_name(
        analysis_backend
    )


def get_checkpoint_root_dir(output_dir: Path) -> Path:
    """Return the root directory for analysis checkpoints.

    :param output_dir: Site-level processed output directory.
    :return: Root directory for shared checkpoint storage.
    """
    return output_dir / CHECKPOINTS_DIRNAME


def get_checkpoint_backend_dir(
    output_dir: Path,
    analysis_backend: str,
) -> Path:
    """Return the checkpoint directory for one backend.

    :param output_dir: Site-level processed output directory.
    :param analysis_backend: Canonical backend identifier.
    :return: Backend-specific checkpoint directory.
    """
    return get_checkpoint_root_dir(output_dir) / backend_partition_name(
        analysis_backend
    )


def _partition_date_from_row(detection_row: dict[str, object]) -> str:
    """Return a YYYY-MM-DD partition key for one detection row.

    The function prefers ``detection_timestamp`` and falls back to ``timestamp``.

    :param detection_row: One normalized detection record.
    :return: ISO date string or ``unknown`` when no parseable timestamp exists.
    """
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
    """Return the hive-style partition directory for one date.

    :param dataset_dir: Backend-specific detection dataset directory.
    :param partition_date: ISO date string or ``unknown``.
    :return: Partition directory for that date.
    """
    if partition_date == 'unknown':
        return dataset_dir / 'date=unknown'

    year, month, day = partition_date.split('-')
    return dataset_dir / f'year={year}' / f'month={month}' / f'day={day}'


def load_detection_dataframe(
    detections_path: Path,
    schema: dict[str, pl.DataType],
) -> pl.DataFrame:
    """Load detections from either a parquet file or a partitioned dataset.

    :param detections_path: Path to a detection parquet file or dataset root.
    :param schema: Schema to use when returning an empty DataFrame.
    :return: Loaded detection rows.
    :raises FileNotFoundError: If the given path does not exist.
    """
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
    write_csv: bool = False,
) -> Path:
    """Write detections to a date-partitioned parquet dataset.

    :param detections_df: Normalized detection rows to write.
    :param dataset_dir: Backend-specific dataset root directory.
    :param stem: Base file stem for parquet files within each partition.
    :param write_csv: Whether to also write a CSV copy in each partition.
    :return: The dataset root directory.
    """
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
        if write_csv:
            csv_path = partition_dir / f'{stem}.csv'
            logger.info(
                'Writing %d detections CSV for %s to %s',
                partition_df.height,
                partition_date,
                csv_path,
            )
            partition_df.write_csv(csv_path)

    return dataset_dir
