"""Noisy-OR window evidence summaries for acoustic detections."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta
import logging
from pathlib import Path
from typing import Any

import polars as pl

from audio_ecology.config import DetectionUncertaintyConfig

logger = logging.getLogger(__name__)

DETECTION_WINDOW_EVIDENCE_STEM = 'detection_window_evidence'
BIRD_WINDOW_EVIDENCE_STEM = DETECTION_WINDOW_EVIDENCE_STEM

DETECTION_WINDOW_EVIDENCE_SCHEMA = {
    'file_path': pl.Utf8,
    'file_name': pl.Utf8,
    'analysis_backend': pl.Utf8,
    'model_name': pl.Utf8,
    'window_index': pl.Int64,
    'window_start_s': pl.Float64,
    'window_end_s': pl.Float64,
    'window_start_timestamp': pl.Utf8,
    'window_end_timestamp': pl.Utf8,
    'scientific_name': pl.Utf8,
    'common_name': pl.Utf8,
    'n_detections': pl.Int64,
    'n_events': pl.Int64,
    'max_confidence': pl.Float64,
    'mean_confidence': pl.Float64,
    'mean_event_confidence': pl.Float64,
    'noisy_or_evidence': pl.Float64,
    'detection_uncertainty': pl.Float64,
    'evidence_class': pl.Utf8,
    'period_duration_s': pl.Float64,
    'event_gap_s': pl.Float64,
}
BIRD_WINDOW_EVIDENCE_SCHEMA = DETECTION_WINDOW_EVIDENCE_SCHEMA


def _parse_timestamp(value: object) -> datetime | None:
    """Parse a datetime-like value from normalized detection rows."""
    if value is None:
        return None

    if isinstance(value, datetime):
        return value

    value_text = str(value)
    if not value_text:
        return None

    try:
        return datetime.fromisoformat(value_text.replace('Z', '+00:00'))
    except ValueError:
        return None


def _timestamp_at(file_timestamp: object, offset_s: float) -> str | None:
    """Return an ISO timestamp offset from the source file timestamp."""
    timestamp = _parse_timestamp(file_timestamp)
    if timestamp is None:
        return None
    return (timestamp + timedelta(seconds=offset_s)).isoformat()


def _detection_timestamp_at(
    row: dict[str, Any],
    offset_column: str,
) -> datetime | None:
    """Return an absolute detection timestamp from row timestamp plus offset."""
    timestamp = _parse_timestamp(row.get('timestamp'))
    if timestamp is None:
        return None
    return timestamp + timedelta(seconds=float(row[offset_column]))


def _comparable_timestamp(
    timestamp: datetime,
    reference: datetime,
) -> datetime:
    """Align timestamp timezone awareness with a reference timestamp."""
    if timestamp.tzinfo is None and reference.tzinfo is not None:
        return timestamp.replace(tzinfo=reference.tzinfo)

    if timestamp.tzinfo is not None and reference.tzinfo is None:
        return timestamp.replace(tzinfo=None)

    return timestamp


def _clamp_score(value: object) -> float:
    """Clamp a confidence score to the noisy-OR input range."""
    return min(1.0, max(0.0, float(value)))


def _noisy_or(scores: list[float]) -> float:
    """Combine independent-ish event scores using noisy-OR."""
    miss_probability = 1.0
    for score in scores:
        miss_probability *= 1.0 - score
    return 1.0 - miss_probability


def _evidence_class(
    evidence: float,
    max_confidence: float,
    n_events: int,
    config: DetectionUncertaintyConfig,
) -> str:
    """Classify window evidence with light guardrails against weak-score piles."""
    if evidence >= config.strong_threshold and (
        max_confidence >= 0.80 or n_events >= 2
    ):
        return 'strong'

    if evidence >= config.probable_threshold and max_confidence >= 0.50:
        return 'probable'

    if evidence >= config.possible_threshold:
        return 'possible'

    return 'weak'


def _event_scores(
    rows: list[dict[str, Any]],
    event_gap_s: float,
) -> list[float]:
    """Collapse nearby same-species detections into events and score each event."""
    if not rows:
        return []

    sorted_rows = sorted(rows, key=lambda row: float(row['detection_start_s']))
    scores: list[float] = []
    current_event_max = _clamp_score(sorted_rows[0]['confidence'])
    previous_end_s = float(sorted_rows[0]['detection_end_s'])

    for row in sorted_rows[1:]:
        start_s = float(row['detection_start_s'])
        end_s = float(row['detection_end_s'])
        confidence = _clamp_score(row['confidence'])

        if start_s - previous_end_s > event_gap_s:
            scores.append(current_event_max)
            current_event_max = confidence
        else:
            current_event_max = max(current_event_max, confidence)

        previous_end_s = max(previous_end_s, end_s)

    scores.append(current_event_max)
    return scores


def _event_scores_by_timestamp(
    rows: list[dict[str, Any]],
    event_gap_s: float,
) -> list[float]:
    """Collapse nearby same-species detections using absolute timestamps."""
    timed_rows = [
        (
            _detection_timestamp_at(row, 'detection_start_s'),
            _detection_timestamp_at(row, 'detection_end_s'),
            _clamp_score(row['confidence']),
        )
        for row in rows
    ]
    timed_rows = [
        row
        for row in timed_rows
        if row[0] is not None and row[1] is not None
    ]
    if not timed_rows:
        return []

    sorted_rows = sorted(timed_rows, key=lambda row: row[0])
    first_start, first_end, first_confidence = sorted_rows[0]
    if first_start is None or first_end is None:
        return []

    scores: list[float] = []
    current_event_max = first_confidence
    previous_end = first_end

    for start, end, confidence in sorted_rows[1:]:
        if start is None or end is None:
            continue

        if (start - previous_end).total_seconds() > event_gap_s:
            scores.append(current_event_max)
            current_event_max = confidence
        else:
            current_event_max = max(current_event_max, confidence)

        previous_end = max(previous_end, end)

    scores.append(current_event_max)
    return scores


def _required_detection_columns() -> set[str]:
    """Return columns required for noisy-OR detection aggregation."""
    return {
        'file_path',
        'file_name',
        'detection_start_s',
        'detection_end_s',
        'scientific_name',
        'common_name',
        'confidence',
    }


def _optional_group_value(row: dict[str, Any], column: str) -> str | None:
    """Return an optional grouping value when the detection schema provides it."""
    value = row.get(column)
    if value is None:
        return None
    return str(value)


def _check_required_columns(detections_df: pl.DataFrame) -> None:
    """Validate detection columns for noisy-OR aggregation."""
    missing_columns = _required_detection_columns().difference(detections_df.columns)
    if missing_columns:
        raise ValueError(
            'Detections DataFrame is missing required columns: '
            f'{sorted(missing_columns)}'
        )


def _log_detection_input_summary(
    detections_df: pl.DataFrame,
    start_time: datetime,
    end_time: datetime,
    config: DetectionUncertaintyConfig,
) -> None:
    """Log useful diagnostics before window evidence aggregation."""
    logger.info(
        'Building detection evidence for %s to %s; input rows=%d, '
        'min_confidence=%.3f, event_gap_s=%.3f',
        start_time.isoformat(),
        end_time.isoformat(),
        detections_df.height,
        config.min_confidence,
        config.event_gap_s,
    )

    if detections_df.is_empty():
        logger.info('Detection input is empty')
        return

    species_count = (
        detections_df.select(['scientific_name', 'common_name']).unique().height
    )
    logger.info('Detection input contains %d unique species/taxa', species_count)

    if 'analysis_backend' in detections_df.columns:
        backend_counts = (
            detections_df.group_by('analysis_backend')
            .len()
            .sort('len', descending=True)
            .to_dicts()
        )
        logger.info('Detection rows by analysis_backend: %s', backend_counts)

    if 'model_name' in detections_df.columns:
        model_counts = (
            detections_df.group_by('model_name')
            .len()
            .sort('len', descending=True)
            .head(10)
            .to_dicts()
        )
        logger.info('Top detection rows by model_name: %s', model_counts)

    confidence_stats = detections_df.select(
        pl.col('confidence').min().alias('min_confidence'),
        pl.col('confidence').mean().alias('mean_confidence'),
        pl.col('confidence').max().alias('max_confidence'),
    ).row(0, named=True)
    logger.info('Detection confidence stats: %s', confidence_stats)


def _log_period_filter_summary(
    total_rows: int,
    rows_below_confidence: int,
    rows_missing_timestamp: int,
    rows_before_period: int,
    rows_in_period: int,
    rows_after_period: int,
    min_detection_time: datetime | None,
    max_detection_time: datetime | None,
    groups: dict[tuple[str | None, str | None, str, str], list[dict[str, Any]]],
) -> None:
    """Log diagnostics after period filtering."""
    logger.info(
        'Detection period filter summary: total=%d, below_confidence=%d, '
        'missing_timestamp=%d, before_period=%d, in_period=%d, after_period=%d',
        total_rows,
        rows_below_confidence,
        rows_missing_timestamp,
        rows_before_period,
        rows_in_period,
        rows_after_period,
    )

    if min_detection_time is not None and max_detection_time is not None:
        logger.info(
            'Detection absolute start-time range after confidence filtering: '
            '%s to %s',
            min_detection_time.isoformat(),
            max_detection_time.isoformat(),
        )

    if not groups:
        logger.warning('No detections remained in the configured evidence period')
        return

    group_rows = [
        {
            'analysis_backend': analysis_backend,
            'model_name': model_name,
            'scientific_name': scientific_name,
            'common_name': common_name,
            'n_detections': len(rows),
            'max_confidence': max(_clamp_score(row['confidence']) for row in rows),
        }
        for (
            analysis_backend,
            model_name,
            scientific_name,
            common_name,
        ), rows in groups.items()
    ]
    logger.info('Evidence groups retained in period: %d', len(group_rows))
    logger.info(
        'Top evidence groups by detection count: %s',
        sorted(
            group_rows,
            key=lambda row: (row['n_detections'], row['max_confidence']),
            reverse=True,
        )[:20],
    )


def build_noisy_or_species_windows(
    detections_df: pl.DataFrame,
    window_s: float = 600.0,
    config: DetectionUncertaintyConfig | None = None,
) -> pl.DataFrame:
    """Summarize detections into fixed-length noisy-OR evidence windows."""
    uncertainty_config = config or DetectionUncertaintyConfig()
    if window_s <= 0:
        raise ValueError('window_s must be greater than 0')

    if detections_df.is_empty():
        return pl.DataFrame(schema=DETECTION_WINDOW_EVIDENCE_SCHEMA)

    _check_required_columns(detections_df)

    groups: dict[
        tuple[str, str, str | None, str | None, int, str, str],
        list[dict[str, Any]],
    ] = defaultdict(list)
    for row in detections_df.iter_rows(named=True):
        confidence = _clamp_score(row['confidence'])
        if confidence < uncertainty_config.min_confidence:
            continue

        detection_start_s = float(row['detection_start_s'])
        window_index = int(detection_start_s // window_s)
        group_key = (
            str(row['file_path']),
            str(row['file_name']),
            _optional_group_value(row, 'analysis_backend'),
            _optional_group_value(row, 'model_name'),
            window_index,
            str(row['scientific_name']),
            str(row['common_name']),
        )
        groups[group_key].append(row)

    output_rows: list[dict[str, object]] = []
    for (
        file_path,
        file_name,
        analysis_backend,
        model_name,
        window_index,
        scientific_name,
        common_name,
    ), rows in groups.items():
        window_start_s = float(window_index * window_s)
        window_end_s = window_start_s + window_s
        confidences = [_clamp_score(row['confidence']) for row in rows]
        event_scores = _event_scores(
            rows=rows,
            event_gap_s=uncertainty_config.event_gap_s,
        )
        evidence = _noisy_or(event_scores)
        max_confidence = max(confidences)

        output_rows.append(
            {
                'file_path': file_path,
                'file_name': file_name,
                'analysis_backend': analysis_backend,
                'model_name': model_name,
                'window_index': window_index,
                'window_start_s': window_start_s,
                'window_end_s': window_end_s,
                'window_start_timestamp': _timestamp_at(
                    rows[0].get('timestamp'),
                    window_start_s,
                ),
                'window_end_timestamp': _timestamp_at(
                    rows[0].get('timestamp'),
                    window_end_s,
                ),
                'scientific_name': scientific_name,
                'common_name': common_name,
                'n_detections': len(rows),
                'n_events': len(event_scores),
                'max_confidence': max_confidence,
                'mean_confidence': sum(confidences) / len(confidences),
                'mean_event_confidence': sum(event_scores) / len(event_scores),
                'noisy_or_evidence': evidence,
                'detection_uncertainty': 1.0 - evidence,
                'evidence_class': _evidence_class(
                    evidence=evidence,
                    max_confidence=max_confidence,
                    n_events=len(event_scores),
                    config=uncertainty_config,
                ),
                'period_duration_s': window_s,
                'event_gap_s': uncertainty_config.event_gap_s,
            }
        )

    if not output_rows:
        return pl.DataFrame(schema=DETECTION_WINDOW_EVIDENCE_SCHEMA)

    return pl.DataFrame(output_rows, schema=DETECTION_WINDOW_EVIDENCE_SCHEMA).sort(
        [
            'file_path',
            'window_index',
            'analysis_backend',
            'model_name',
            'scientific_name',
        ]
    )


def build_noisy_or_species_time_period(
    detections_df: pl.DataFrame,
    start_time: datetime,
    end_time: datetime,
    config: DetectionUncertaintyConfig | None = None,
) -> pl.DataFrame:
    """Summarize detections into species evidence for one absolute time period."""
    uncertainty_config = config or DetectionUncertaintyConfig()
    if end_time <= start_time:
        raise ValueError('end_time must be after start_time')

    if detections_df.is_empty():
        logger.info(
            'Building detection evidence for %s to %s; input rows=0',
            start_time.isoformat(),
            end_time.isoformat(),
        )
        return pl.DataFrame(schema=DETECTION_WINDOW_EVIDENCE_SCHEMA)

    _check_required_columns(detections_df)
    if 'timestamp' not in detections_df.columns:
        raise ValueError(
            'Detections DataFrame is missing required columns: [\'timestamp\']'
        )

    _log_detection_input_summary(
        detections_df=detections_df,
        start_time=start_time,
        end_time=end_time,
        config=uncertainty_config,
    )

    groups: dict[tuple[str | None, str | None, str, str], list[dict[str, Any]]] = (
        defaultdict(list)
    )
    rows_below_confidence = 0
    rows_missing_timestamp = 0
    rows_before_period = 0
    rows_in_period = 0
    rows_after_period = 0
    min_detection_time: datetime | None = None
    max_detection_time: datetime | None = None

    for row in detections_df.iter_rows(named=True):
        confidence = _clamp_score(row['confidence'])
        if confidence < uncertainty_config.min_confidence:
            rows_below_confidence += 1
            continue

        detection_start = _detection_timestamp_at(row, 'detection_start_s')
        if detection_start is None:
            rows_missing_timestamp += 1
            continue

        detection_start = _comparable_timestamp(detection_start, start_time)
        min_detection_time = (
            detection_start
            if min_detection_time is None
            else min(min_detection_time, detection_start)
        )
        max_detection_time = (
            detection_start
            if max_detection_time is None
            else max(max_detection_time, detection_start)
        )

        if detection_start < start_time:
            rows_before_period += 1
            continue

        if detection_start >= end_time:
            rows_after_period += 1
            continue

        rows_in_period += 1
        group_key = (
            _optional_group_value(row, 'analysis_backend'),
            _optional_group_value(row, 'model_name'),
            str(row['scientific_name']),
            str(row['common_name']),
        )
        groups[group_key].append(row)

    _log_period_filter_summary(
        total_rows=detections_df.height,
        rows_below_confidence=rows_below_confidence,
        rows_missing_timestamp=rows_missing_timestamp,
        rows_before_period=rows_before_period,
        rows_in_period=rows_in_period,
        rows_after_period=rows_after_period,
        min_detection_time=min_detection_time,
        max_detection_time=max_detection_time,
        groups=groups,
    )

    output_rows: list[dict[str, object]] = []
    period_duration_s = (end_time - start_time).total_seconds()
    for (analysis_backend, model_name, scientific_name, common_name), rows in (
        groups.items()
    ):
        confidences = [_clamp_score(row['confidence']) for row in rows]
        event_scores = _event_scores_by_timestamp(
            rows=rows,
            event_gap_s=uncertainty_config.event_gap_s,
        )
        if not event_scores:
            continue

        evidence = _noisy_or(event_scores)
        max_confidence = max(confidences)
        output_rows.append(
            {
                'file_path': None,
                'file_name': None,
                'analysis_backend': analysis_backend,
                'model_name': model_name,
                'window_index': 0,
                'window_start_s': 0.0,
                'window_end_s': period_duration_s,
                'window_start_timestamp': start_time.isoformat(),
                'window_end_timestamp': end_time.isoformat(),
                'scientific_name': scientific_name,
                'common_name': common_name,
                'n_detections': len(rows),
                'n_events': len(event_scores),
                'max_confidence': max_confidence,
                'mean_confidence': sum(confidences) / len(confidences),
                'mean_event_confidence': sum(event_scores) / len(event_scores),
                'noisy_or_evidence': evidence,
                'detection_uncertainty': 1.0 - evidence,
                'evidence_class': _evidence_class(
                    evidence=evidence,
                    max_confidence=max_confidence,
                    n_events=len(event_scores),
                    config=uncertainty_config,
                ),
                'period_duration_s': period_duration_s,
                'event_gap_s': uncertainty_config.event_gap_s,
            }
        )

    if not output_rows:
        logger.info('No evidence rows were produced for the configured period')
        return pl.DataFrame(schema=DETECTION_WINDOW_EVIDENCE_SCHEMA)

    logger.info('Produced %d detection evidence rows', len(output_rows))
    return pl.DataFrame(output_rows, schema=DETECTION_WINDOW_EVIDENCE_SCHEMA).sort(
        ['analysis_backend', 'model_name', 'scientific_name']
    )


def write_noisy_or_species_windows(
    evidence_df: pl.DataFrame,
    output_dir: Path,
    stem: str = DETECTION_WINDOW_EVIDENCE_STEM,
    write_csv: bool = False,
) -> tuple[Path, Path | None]:
    """Write window-level noisy-OR evidence outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = output_dir / f'{stem}.parquet'
    csv_path = output_dir / f'{stem}.csv' if write_csv else None

    logger.info('Writing window evidence parquet to %s', parquet_path)
    evidence_df.write_parquet(parquet_path)
    if csv_path is not None:
        logger.info('Writing window evidence CSV to %s', csv_path)
        evidence_df.write_csv(csv_path)

    logger.info('Wrote window evidence outputs with %d rows', evidence_df.height)
    return parquet_path, csv_path
