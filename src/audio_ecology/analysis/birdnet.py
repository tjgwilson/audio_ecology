"""BirdNET Python package integration utilities."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
import importlib
import importlib.util
import logging
from pathlib import Path
from typing import Any

import polars as pl

from audio_ecology.config import PipelineConfig
from audio_ecology.models import BirdDetectionRecord

logger = logging.getLogger(__name__)

BIRDNET_BACKEND = 'birdnet'
BIRDNET_DETECTIONS_STEM = 'birdnet_detections'

BIRDNET_DETECTION_SCHEMA = {
    'file_path': pl.Utf8,
    'file_name': pl.Utf8,
    'detection_start_s': pl.Float64,
    'detection_end_s': pl.Float64,
    'detection_duration_s': pl.Float64,
    'detection_timestamp': pl.Utf8,
    'timestamp': pl.Utf8,
    'latitude': pl.Float64,
    'longitude': pl.Float64,
    'temperature_int_c': pl.Float64,
    'scientific_name': pl.Utf8,
    'common_name': pl.Utf8,
    'confidence': pl.Float64,
    'analysis_backend': pl.Utf8,
    'model_name': pl.Utf8,
    'source_result_path': pl.Utf8,
}


def get_birdnet_output_dir(config: PipelineConfig) -> Path:
    """Return the directory used for normalized BirdNET outputs."""
    if config.birdnet.output_dir is not None:
        return config.birdnet.output_dir
    return config.output_dir / 'birdnet'


def birdnet_week_from_timestamp(timestamp: datetime | None) -> int | None:
    """Map a timestamp to BirdNET's 1-48 four-weeks-per-month index."""
    if timestamp is None:
        return None

    week_in_month = min(4, ((timestamp.day - 1) // 7) + 1)
    return ((timestamp.month - 1) * 4) + week_in_month


def load_birdnet_model(config: PipelineConfig) -> Any:
    """Load the configured BirdNET acoustic model."""
    logger.info(
        'Loading BirdNET model acoustic-%s-%s',
        config.birdnet.model_version,
        config.birdnet.model_backend,
    )
    if importlib.util.find_spec('birdnet') is None:
        raise RuntimeError(
            'The birdnet package is not installed in this Python environment. '
            "Install it with: python -m pip install -e '.[birds]'"
        )

    birdnet = importlib.import_module('birdnet')
    model = birdnet.load(
        'acoustic',
        config.birdnet.model_version,
        config.birdnet.model_backend,
    )
    logger.info('Loaded BirdNET model')
    return model


def _prediction_rows_to_polars(predictions: Any) -> pl.DataFrame:
    """Convert BirdNET prediction outputs to a Polars DataFrame."""
    if predictions is None:
        return pl.DataFrame()

    if isinstance(predictions, pl.DataFrame):
        return predictions

    if isinstance(predictions, dict):
        return pl.DataFrame(predictions)

    dtype_names = getattr(getattr(predictions, 'dtype', None), 'names', None)
    if dtype_names is not None:
        return pl.DataFrame(
            [
                {
                    field_name: prediction_row[field_name].item()
                    if hasattr(prediction_row[field_name], 'item')
                    else prediction_row[field_name]
                    for field_name in dtype_names
                }
                for prediction_row in predictions
            ]
        )

    if hasattr(predictions, 'to_dict'):
        try:
            return pl.DataFrame(predictions.to_dict(orient='records'))
        except TypeError:
            pass

    prediction_rows = list(predictions)
    if not prediction_rows:
        return pl.DataFrame()

    row_dtype_names = getattr(
        getattr(prediction_rows[0], 'dtype', None),
        'names',
        None,
    )
    if row_dtype_names is not None:
        return pl.DataFrame(
            [
                {
                    field_name: prediction_row[field_name].item()
                    if hasattr(prediction_row[field_name], 'item')
                    else prediction_row[field_name]
                    for field_name in row_dtype_names
                }
                for prediction_row in prediction_rows
            ]
        )

    return pl.DataFrame(prediction_rows)


def _time_to_seconds(value: object) -> float:
    """Convert BirdNET time values to seconds."""
    if isinstance(value, int | float):
        return float(value)

    value_text = str(value)
    if ':' not in value_text:
        return float(value_text)

    parts = value_text.split(':')
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return (float(hours) * 3600) + (float(minutes) * 60) + float(seconds)

    if len(parts) == 2:
        minutes, seconds = parts
        return (float(minutes) * 60) + float(seconds)

    return float(value_text)


def _split_species_name(species_name: str) -> tuple[str, str]:
    """Split BirdNET species names into scientific and common names."""
    if '_' not in species_name:
        return species_name, ''

    scientific_name, common_name = species_name.split('_', maxsplit=1)
    return scientific_name, common_name


def _inventory_metadata_by_path(
    inventory_df: pl.DataFrame,
) -> dict[str, dict[str, object]]:
    """Build a lookup from absolute file path to inventory metadata."""
    if inventory_df.is_empty():
        return {}

    return {
        str(Path(str(row['file_path'])).resolve()): row
        for row in inventory_df.iter_rows(named=True)
        if row.get('file_path') is not None
    }


def _inventory_metadata_by_file_name(
    inventory_df: pl.DataFrame,
) -> dict[str, dict[str, object]]:
    """Build a lookup from file name to inventory metadata."""
    if inventory_df.is_empty():
        return {}

    return {
        str(row['file_name']): row
        for row in inventory_df.iter_rows(named=True)
        if row.get('file_name') is not None
    }


def _metadata_for_prediction(
    prediction_row: dict[str, object],
    inventory_by_path: dict[str, dict[str, object]],
    inventory_by_file_name: dict[str, dict[str, object]],
) -> dict[str, object] | None:
    """Find inventory metadata for one BirdNET prediction row."""
    input_path = prediction_row.get('input')
    if input_path is not None:
        resolved_input_path = str(Path(str(input_path)).resolve())
        metadata = inventory_by_path.get(resolved_input_path)
        if metadata is not None:
            return metadata

        metadata = inventory_by_file_name.get(Path(str(input_path)).name)
        if metadata is not None:
            return metadata

    return None


def _detection_record_to_row(
    detection_record: BirdDetectionRecord,
) -> dict[str, object]:
    """Convert a BirdDetectionRecord to a Polars-ready row."""
    row = detection_record.model_dump(mode='json')
    row['file_path'] = str(detection_record.file_path)
    row['detection_timestamp'] = (
        detection_record.detection_timestamp.isoformat()
        if detection_record.detection_timestamp is not None
        else None
    )
    row['source_result_path'] = (
        str(detection_record.source_result_path)
        if detection_record.source_result_path is not None
        else None
    )
    return row


def normalise_birdnet_predictions(
    predictions: Any,
    inventory_df: pl.DataFrame,
    model_name: str | None = None,
) -> pl.DataFrame:
    """Normalize BirdNET package predictions into the project schema."""
    predictions_df = _prediction_rows_to_polars(predictions)
    logger.debug('Normalizing %d BirdNET prediction rows', predictions_df.height)
    if predictions_df.is_empty():
        return pl.DataFrame(schema=BIRDNET_DETECTION_SCHEMA)

    inventory_by_path = _inventory_metadata_by_path(inventory_df)
    inventory_by_file_name = _inventory_metadata_by_file_name(inventory_df)
    rows: list[dict[str, object]] = []

    for prediction_row in predictions_df.iter_rows(named=True):
        metadata = _metadata_for_prediction(
            prediction_row=prediction_row,
            inventory_by_path=inventory_by_path,
            inventory_by_file_name=inventory_by_file_name,
        )
        if metadata is None:
            logger.debug(
                'Skipping prediction with no inventory match: %s',
                prediction_row,
            )
            continue

        confidence = float(prediction_row['confidence'])
        species_name = str(prediction_row['species_name'])
        scientific_name, common_name = _split_species_name(species_name)
        detection_start_s = _time_to_seconds(prediction_row['start_time'])
        detection_end_s = _time_to_seconds(prediction_row['end_time'])

        detection_record = BirdDetectionRecord(
            file_path=Path(str(metadata['file_path'])),
            file_name=str(metadata['file_name']),
            detection_start_s=detection_start_s,
            detection_end_s=detection_end_s,
            detection_duration_s=detection_end_s - detection_start_s,
            timestamp=metadata.get('timestamp'),
            latitude=metadata.get('latitude'),
            longitude=metadata.get('longitude'),
            temperature_int_c=metadata.get('temperature_int_c'),
            scientific_name=scientific_name,
            common_name=common_name,
            confidence=confidence,
            analysis_backend=BIRDNET_BACKEND,
            model_name=model_name,
        )
        rows.append(_detection_record_to_row(detection_record))

    if not rows:
        logger.info('No BirdNET predictions matched inventory records')
        return pl.DataFrame(schema=BIRDNET_DETECTION_SCHEMA)

    detections_df = pl.DataFrame(rows, schema=BIRDNET_DETECTION_SCHEMA)
    logger.info('Normalized %d BirdNET detections', detections_df.height)
    return detections_df


def write_birdnet_detection_outputs(
    detections_df: pl.DataFrame,
    output_dir: Path,
    stem: str = BIRDNET_DETECTIONS_STEM,
) -> tuple[Path, Path]:
    """Write normalized BirdNET detections to Parquet and CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = output_dir / f'{stem}.parquet'
    csv_path = output_dir / f'{stem}.csv'

    logger.info(
        'Writing BirdNET detections to %s and %s',
        parquet_path,
        csv_path,
    )
    detections_df.write_parquet(parquet_path)
    detections_df.write_csv(csv_path)
    logger.info('Wrote BirdNET detection outputs with %d rows', detections_df.height)

    return parquet_path, csv_path


def _predict_audio_file(
    model: Any,
    audio_path: Path,
    config: PipelineConfig,
) -> pl.DataFrame:
    """Predict BirdNET species for one audio file."""
    birdnet_config = config.birdnet
    logger.debug('Running BirdNET prediction for %s', audio_path)

    try:
        predictions = model.predict(
            str(audio_path),
            min_confidence=birdnet_config.min_confidence,
            batch_size=birdnet_config.batch_size,
            chunk_overlap_s=birdnet_config.overlap_s,
            bandpass_fmin=birdnet_config.fmin_hz,
            bandpass_fmax=birdnet_config.fmax_hz,
            sigmoid_sensitivity=birdnet_config.sensitivity,
        )
    except TypeError:
        predictions = model.predict(str(audio_path))

    predictions_df = _prediction_rows_to_polars(predictions)
    if predictions_df.is_empty() or 'confidence' not in predictions_df.columns:
        logger.debug('BirdNET returned no confidence-scored rows for %s', audio_path)
        return predictions_df

    filtered_df = predictions_df.filter(
        pl.col('confidence') >= birdnet_config.min_confidence
    )
    logger.debug(
        'BirdNET prediction for %s produced %d rows, %d above threshold',
        audio_path,
        predictions_df.height,
        filtered_df.height,
    )
    return filtered_df


def run_birdnet_predictions(
    model: Any,
    inventory_df: pl.DataFrame,
    config: PipelineConfig,
) -> pl.DataFrame:
    """Run BirdNET predictions for each readable inventory file."""
    prediction_dfs: list[pl.DataFrame] = []
    readable_count = inventory_df.filter(pl.col('readable_wav')).height
    logger.info('Running BirdNET predictions for %d readable files', readable_count)

    for index, inventory_row in enumerate(inventory_df.iter_rows(named=True), start=1):
        if inventory_row.get('readable_wav') is False:
            logger.debug('Skipping unreadable file %s', inventory_row.get('file_path'))
            continue

        audio_path = Path(str(inventory_row['file_path']))
        logger.info(
            'Running BirdNET prediction %d/%d for %s',
            index,
            inventory_df.height,
            audio_path.name,
        )
        prediction_df = _predict_audio_file(
            model=model,
            audio_path=audio_path,
            config=config,
        )
        if not prediction_df.is_empty():
            prediction_dfs.append(prediction_df)

    if not prediction_dfs:
        logger.info('BirdNET produced no predictions above threshold')
        return pl.DataFrame()

    predictions_df = pl.concat(prediction_dfs, how='diagonal')
    logger.info('BirdNET produced %d prediction rows', predictions_df.height)
    return predictions_df


def run_birdnet_analysis(
    config: PipelineConfig,
    inventory_df: pl.DataFrame,
) -> pl.DataFrame:
    """Run BirdNET and return normalized detection records."""
    output_dir = get_birdnet_output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info('Starting BirdNET analysis')

    model = load_birdnet_model(config)
    predictions_df = run_birdnet_predictions(
        model=model,
        inventory_df=inventory_df,
        config=config,
    )
    detections_df = normalise_birdnet_predictions(
        predictions=predictions_df,
        inventory_df=inventory_df,
        model_name=(
            f'acoustic-{config.birdnet.model_version}-'
            f'{config.birdnet.model_backend}'
        ),
    )

    write_birdnet_detection_outputs(
        detections_df=detections_df,
        output_dir=output_dir,
    )
    logger.info('Finished BirdNET analysis')
    return detections_df
