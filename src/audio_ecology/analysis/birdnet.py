"""BirdNET Python package integration utilities."""

from __future__ import annotations

from datetime import datetime, timedelta
import importlib
import importlib.util
import logging
from pathlib import Path
from typing import Any

import polars as pl

from audio_ecology.analysis.checkpointing import AnalysisCheckpointStore
from audio_ecology.analysis.storage import (
    DETECTIONS_STEM,
    backend_partition_name,
    get_detection_dataset_dir,
    load_detection_dataframe,
    write_detection_dataset,
)
from audio_ecology.config import PipelineConfig
from audio_ecology.models import BirdDetectionRecord
from audio_ecology.solar import calculate_solar_metadata

logger = logging.getLogger(__name__)

BIRDNET_BACKEND = 'birdnet'
BIRDNET_DETECTIONS_STEM = 'birdnet_detections'
BIRDNET_LOCATION_SPECIES_STEM = 'birdnet_location_species'

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
    'sunrise_timestamp': pl.Utf8,
    'sunset_timestamp': pl.Utf8,
    'minutes_from_sunrise': pl.Float64,
    'minutes_to_sunset': pl.Float64,
    'is_daylight': pl.Boolean,
    'temperature_int_c': pl.Float64,
    'deployment_id': pl.Utf8,
    'habitat_label': pl.Utf8,
    'detection_targets': pl.List(pl.Utf8),
    'scientific_name': pl.Utf8,
    'common_name': pl.Utf8,
    'confidence': pl.Float64,
    'analysis_backend': pl.Utf8,
    'model_name': pl.Utf8,
    'source_result_path': pl.Utf8,
}

BIRDNET_LOCATION_SPECIES_SCHEMA = {
    'latitude': pl.Float64,
    'longitude': pl.Float64,
    'birdnet_week': pl.Int64,
    'species_name': pl.Utf8,
    'scientific_name': pl.Utf8,
    'common_name': pl.Utf8,
    'location_confidence': pl.Float64,
}

LocationSpeciesCache = dict[
    tuple[float, float, int | None],
    list[dict[str, object]],
]


def get_birdnet_output_dir(config: PipelineConfig) -> Path:
    """Return the directory used for BirdNET-specific auxiliary outputs.

    :param config: Loaded pipeline configuration.
    :return: BirdNET auxiliary output directory.
    """
    return config.output_dir / 'birdnet'


def get_birdnet_detection_dataset_dir(
    config: PipelineConfig,
) -> Path:
    """Return the canonical shared detection dataset directory for BirdNET.

    :param config: Loaded pipeline configuration.
    :return: Shared BirdNET detection dataset directory.
    """
    return get_detection_dataset_dir(
        output_dir=config.output_dir,
        analysis_backend=BIRDNET_BACKEND,
    )


def get_birdnet_checkpoint_store(
    config: PipelineConfig,
) -> AnalysisCheckpointStore:
    """Return the checkpoint store used for BirdNET detections.

    :param config: Loaded pipeline configuration.
    :return: Configured BirdNET checkpoint store.
    """
    return AnalysisCheckpointStore(
        output_dir=config.output_dir,
        backend_name=backend_partition_name(BIRDNET_BACKEND),
        schema=BIRDNET_DETECTION_SCHEMA,
    )


def birdnet_week_from_timestamp(timestamp: datetime | None) -> int | None:
    """Map a timestamp to BirdNET's 1-48 four-weeks-per-month index.

    :param timestamp: Timestamp to map.
    :return: BirdNET week index or ``None`` when timestamp is missing.
    """
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


def load_birdnet_geo_model(config: PipelineConfig) -> Any:
    """Load the configured BirdNET species range model."""
    logger.info(
        'Loading BirdNET geo model geo-%s-%s',
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
        'geo',
        config.birdnet.model_version,
        config.birdnet.model_backend,
    )
    logger.info('Loaded BirdNET geo model')
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


def _species_rows_from_geo_predictions(predictions: Any) -> list[dict[str, object]]:
    """Return BirdNET species rows from a geo prediction result."""
    if predictions is None:
        return []

    if hasattr(predictions, 'to_dataframe'):
        predictions = predictions.to_dataframe()

    predictions_df = _prediction_rows_to_polars(predictions)
    if not predictions_df.is_empty() and 'species_name' in predictions_df.columns:
        rows: list[dict[str, object]] = []
        for prediction_row in predictions_df.iter_rows(named=True):
            species_name = str(prediction_row['species_name'])
            scientific_name, common_name = _split_species_name(species_name)
            confidence = prediction_row.get('confidence')
            rows.append(
                {
                    'species_name': species_name,
                    'scientific_name': scientific_name,
                    'common_name': common_name,
                    'location_confidence': float(confidence)
                    if confidence is not None
                    else None,
                }
            )
        return sorted(rows, key=lambda row: str(row['species_name']))

    if hasattr(predictions, 'to_set'):
        rows = []
        for species_name_value in predictions.to_set():
            species_name = str(species_name_value)
            scientific_name, common_name = _split_species_name(species_name)
            rows.append(
                {
                    'species_name': species_name,
                    'scientific_name': scientific_name,
                    'common_name': common_name,
                    'location_confidence': None,
                }
            )
        return sorted(rows, key=lambda row: str(row['species_name']))

    return []


def _location_filter_key(
    inventory_row: dict[str, object],
) -> tuple[float, float, int | None] | None:
    """Return a cache key for BirdNET geo filtering, if location is available."""
    latitude = inventory_row.get('latitude')
    longitude = inventory_row.get('longitude')
    if latitude is None or longitude is None:
        return None

    timestamp = inventory_row.get('timestamp')
    week = birdnet_week_from_timestamp(
        timestamp if isinstance(timestamp, datetime) else None
    )
    return (float(latitude), float(longitude), week)


def _location_species_filter(
    geo_model: Any | None,
    inventory_row: dict[str, object],
    config: PipelineConfig,
    species_filter_cache: LocationSpeciesCache,
) -> list[str] | None:
    """Return a BirdNET custom species list for the inventory row location."""
    if geo_model is None:
        return None

    filter_key = _location_filter_key(inventory_row)
    if filter_key is None:
        logger.debug(
            'Skipping BirdNET location filter for %s with missing location',
            inventory_row.get('file_path'),
        )
        return None

    if filter_key not in species_filter_cache:
        latitude, longitude, week = filter_key
        logger.debug(
            'Predicting BirdNET location species for lat=%s lon=%s week=%s',
            latitude,
            longitude,
            week,
        )
        geo_predictions = geo_model.predict(
            latitude,
            longitude,
            week=week,
            min_confidence=config.birdnet.location_min_confidence,
        )
        species_filter_cache[filter_key] = _species_rows_from_geo_predictions(
            geo_predictions
        )
        logger.debug(
            'BirdNET location filter for lat=%s lon=%s week=%s produced %d species',
            latitude,
            longitude,
            week,
            len(species_filter_cache[filter_key]),
        )

    return [
        str(species_row['species_name'])
        for species_row in species_filter_cache[filter_key]
    ]


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
        detection_timestamp = None
        timestamp_value = metadata.get('timestamp')
        if isinstance(timestamp_value, datetime):
            detection_timestamp = timestamp_value + timedelta(
                seconds=detection_start_s
            )
        solar_metadata = calculate_solar_metadata(
            timestamp=detection_timestamp,
            latitude=metadata.get('latitude'),
            longitude=metadata.get('longitude'),
        )

        detection_record = BirdDetectionRecord(
            file_path=Path(str(metadata['file_path'])),
            file_name=str(metadata['file_name']),
            detection_start_s=detection_start_s,
            detection_end_s=detection_end_s,
            detection_duration_s=detection_end_s - detection_start_s,
            timestamp=metadata.get('timestamp'),
            latitude=metadata.get('latitude'),
            longitude=metadata.get('longitude'),
            sunrise_timestamp=solar_metadata.sunrise_timestamp,
            sunset_timestamp=solar_metadata.sunset_timestamp,
            minutes_from_sunrise=solar_metadata.minutes_from_sunrise,
            minutes_to_sunset=solar_metadata.minutes_to_sunset,
            is_daylight=solar_metadata.is_daylight,
            temperature_int_c=metadata.get('temperature_int_c'),
            deployment_id=metadata.get('deployment_id'),
            habitat_label=metadata.get('habitat_label'),
            detection_targets=metadata.get('detection_targets') or [],
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
    dataset_dir: Path,
    write_csv: bool = False,
) -> Path:
    """Write normalized BirdNET detections.

    :param detections_df: Normalized BirdNET detection rows.
    :param dataset_dir: Shared BirdNET detection dataset directory.
    :param write_csv: Whether to also write CSV copies inside each partition.
    :return: Detection dataset directory.
    """
    dataset_path = write_detection_dataset(
        detections_df=detections_df,
        dataset_dir=dataset_dir,
        stem=DETECTIONS_STEM,
        write_csv=write_csv,
    )
    logger.info('Wrote BirdNET detection outputs with %d rows', detections_df.height)

    return dataset_path


def _location_species_cache_to_dataframe(
    species_filter_cache: LocationSpeciesCache,
) -> pl.DataFrame:
    """Convert cached BirdNET geo species lists to a Polars DataFrame."""
    rows: list[dict[str, object]] = []
    for (
        latitude,
        longitude,
        birdnet_week,
    ), species_rows in species_filter_cache.items():
        for species_row in species_rows:
            rows.append(
                {
                    'latitude': latitude,
                    'longitude': longitude,
                    'birdnet_week': birdnet_week,
                    'species_name': species_row['species_name'],
                    'scientific_name': species_row['scientific_name'],
                    'common_name': species_row['common_name'],
                    'location_confidence': species_row['location_confidence'],
                }
            )

    if not rows:
        return pl.DataFrame(schema=BIRDNET_LOCATION_SPECIES_SCHEMA)

    return pl.DataFrame(rows, schema=BIRDNET_LOCATION_SPECIES_SCHEMA).sort(
        ['latitude', 'longitude', 'birdnet_week', 'species_name']
    )


def write_birdnet_location_species_output(
    species_filter_cache: LocationSpeciesCache,
    output_dir: Path,
    stem: str = BIRDNET_LOCATION_SPECIES_STEM,
) -> Path:
    """Write BirdNET location-filter species lists to CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f'{stem}.csv'
    species_df = _location_species_cache_to_dataframe(species_filter_cache)

    logger.info(
        'Writing BirdNET location species list to %s with %d rows',
        csv_path,
        species_df.height,
    )
    species_df.write_csv(csv_path)
    return csv_path


def _filter_predictions_to_species(
    predictions_df: pl.DataFrame,
    custom_species_list: list[str] | None,
) -> pl.DataFrame:
    """Restrict prediction rows to the configured BirdNET species names."""
    if (
        custom_species_list is None
        or predictions_df.is_empty()
        or 'species_name' not in predictions_df.columns
    ):
        return predictions_df

    return predictions_df.filter(pl.col('species_name').is_in(custom_species_list))


def _predict_audio_file(
    model: Any,
    audio_path: Path,
    config: PipelineConfig,
    custom_species_list: list[str] | None = None,
) -> pl.DataFrame:
    """Predict BirdNET species for one audio file."""
    birdnet_config = config.birdnet
    logger.debug('Running BirdNET prediction for %s', audio_path)

    predict_kwargs: dict[str, object] = {
        'default_confidence_threshold': birdnet_config.min_confidence,
        'batch_size': birdnet_config.batch_size,
        'overlap_duration_s': birdnet_config.overlap_s,
        'bandpass_fmin': birdnet_config.fmin_hz,
        'bandpass_fmax': birdnet_config.fmax_hz,
        'sigmoid_sensitivity': birdnet_config.sensitivity,
    }
    prediction_attempts: list[tuple[str, dict[str, object], bool]] = []

    if custom_species_list is not None:
        prediction_attempts.append(
            (
                'current BirdNET parameters with location species list',
                {**predict_kwargs, 'custom_species_list': custom_species_list},
                True,
            )
        )
    prediction_attempts.append(('current BirdNET parameters', predict_kwargs, False))

    legacy_kwargs: dict[str, object] = {
        'min_confidence': birdnet_config.min_confidence,
        'batch_size': birdnet_config.batch_size,
        'chunk_overlap_s': birdnet_config.overlap_s,
        'bandpass_fmin': birdnet_config.fmin_hz,
        'bandpass_fmax': birdnet_config.fmax_hz,
        'sigmoid_sensitivity': birdnet_config.sensitivity,
    }
    if custom_species_list is not None:
        prediction_attempts.append(
            (
                'legacy BirdNET parameters with location species list',
                {**legacy_kwargs, 'custom_species_list': custom_species_list},
                True,
            )
        )
    prediction_attempts.append(('legacy BirdNET parameters', legacy_kwargs, False))
    prediction_attempts.append(('bare BirdNET prediction', {}, False))

    predictions = None
    location_filter_applied_by_model = False
    for attempt_label, attempt_kwargs, applies_location_filter in prediction_attempts:
        try:
            predictions = model.predict(str(audio_path), **attempt_kwargs)
            location_filter_applied_by_model = applies_location_filter
            break
        except TypeError as error:
            logger.debug(
                'BirdNET prediction attempt failed for %s using %s: %s',
                audio_path,
                attempt_label,
                error,
            )
            continue

    if predictions is None:
        raise RuntimeError(f'BirdNET prediction failed for {audio_path}')

    predictions_df = _prediction_rows_to_polars(predictions)
    if predictions_df.is_empty() or 'confidence' not in predictions_df.columns:
        logger.debug('BirdNET returned no confidence-scored rows for %s', audio_path)
        return predictions_df

    if custom_species_list is not None and not location_filter_applied_by_model:
        before_location_filter_count = predictions_df.height
        predictions_df = _filter_predictions_to_species(
            predictions_df=predictions_df,
            custom_species_list=custom_species_list,
        )
        logger.info(
            'Applied BirdNET location species filter after prediction for %s: '
            '%d rows kept from %d',
            audio_path.name,
            predictions_df.height,
            before_location_filter_count,
        )

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
    geo_model: Any | None = None,
    location_species_output_dir: Path | None = None,
) -> pl.DataFrame:
    """Run BirdNET predictions for each readable inventory file."""
    prediction_dfs: list[pl.DataFrame] = []
    species_filter_cache: LocationSpeciesCache = {}
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
        custom_species_list = _location_species_filter(
            geo_model=geo_model,
            inventory_row=inventory_row,
            config=config,
            species_filter_cache=species_filter_cache,
        )
        prediction_df = _predict_audio_file(
            model=model,
            audio_path=audio_path,
            config=config,
            custom_species_list=custom_species_list,
        )
        if not prediction_df.is_empty():
            prediction_dfs.append(prediction_df)

    if location_species_output_dir is not None and geo_model is not None:
        write_birdnet_location_species_output(
            species_filter_cache=species_filter_cache,
            output_dir=location_species_output_dir,
        )

    if not prediction_dfs:
        logger.info('BirdNET produced no predictions above threshold')
        return pl.DataFrame()

    predictions_df = pl.concat(prediction_dfs, how='diagonal')
    logger.info('BirdNET produced %d prediction rows', predictions_df.height)
    return predictions_df


def run_birdnet_analysis(
    config: PipelineConfig,
    inventory_df: pl.DataFrame,
    overwrite_checkpoints: bool = False,
) -> pl.DataFrame:
    """Run BirdNET and return normalized detection records.

    :param config: Loaded pipeline configuration.
    :param inventory_df: Inventory rows to analyse.
    :param overwrite_checkpoints: Whether to ignore existing file checkpoints.
    :return: Normalized BirdNET detection rows.
    """
    output_dir = get_birdnet_output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    detection_dataset_dir = get_birdnet_detection_dataset_dir(config)
    detection_dataset_dir.mkdir(parents=True, exist_ok=True)
    logger.info('Starting BirdNET analysis')

    checkpoint_store = get_birdnet_checkpoint_store(config)
    model: Any | None = None
    geo_model = (
        load_birdnet_geo_model(config) if config.birdnet.use_location_filter else None
    )
    species_filter_cache: LocationSpeciesCache = {}
    model_name = (
        f'acoustic-{config.birdnet.model_version}-'
        f'{config.birdnet.model_backend}'
    )
    detection_dfs: list[pl.DataFrame] = []

    readable_count = inventory_df.filter(pl.col('readable_wav')).height
    logger.info(
        'Running BirdNET analysis for %d readable files with checkpoints in %s',
        readable_count,
        checkpoint_store.checkpoint_dir,
    )

    for index, inventory_row in enumerate(inventory_df.iter_rows(named=True), start=1):
        if inventory_row.get('readable_wav') is False:
            logger.debug('Skipping unreadable file %s', inventory_row.get('file_path'))
            continue

        audio_path = Path(str(inventory_row['file_path']))
        if checkpoint_store.exists(audio_path) and not overwrite_checkpoints:
            detection_dfs.append(checkpoint_store.read(audio_path))
            logger.info(
                'Skipping BirdNET prediction %d/%d for %s; checkpoint exists',
                index,
                inventory_df.height,
                audio_path.name,
            )
            continue

        logger.info(
            'Running BirdNET prediction %d/%d for %s',
            index,
            inventory_df.height,
            audio_path.name,
        )
        if model is None:
            model = load_birdnet_model(config)

        custom_species_list = _location_species_filter(
            geo_model=geo_model,
            inventory_row=inventory_row,
            config=config,
            species_filter_cache=species_filter_cache,
        )
        predictions_df = _predict_audio_file(
            model=model,
            audio_path=audio_path,
            config=config,
            custom_species_list=custom_species_list,
        )
        detection_df = normalise_birdnet_predictions(
            predictions=predictions_df,
            inventory_df=pl.DataFrame([inventory_row]),
            model_name=model_name,
        )
        checkpoint_store.write(audio_path, detection_df)
        detection_dfs.append(detection_df)

    if geo_model is not None and species_filter_cache:
        write_birdnet_location_species_output(
            species_filter_cache=species_filter_cache,
            output_dir=output_dir,
        )

    detections_df = _combine_detection_dfs(detection_dfs)

    write_birdnet_detection_outputs(
        detections_df=detections_df,
        dataset_dir=detection_dataset_dir,
        write_csv=config.outputs.write_csv,
    )
    logger.info('Finished BirdNET analysis')
    return detections_df


def _combine_detection_dfs(detection_dfs: list[pl.DataFrame]) -> pl.DataFrame:
    """Combine per-file detection DataFrames."""
    non_empty_dfs = [
        detection_df
        for detection_df in detection_dfs
        if not detection_df.is_empty()
    ]
    if not non_empty_dfs:
        return pl.DataFrame(schema=BIRDNET_DETECTION_SCHEMA)

    return pl.concat(non_empty_dfs, how='diagonal_relaxed')
