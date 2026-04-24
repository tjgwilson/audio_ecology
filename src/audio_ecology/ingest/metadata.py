"""Metadata extraction and record building for AudioMoth WAV files."""

from __future__ import annotations

from datetime import datetime, timezone
import logging
from pathlib import Path
import wave

from audio_ecology.config import PipelineConfig
from audio_ecology.constants import (
    AUDIOMOTH_FILENAME_TIMESTAMP_FORMAT,
    GUANO_LOCATION_FIELD,
    GUANO_TEMPERATURE_INT_FIELD,
    GUANO_TIMESTAMP_FIELD,
    GUANO_TIMESTAMP_FORMAT,
    LOCATION_SOURCE_DEPLOYMENT_CONFIG,
    LOCATION_SOURCE_DEVICE_CONFIG,
    LOCATION_SOURCE_GUANO,
    LOCATION_SOURCE_MISSING,
    LOCATION_SOURCE_SITE_CONFIG,
    TIMESTAMP_SOURCE_FILENAME,
    TIMESTAMP_SOURCE_GUANO,
    TIMESTAMP_SOURCE_MISSING,
)
from audio_ecology.models import AudioFileRecord

logger = logging.getLogger(__name__)


def resolve_deployment(
    device_id: str | None,
    config: PipelineConfig,
) -> tuple[str | None, object | None]:
    """Resolve the active deployment for a device, if configured.

    :param device_id: Parsed device ID.
    :param config: Pipeline configuration.
    :return: Deployment ID and config object, or ``(None, None)``.
    """
    if device_id is None:
        return None, None

    for deployment_id, deployment_config in config.deployments.items():
        if deployment_config.device_id == device_id:
            return deployment_id, deployment_config

    return None, None


def extract_device_id(file_name: str) -> str | None:
    """Extract the device ID from an AudioMoth file name.

    :param file_name: Name of the file.
    :return: Device ID, or None if parsing fails.
    """
    parts = file_name.split('_', maxsplit=1)
    if len(parts) < 2:
        return None
    return parts[0]


def extract_filename_timestamp(file_name: str) -> datetime | None:
    """Extract the timestamp from an AudioMoth file name.

    :param file_name: Name of the file.
    :return: Parsed timestamp, or None if parsing fails.
    """
    stem = Path(file_name).stem
    parts = stem.split('_')

    if len(parts) < 3:
        return None

    timestamp_str = f'{parts[1]}_{parts[2]}'

    try:
        return datetime.strptime(
            timestamp_str,
            AUDIOMOTH_FILENAME_TIMESTAMP_FORMAT,
        )
    except ValueError:
        return None


def extract_guano_fields(file_path: Path) -> dict[str, str]:
    """Extract GUANO key value pairs from a WAV file.

    :param file_path: Path to the WAV file.
    :return: Dictionary of GUANO fields.
    """
    raw_bytes = file_path.read_bytes()
    marker = b'GUANO|Version'
    start_index = raw_bytes.find(marker)

    if start_index == -1:
        return {}

    guano_text = raw_bytes[start_index:].decode('utf-8', errors='ignore')
    fields: dict[str, str] = {}

    for line in guano_text.splitlines():
        if ':' not in line:
            continue
        key, value = line.split(':', maxsplit=1)
        fields[key.strip()] = value.strip()

    return fields


def extract_guano_timestamp(guano_fields: dict[str, str]) -> datetime | None:
    """Extract the GUANO timestamp.

    :param guano_fields: Parsed GUANO fields.
    :return: Parsed UTC datetime, or None.
    """
    timestamp_str = guano_fields.get(GUANO_TIMESTAMP_FIELD)
    if timestamp_str is None:
        return None

    try:
        return datetime.strptime(
            timestamp_str,
            GUANO_TIMESTAMP_FORMAT,
        ).replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def extract_guano_location(
    guano_fields: dict[str, str],
) -> tuple[float | None, float | None]:
    """Extract latitude and longitude from the GUANO location field.

    :param guano_fields: Parsed GUANO fields.
    :return: Latitude and longitude.
    """
    loc_position = guano_fields.get(GUANO_LOCATION_FIELD)
    if loc_position is None:
        return None, None

    parts = loc_position.split()
    if len(parts) != 2:
        return None, None

    try:
        latitude = float(parts[0])
        longitude = float(parts[1])
    except ValueError:
        return None, None

    return latitude, longitude


def extract_temperature_int(
    guano_fields: dict[str, str],
) -> float | None:
    """Extract the internal temperature from GUANO metadata.

    :param guano_fields: Parsed GUANO fields.
    :return: Internal temperature in Celsius, or None.
    """
    temperature_str = guano_fields.get(GUANO_TEMPERATURE_INT_FIELD)
    if temperature_str is None:
        return None

    try:
        return float(temperature_str)
    except ValueError:
        return None


def extract_wav_properties(
    file_path: Path,
) -> tuple[int | None, float | None, bool, str | None]:
    """Extract basic WAV properties and perform a simple readability check.

    :param file_path: Path to the WAV file.
    :return: Sample rate, duration, readable flag, and optional note.
    """
    try:
        with wave.open(str(file_path), 'rb') as wav_handle:
            sample_rate_hz = wav_handle.getframerate()
            frame_count = wav_handle.getnframes()
            duration_s = (
                frame_count / sample_rate_hz if sample_rate_hz > 0 else None
            )

            _ = wav_handle.readframes(min(frame_count, 1024))

        return sample_rate_hz, duration_s, True, None
    except (wave.Error, OSError) as exc:
        return None, None, False, f'WAV read failed: {exc}'


def resolve_location(
    device_id: str | None,
    guano_latitude: float | None,
    guano_longitude: float | None,
    config: PipelineConfig,
) -> tuple[
    float | None,
    float | None,
    str,
    str | None,
    str | None,
    str | None,
    list[str],
]:
    """Resolve final location using GUANO first, then config fallbacks.

    :param device_id: Parsed device ID.
    :param guano_latitude: Latitude from GUANO.
    :param guano_longitude: Longitude from GUANO.
    :param config: Pipeline configuration.
    :return: Latitude, longitude, source, device label, deployment ID, habitat
        label, and deployment detection targets.
    """
    device_label: str | None = None
    deployment_id, deployment_config = resolve_deployment(
        device_id=device_id,
        config=config,
    )
    habitat_label = (
        deployment_config.habitat_label if deployment_config is not None else None
    )
    detection_targets = (
        list(deployment_config.detection_targets)
        if deployment_config is not None
        else []
    )

    if device_id is not None and device_id in config.devices:
        device_label = config.devices[device_id].label

    if guano_latitude is not None and guano_longitude is not None:
        return (
            guano_latitude,
            guano_longitude,
            LOCATION_SOURCE_GUANO,
            device_label,
            deployment_id,
            habitat_label,
            detection_targets,
        )

    if (
        deployment_config is not None
        and deployment_config.fallback_location is not None
        and deployment_config.fallback_location.latitude is not None
        and deployment_config.fallback_location.longitude is not None
    ):
        return (
            deployment_config.fallback_location.latitude,
            deployment_config.fallback_location.longitude,
            LOCATION_SOURCE_DEPLOYMENT_CONFIG,
            device_label,
            deployment_id,
            habitat_label,
            detection_targets,
        )

    if device_id is not None and device_id in config.devices:
        device_config = config.devices[device_id]
        device_label = device_config.label

        if (
            device_config.fallback_location is not None
            and device_config.fallback_location.latitude is not None
            and device_config.fallback_location.longitude is not None
        ):
            return (
                device_config.fallback_location.latitude,
                device_config.fallback_location.longitude,
                LOCATION_SOURCE_DEVICE_CONFIG,
                device_label,
                deployment_id,
                habitat_label,
                detection_targets,
            )

    if (
        config.fallback_location is not None
        and config.fallback_location.latitude is not None
        and config.fallback_location.longitude is not None
    ):
        return (
            config.fallback_location.latitude,
            config.fallback_location.longitude,
            LOCATION_SOURCE_SITE_CONFIG,
            device_label,
            deployment_id,
            habitat_label,
            detection_targets,
        )

    return (
        None,
        None,
        LOCATION_SOURCE_MISSING,
        device_label,
        deployment_id,
        habitat_label,
        detection_targets,
    )


def build_audio_file_record(
    file_path: Path,
    config: PipelineConfig,
) -> AudioFileRecord:
    """Build an inventory record for one WAV file.

    :param file_path: Path to the WAV file.
    :param config: Pipeline configuration.
    :return: Inventory record.
    """
    logger.debug('Building audio file record for %s', file_path)
    file_name = file_path.name
    device_id = extract_device_id(file_name)
    filename_timestamp = extract_filename_timestamp(file_name)

    sample_rate_hz, duration_s, readable_wav, wav_note = extract_wav_properties(
        file_path
    )

    guano_fields = extract_guano_fields(file_path)
    guano_present = bool(guano_fields)

    guano_timestamp = extract_guano_timestamp(guano_fields)
    guano_latitude, guano_longitude = extract_guano_location(guano_fields)
    temperature_int_c = extract_temperature_int(guano_fields)

    if guano_timestamp is not None:
        timestamp = guano_timestamp
        timestamp_source = TIMESTAMP_SOURCE_GUANO
    elif filename_timestamp is not None:
        timestamp = filename_timestamp
        timestamp_source = TIMESTAMP_SOURCE_FILENAME
    else:
        timestamp = None
        timestamp_source = TIMESTAMP_SOURCE_MISSING

    (
        latitude,
        longitude,
        location_source,
        device_label,
        deployment_id,
        habitat_label,
        detection_targets,
    ) = resolve_location(
        device_id=device_id,
        guano_latitude=guano_latitude,
        guano_longitude=guano_longitude,
        config=config,
    )

    notes: list[str] = []
    if wav_note is not None:
        notes.append(wav_note)
        logger.warning('WAV metadata read failed for %s: %s', file_path, wav_note)

    note_text = '; '.join(notes) if notes else None

    record = AudioFileRecord(
        file_path=file_path,
        file_name=file_name,
        device_id=device_id,
        device_label=device_label,
        deployment_id=deployment_id,
        habitat_label=habitat_label,
        detection_targets=detection_targets,
        timestamp=timestamp,
        timestamp_source=timestamp_source,
        filename_timestamp=filename_timestamp,
        sample_rate_hz=sample_rate_hz,
        duration_s=duration_s,
        file_size_bytes=file_path.stat().st_size,
        latitude=latitude,
        longitude=longitude,
        location_source=location_source,
        temperature_int_c=temperature_int_c,
        guano_present=guano_present,
        readable_wav=readable_wav,
        notes=note_text,
    )
    logger.debug(
        'Built record for %s: readable=%s sample_rate=%s duration=%s '
        'timestamp_source=%s location_source=%s guano=%s',
        file_name,
        readable_wav,
        sample_rate_hz,
        duration_s,
        timestamp_source,
        location_source,
        guano_present,
    )
    return record
