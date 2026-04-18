from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from audio_ecology.config import (
    DeviceConfig,
    LocationConfig,
    PipelineConfig,
)
from audio_ecology.constants import (
    LOCATION_SOURCE_DEVICE_CONFIG,
    LOCATION_SOURCE_GUANO,
    LOCATION_SOURCE_SITE_CONFIG,
    TIMESTAMP_SOURCE_FILENAME,
    TIMESTAMP_SOURCE_GUANO,
)
from audio_ecology.ingest.metadata import (
    build_audio_file_record,
    extract_device_id,
    extract_filename_timestamp,
)
from conftest import create_test_wav


def make_config(tmp_path: Path) -> PipelineConfig:
    """Create a simple pipeline config for tests."""
    return PipelineConfig(
        project_root=tmp_path,
        input_dir=tmp_path / 'raw',
        output_dir=tmp_path / 'processed',
        site_name='Test Site',
        fallback_location=LocationConfig(
            latitude=51.5,
            longitude=-2.1,
        ),
        devices={
            '24F319046907737B': DeviceConfig(
                label='am_1',
                fallback_location=LocationConfig(
                    latitude=50.1,
                    longitude=-3.2,
                ),
            )
        },
        analyses=['inventory'],
    )


def test_extract_device_id() -> None:
    file_name = '24F319046907737B_20260417_223541.WAV'
    assert extract_device_id(file_name) == '24F319046907737B'


def test_extract_filename_timestamp() -> None:
    file_name = '24F319046907737B_20260417_223541.WAV'
    expected = datetime(2026, 4, 17, 22, 35, 41)

    assert extract_filename_timestamp(file_name) == expected


def test_build_audio_file_record_uses_guano_timestamp_and_location(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    wav_path = create_test_wav(
        tmp_path / 'raw' / '24F319046907737B_20260417_223541.WAV',
        sample_rate_hz=16000,
        duration_s=2.0,
        guano_fields={
            'Timestamp': '2026-04-16T21:43:00Z',
            'Loc Position': '50.432584 -3.672039',
            'Temperature Int': '18.5',
        },
    )

    record = build_audio_file_record(wav_path, config)

    assert record.device_id == '24F319046907737B'
    assert record.device_label == 'am_1'
    assert record.timestamp == datetime(
        2026, 4, 16, 21, 43, 0, tzinfo=timezone.utc
    )
    assert record.timestamp_source == TIMESTAMP_SOURCE_GUANO
    assert record.filename_timestamp == datetime(2026, 4, 17, 22, 35, 41)
    assert record.latitude == 50.432584
    assert record.longitude == -3.672039
    assert record.location_source == LOCATION_SOURCE_GUANO
    assert record.temperature_int_c == 18.5
    assert record.sample_rate_hz == 16000
    assert record.duration_s == 2.0
    assert record.guano_present is True
    assert record.readable_wav is True
    assert record.notes is None


def test_build_audio_file_record_falls_back_to_filename_timestamp_and_device_location(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    wav_path = create_test_wav(
        tmp_path / 'raw' / '24F319046907737B_20260417_223541.WAV',
        guano_fields={},
    )

    record = build_audio_file_record(wav_path, config)

    assert record.timestamp == datetime(2026, 4, 17, 22, 35, 41)
    assert record.timestamp_source == TIMESTAMP_SOURCE_FILENAME
    assert record.latitude == 50.1
    assert record.longitude == -3.2
    assert record.location_source == LOCATION_SOURCE_DEVICE_CONFIG
    assert record.temperature_int_c is None


def test_build_audio_file_record_falls_back_to_site_location(
    tmp_path: Path,
) -> None:
    config = PipelineConfig(
        project_root=tmp_path,
        input_dir=tmp_path / 'raw',
        output_dir=tmp_path / 'processed',
        site_name='Test Site',
        fallback_location=LocationConfig(
            latitude=51.5,
            longitude=-2.1,
        ),
        devices={},
        analyses=['inventory'],
    )

    wav_path = create_test_wav(
        tmp_path / 'raw' / 'UNKNOWN_20260417_223541.WAV',
        guano_fields={},
    )

    record = build_audio_file_record(wav_path, config)

    assert record.device_id == 'UNKNOWN'
    assert record.device_label is None
    assert record.latitude == 51.5
    assert record.longitude == -2.1
    assert record.location_source == LOCATION_SOURCE_SITE_CONFIG


def test_build_audio_file_record_handles_missing_guano_and_bad_filename(
    tmp_path: Path,
) -> None:
    config = PipelineConfig(
        project_root=tmp_path,
        input_dir=tmp_path / 'raw',
        output_dir=tmp_path / 'processed',
        site_name='Test Site',
        devices={},
        analyses=[],
    )

    wav_path = create_test_wav(tmp_path / 'raw' / 'not_a_standard_name.WAV')

    record = build_audio_file_record(wav_path, config)

    assert record.device_id is not None
    assert record.timestamp is None
    assert record.latitude is None
    assert record.longitude is None
    assert record.temperature_int_c is None
    assert record.location_source == 'missing'


def test_build_audio_file_record_marks_unreadable_wav(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    wav_path = tmp_path / 'raw' / '24F319046907737B_20260417_223541.WAV'
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    wav_path.write_bytes(b'not really a wav')

    record = build_audio_file_record(wav_path, config)

    assert record.readable_wav is False
    assert record.sample_rate_hz is None
    assert record.duration_s is None
    assert record.notes is not None
    assert 'WAV read failed' in record.notes