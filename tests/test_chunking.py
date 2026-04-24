from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import wave

from audio_ecology.config import ChunkingConfig
from audio_ecology.ingest.chunking import (
    build_chunk_file_name,
    build_chunk_records,
    build_chunk_records_for_file,
    get_chunk_output_dir,
    should_chunk_file,
    write_chunk_wav,
    write_chunk_wavs,
)
from audio_ecology.models import AudioChunkRecord, AudioFileRecord


def create_pcm_wav(
    file_path: Path,
    sample_rate_hz: int = 8000,
    duration_s: float = 1.0,
) -> Path:
    """Create a small PCM WAV file for chunking tests.

    :param file_path: Output WAV path.
    :param sample_rate_hz: Sample rate in Hz.
    :param duration_s: Duration in seconds.
    :return: Path to the created WAV file.
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)

    frame_count = int(sample_rate_hz * duration_s)

    with wave.open(str(file_path), 'wb') as wav_handle:
        wav_handle.setnchannels(1)
        wav_handle.setsampwidth(2)
        wav_handle.setframerate(sample_rate_hz)
        wav_handle.writeframes(b'\x00\x00' * frame_count)

    return file_path


def make_audio_file_record(
    tmp_path: Path,
    *,
    duration_s: float = 10.0,
    sample_rate_hz: int = 8000,
    readable_wav: bool = True,
) -> AudioFileRecord:
    """Create a simple AudioFileRecord backed by a real WAV file.

    :param tmp_path: Temporary directory.
    :param duration_s: Duration in seconds.
    :param sample_rate_hz: Sample rate in Hz.
    :param readable_wav: Whether the record should be marked readable.
    :return: Audio file record.
    """
    wav_path = create_pcm_wav(
        tmp_path / 'raw' / '24F319046907737B_20260417_223541.WAV',
        sample_rate_hz=sample_rate_hz,
        duration_s=duration_s,
    )

    return AudioFileRecord(
        file_path=wav_path,
        file_name=wav_path.name,
        device_id='24F319046907737B',
        device_label='am_1',
        deployment_id='wyke_woods_spring_2026',
        habitat_label='mixed_woodland',
        timestamp=datetime(2026, 4, 17, 22, 35, 41, tzinfo=timezone.utc),
        sample_rate_hz=sample_rate_hz,
        duration_s=duration_s,
        latitude=50.432584,
        longitude=-3.672039,
        readable_wav=readable_wav,
    )


def test_should_chunk_file_true_for_valid_record(tmp_path: Path) -> None:
    record = make_audio_file_record(tmp_path)

    assert should_chunk_file(record) is True


def test_should_chunk_file_false_for_unreadable_record(tmp_path: Path) -> None:
    record = make_audio_file_record(tmp_path, readable_wav=False)

    assert should_chunk_file(record) is False


def test_should_chunk_file_false_for_missing_duration(tmp_path: Path) -> None:
    record = make_audio_file_record(tmp_path)
    record.duration_s = None

    assert should_chunk_file(record) is False


def test_build_chunk_records_for_file_without_overlap(tmp_path: Path) -> None:
    record = make_audio_file_record(tmp_path, duration_s=10.0)

    chunking_config = ChunkingConfig(
        enabled=True,
        duration_s=3.0,
        overlap_s=0.0,
    )

    chunk_records = build_chunk_records_for_file(
        record=record,
        chunking_config=chunking_config,
        analysis_targets=['bird', 'bat'],
    )

    assert len(chunk_records) == 4

    assert chunk_records[0].chunk_index == 0
    assert chunk_records[0].chunk_start_s == 0.0
    assert chunk_records[0].chunk_end_s == 3.0
    assert chunk_records[0].chunk_duration_s == 3.0
    assert chunk_records[0].analysis_targets == ['bird', 'bat']
    assert chunk_records[0].deployment_id == 'wyke_woods_spring_2026'
    assert chunk_records[0].habitat_label == 'mixed_woodland'

    assert chunk_records[1].chunk_start_s == 3.0
    assert chunk_records[1].chunk_end_s == 6.0

    assert chunk_records[2].chunk_start_s == 6.0
    assert chunk_records[2].chunk_end_s == 9.0

    assert chunk_records[3].chunk_start_s == 9.0
    assert chunk_records[3].chunk_end_s == 10.0
    assert chunk_records[3].chunk_duration_s == 1.0


def test_build_chunk_records_for_file_with_overlap(tmp_path: Path) -> None:
    record = make_audio_file_record(tmp_path, duration_s=10.0)

    chunking_config = ChunkingConfig(
        enabled=True,
        duration_s=3.0,
        overlap_s=1.0,
    )

    chunk_records = build_chunk_records_for_file(
        record=record,
        chunking_config=chunking_config,
    )

    assert len(chunk_records) == 5

    expected_windows = [
        (0.0, 3.0),
        (2.0, 5.0),
        (4.0, 7.0),
        (6.0, 9.0),
        (8.0, 10.0),
    ]

    actual_windows = [
        (record.chunk_start_s, record.chunk_end_s)
        for record in chunk_records
    ]

    assert actual_windows == expected_windows


def test_build_chunk_records_for_file_returns_empty_when_disabled(
    tmp_path: Path,
) -> None:
    record = make_audio_file_record(tmp_path)

    chunking_config = ChunkingConfig(
        enabled=False,
        duration_s=3.0,
        overlap_s=0.0,
    )

    chunk_records = build_chunk_records_for_file(
        record=record,
        chunking_config=chunking_config,
    )

    assert chunk_records == []


def test_build_chunk_records_combines_multiple_files(tmp_path: Path) -> None:
    record_a = make_audio_file_record(tmp_path / 'a', duration_s=4.0)
    record_b = make_audio_file_record(tmp_path / 'b', duration_s=5.0)

    chunking_config = ChunkingConfig(
        enabled=True,
        duration_s=2.0,
        overlap_s=0.0,
    )

    chunk_records = build_chunk_records(
        records=[record_a, record_b],
        chunking_config=chunking_config,
        analysis_targets=['bird'],
    )

    assert len(chunk_records) == 5
    assert all(record.analysis_targets == ['bird'] for record in chunk_records)


def test_chunk_timestamp_property(tmp_path: Path) -> None:
    record = make_audio_file_record(tmp_path, duration_s=5.0)

    chunking_config = ChunkingConfig(
        enabled=True,
        duration_s=2.0,
        overlap_s=0.0,
    )

    chunk_records = build_chunk_records_for_file(
        record=record,
        chunking_config=chunking_config,
    )

    assert chunk_records[1].chunk_start_s == 2.0
    assert chunk_records[1].chunk_timestamp == datetime(
        2026, 4, 17, 22, 35, 43, tzinfo=timezone.utc
    )


def test_build_chunk_file_name(tmp_path: Path) -> None:
    record = make_audio_file_record(tmp_path, duration_s=5.0)

    chunk_record = AudioChunkRecord(
        parent_file_path=record.file_path,
        parent_file_name=record.file_name,
        chunk_index=3,
        chunk_start_s=2.0,
        chunk_end_s=5.0,
        chunk_duration_s=3.0,
        timestamp=record.timestamp,
        sample_rate_hz=record.sample_rate_hz,
    )

    chunk_file_name = build_chunk_file_name(chunk_record)

    assert (
        chunk_file_name
        == '24F319046907737B_20260417_223541__chunk_000003__2000_5000.wav'
    )


def test_get_chunk_output_dir_uses_configured_dir(tmp_path: Path) -> None:
    configured_dir = tmp_path / 'custom_chunks'

    chunking_config = ChunkingConfig(
        enabled=True,
        output_dir=configured_dir,
    )

    output_dir = get_chunk_output_dir(
        chunking_config=chunking_config,
        default_output_dir=tmp_path / 'processed',
    )

    assert output_dir == configured_dir


def test_get_chunk_output_dir_uses_default_when_not_configured(
    tmp_path: Path,
) -> None:
    chunking_config = ChunkingConfig(enabled=True)

    output_dir = get_chunk_output_dir(
        chunking_config=chunking_config,
        default_output_dir=tmp_path / 'processed',
    )

    assert output_dir == tmp_path / 'processed' / 'chunks'


def test_write_chunk_wav_creates_expected_file_and_duration(
    tmp_path: Path,
) -> None:
    record = make_audio_file_record(tmp_path, duration_s=2.0, sample_rate_hz=8000)

    chunk_record = AudioChunkRecord(
        parent_file_path=record.file_path,
        parent_file_name=record.file_name,
        chunk_index=0,
        chunk_start_s=0.5,
        chunk_end_s=1.5,
        chunk_duration_s=1.0,
        timestamp=record.timestamp,
        sample_rate_hz=record.sample_rate_hz,
    )

    output_path = write_chunk_wav(
        chunk_record=chunk_record,
        output_dir=tmp_path / 'chunks',
    )

    assert output_path.exists()

    with wave.open(str(output_path), 'rb') as wav_handle:
        assert wav_handle.getframerate() == 8000
        assert wav_handle.getnchannels() == 1
        assert wav_handle.getsampwidth() == 2
        assert wav_handle.getnframes() == 8000


def test_write_chunk_wavs_updates_chunk_file_paths(tmp_path: Path) -> None:
    record = make_audio_file_record(tmp_path, duration_s=2.0, sample_rate_hz=8000)

    chunking_config = ChunkingConfig(
        enabled=True,
        duration_s=1.0,
        overlap_s=0.0,
        write_audio_files=True,
        output_dir=tmp_path / 'chunks',
    )

    chunk_records = build_chunk_records_for_file(
        record=record,
        chunking_config=chunking_config,
    )

    updated_records = write_chunk_wavs(
        chunk_records=chunk_records,
        chunking_config=chunking_config,
        default_output_dir=tmp_path / 'processed',
    )

    assert len(updated_records) == 2
    assert updated_records[0].chunk_file_path is not None
    assert updated_records[1].chunk_file_path is not None
    assert updated_records[0].chunk_file_path.exists()
    assert updated_records[1].chunk_file_path.exists()


def test_write_chunk_wavs_returns_unchanged_records_when_disabled(
    tmp_path: Path,
) -> None:
    record = make_audio_file_record(tmp_path, duration_s=2.0)

    chunking_config = ChunkingConfig(
        enabled=True,
        duration_s=1.0,
        overlap_s=0.0,
        write_audio_files=False,
    )

    chunk_records = build_chunk_records_for_file(
        record=record,
        chunking_config=chunking_config,
    )

    updated_records = write_chunk_wavs(
        chunk_records=chunk_records,
        chunking_config=chunking_config,
        default_output_dir=tmp_path / 'processed',
    )

    assert len(updated_records) == 2
    assert updated_records[0].chunk_file_path is None
    assert updated_records[1].chunk_file_path is None
