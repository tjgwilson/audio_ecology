"""Chunk generation and writing utilities for analysis-ready audio units."""

from __future__ import annotations

from pathlib import Path
import wave

from audio_ecology.config import ChunkingConfig
from audio_ecology.models import AudioChunkRecord, AudioFileRecord


def should_chunk_file(record: AudioFileRecord) -> bool:
    """Return whether a file is eligible for chunking.

    :param record: Audio file record.
    :return: True if the file can be chunked.
    """
    return (
        record.readable_wav
        and record.duration_s is not None
        and record.duration_s > 0
        and record.sample_rate_hz is not None
        and record.sample_rate_hz > 0
    )


def build_chunk_records_for_file(
    record: AudioFileRecord,
    chunking_config: ChunkingConfig,
    analysis_targets: list[str] | None = None,
) -> list[AudioChunkRecord]:
    """Build chunk records for a single audio file.

    :param record: Audio file record.
    :param chunking_config: Chunking configuration.
    :param analysis_targets: Optional analysis targets for these chunks.
    :return: Chunk records.
    """
    if not chunking_config.enabled or not should_chunk_file(record):
        return []

    duration_s = record.duration_s
    if duration_s is None:
        return []

    step_s = chunking_config.duration_s - chunking_config.overlap_s
    if step_s <= 0:
        raise ValueError('Chunk step must be greater than 0')

    chunk_records: list[AudioChunkRecord] = []
    chunk_start_s = 0.0
    chunk_index = 0

    while chunk_start_s < duration_s:
        chunk_end_s = min(chunk_start_s + chunking_config.duration_s, duration_s)

        chunk_records.append(
            AudioChunkRecord(
                parent_file_path=record.file_path,
                parent_file_name=record.file_name,
                device_id=record.device_id,
                device_label=record.device_label,
                chunk_index=chunk_index,
                chunk_start_s=round(chunk_start_s, 6),
                chunk_end_s=round(chunk_end_s, 6),
                chunk_duration_s=round(chunk_end_s - chunk_start_s, 6),
                timestamp=record.timestamp,
                latitude=record.latitude,
                longitude=record.longitude,
                sample_rate_hz=record.sample_rate_hz,
                analysis_targets=analysis_targets or [],
            )
        )

        if chunk_end_s >= duration_s:
            break

        chunk_index += 1
        chunk_start_s += step_s

    return chunk_records


def build_chunk_records(
    records: list[AudioFileRecord],
    chunking_config: ChunkingConfig,
    analysis_targets: list[str] | None = None,
) -> list[AudioChunkRecord]:
    """Build chunk records for many audio files.

    :param records: Audio file records.
    :param chunking_config: Chunking configuration.
    :param analysis_targets: Optional analysis targets for these chunks.
    :return: Chunk records.
    """
    chunk_records: list[AudioChunkRecord] = []

    for record in records:
        chunk_records.extend(
            build_chunk_records_for_file(
                record=record,
                chunking_config=chunking_config,
                analysis_targets=analysis_targets,
            )
        )

    return chunk_records


def get_chunk_output_dir(
    chunking_config: ChunkingConfig,
    default_output_dir: Path,
) -> Path:
    """Return the directory to use for physical chunk WAV outputs.

    :param chunking_config: Chunking configuration.
    :param default_output_dir: Default pipeline output directory.
    :return: Chunk output directory.
    """
    if chunking_config.output_dir is not None:
        return chunking_config.output_dir

    return default_output_dir / 'chunks'


def build_chunk_file_name(chunk_record: AudioChunkRecord) -> str:
    """Build a file name for a physical chunk WAV.

    :param chunk_record: Chunk record.
    :return: Chunk WAV file name.
    """
    parent_stem = Path(chunk_record.parent_file_name).stem
    start_ms = int(round(chunk_record.chunk_start_s * 1000))
    end_ms = int(round(chunk_record.chunk_end_s * 1000))
    return (
        f'{parent_stem}'
        f'__chunk_{chunk_record.chunk_index:06d}'
        f'__{start_ms}_{end_ms}.wav'
    )


def write_chunk_wav(
    chunk_record: AudioChunkRecord,
    output_dir: Path,
) -> Path:
    """Write one physical chunk WAV file.

    :param chunk_record: Chunk record to write.
    :param output_dir: Directory for chunk WAV outputs.
    :return: Path to the written chunk WAV file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / build_chunk_file_name(chunk_record)

    with wave.open(str(chunk_record.parent_file_path), 'rb') as source_wav:
        sample_rate_hz = source_wav.getframerate()
        n_channels = source_wav.getnchannels()
        sample_width = source_wav.getsampwidth()
        comptype = source_wav.getcomptype()
        compname = source_wav.getcompname()

        start_frame = int(round(chunk_record.chunk_start_s * sample_rate_hz))
        end_frame = int(round(chunk_record.chunk_end_s * sample_rate_hz))
        n_frames = max(0, end_frame - start_frame)

        source_wav.setpos(start_frame)
        chunk_frames = source_wav.readframes(n_frames)

    with wave.open(str(output_path), 'wb') as chunk_wav:
        chunk_wav.setnchannels(n_channels)
        chunk_wav.setsampwidth(sample_width)
        chunk_wav.setframerate(sample_rate_hz)
        chunk_wav.setcomptype(comptype, compname)
        chunk_wav.writeframes(chunk_frames)

    return output_path


def write_chunk_wavs(
    chunk_records: list[AudioChunkRecord],
    chunking_config: ChunkingConfig,
    default_output_dir: Path,
) -> list[AudioChunkRecord]:
    """Write physical chunk WAV files and update chunk records.

    :param chunk_records: Chunk records to write.
    :param chunking_config: Chunking configuration.
    :param default_output_dir: Default pipeline output directory.
    :return: Updated chunk records with chunk_file_path values.
    """
    if not chunking_config.write_audio_files:
        return chunk_records

    output_dir = get_chunk_output_dir(
        chunking_config=chunking_config,
        default_output_dir=default_output_dir,
    )

    updated_records: list[AudioChunkRecord] = []

    for chunk_record in chunk_records:
        chunk_file_path = write_chunk_wav(
            chunk_record=chunk_record,
            output_dir=output_dir,
        )
        updated_records.append(
            chunk_record.model_copy(
                update={'chunk_file_path': chunk_file_path}
            )
        )

    return updated_records