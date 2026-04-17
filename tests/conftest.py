from __future__ import annotations

from pathlib import Path
import wave


def create_test_wav(
    file_path: Path,
    sample_rate_hz: int = 8000,
    duration_s: float = 1.0,
    guano_fields: dict[str, str] | None = None,
) -> Path:
    """Create a small PCM WAV file for tests.

    :param file_path: Output WAV path.
    :param sample_rate_hz: Sample rate in Hz.
    :param duration_s: Duration in seconds.
    :param guano_fields: Optional GUANO fields to append.
    :return: Path to the created WAV file.
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)

    frame_count = int(sample_rate_hz * duration_s)

    with wave.open(str(file_path), 'wb') as wav_handle:
        wav_handle.setnchannels(1)
        wav_handle.setsampwidth(2)
        wav_handle.setframerate(sample_rate_hz)
        wav_handle.writeframes(b'\x00\x00' * frame_count)

    if guano_fields:
        lines = ['GUANO|Version: 1.0']
        lines.extend(f'{key}: {value}' for key, value in guano_fields.items())
        guano_text = '\n'.join(lines) + '\n'
        with file_path.open('ab') as handle:
            handle.write(guano_text.encode('utf-8'))

    return file_path