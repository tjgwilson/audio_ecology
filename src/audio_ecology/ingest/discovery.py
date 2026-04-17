"""File discovery utilities for audio ingestion."""

from __future__ import annotations

from pathlib import Path


def discover_wav_files(input_dir: Path) -> list[Path]:
    """Discover WAV files recursively beneath an input directory.

    :param input_dir: Root directory to search.
    :return: Sorted list of WAV file paths.
    :raises FileNotFoundError: If the input directory does not exist.
    """
    if not input_dir.exists():
        raise FileNotFoundError(f'Input directory not found: {input_dir}')

    wav_files = [
        path
        for path in input_dir.rglob('*')
        if path.is_file() and path.suffix.lower() == '.wav'
    ]
    return sorted(wav_files)