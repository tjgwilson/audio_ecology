"""File discovery utilities for audio ingestion."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def discover_wav_files(input_dir: Path) -> list[Path]:
    """Discover WAV files recursively beneath an input directory.

    :param input_dir: Root directory to search.
    :return: Sorted list of WAV file paths.
    :raises FileNotFoundError: If the input directory does not exist.
    """
    logger.info('Discovering WAV files under %s', input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f'Input directory not found: {input_dir}')

    wav_files = [
        path
        for path in input_dir.rglob('*')
        if path.is_file() and path.suffix.lower() == '.wav'
    ]
    sorted_wav_files = sorted(wav_files)
    logger.info('Discovered %d WAV files', len(sorted_wav_files))
    logger.debug('Discovered WAV file paths: %s', sorted_wav_files)
    return sorted_wav_files
