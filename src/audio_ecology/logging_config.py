"""Logging setup helpers for command-line entry points."""

from __future__ import annotations

import logging

LOG_FORMAT = '%(asctime)s %(levelname)s %(name)s - %(message)s'


def configure_logging(level: str = 'INFO') -> None:
    """Configure application logging.

    :param level: Logging level name, such as INFO or DEBUG.
    :raises ValueError: If the level is unknown.
    """
    normalized_level = level.upper()
    numeric_level = getattr(logging, normalized_level, None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Unknown log level: {level}')

    logging.basicConfig(
        level=numeric_level,
        format=LOG_FORMAT,
        force=True,
    )
