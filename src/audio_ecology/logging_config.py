"""Logging setup helpers for command-line entry points."""

from __future__ import annotations

from datetime import datetime, timezone
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from audio_ecology.config import PipelineConfig

LOG_FORMAT = '%(asctime)s %(levelname)s %(name)s - %(message)s'


def configure_logging(
    level: str = 'INFO',
    log_file_path: Path | None = None,
) -> Path | None:
    """Configure application logging.

    :param level: Logging level name, such as INFO or DEBUG.
    :param log_file_path: Optional path for a log file.
    :return: The log file path when file logging is enabled.
    :raises ValueError: If the level is unknown.
    """
    normalized_level = level.upper()
    numeric_level = getattr(logging, normalized_level, None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Unknown log level: {level}')

    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file_path is not None:
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file_path, encoding='utf-8'))

    logging.basicConfig(
        level=numeric_level,
        format=LOG_FORMAT,
        handlers=handlers,
        force=True,
    )
    return log_file_path


def configure_pipeline_logging(
    config: PipelineConfig,
    level: str = 'INFO',
    run_name: str = 'pipeline',
) -> Path | None:
    """Configure logging using pipeline config preferences.

    :param config: Loaded pipeline configuration.
    :param level: Logging level name, such as INFO or DEBUG.
    :param run_name: Component name used in the log file name.
    :return: The log file path when file logging is enabled.
    """
    log_file_path = None
    if config.logging.write_file:
        log_dir = config.logging.output_dir or config.output_dir / 'logs'
        timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        log_file_path = log_dir / f'{timestamp}_{run_name}.log'

    return configure_logging(level=level, log_file_path=log_file_path)
