from __future__ import annotations

import logging
from pathlib import Path

import pytest

from audio_ecology.config import LogConfig, PipelineConfig
from audio_ecology.logging_config import (
    configure_logging,
    configure_pipeline_logging,
)


def test_configure_logging_sets_level() -> None:
    configure_logging('DEBUG')

    assert logging.getLogger().level == logging.DEBUG


def test_configure_logging_rejects_unknown_level() -> None:
    with pytest.raises(ValueError, match='Unknown log level'):
        configure_logging('LOUD')


def test_configure_logging_writes_file(tmp_path: Path) -> None:
    log_path = tmp_path / 'logs' / 'run.log'

    returned_path = configure_logging('INFO', log_file_path=log_path)
    logging.getLogger('audio_ecology.test').info('hello log file')
    logging.shutdown()

    assert returned_path == log_path
    assert 'hello log file' in log_path.read_text(encoding='utf-8')


def test_configure_pipeline_logging_uses_configured_log_dir(
    tmp_path: Path,
) -> None:
    config = PipelineConfig(
        project_root=tmp_path,
        input_dir=tmp_path / 'raw',
        output_dir=tmp_path / 'processed',
        site_name='Test Site',
        logging=LogConfig(
            write_file=True,
            output_dir=tmp_path / 'custom_logs',
        ),
    )

    log_path = configure_pipeline_logging(
        config=config,
        level='DEBUG',
        run_name='inventory',
    )

    assert log_path is not None
    assert log_path.parent == tmp_path / 'custom_logs'
    assert log_path.name.endswith('_inventory.log')
