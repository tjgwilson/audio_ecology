from __future__ import annotations

import logging

import pytest

from audio_ecology.logging_config import configure_logging


def test_configure_logging_sets_level() -> None:
    configure_logging('DEBUG')

    assert logging.getLogger().level == logging.DEBUG


def test_configure_logging_rejects_unknown_level() -> None:
    with pytest.raises(ValueError, match='Unknown log level'):
        configure_logging('LOUD')
