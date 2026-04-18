"""Core Pydantic models for ingestion records."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel


TimestampSource = Literal['guano', 'filename', 'missing']
LocationSource = Literal['guano', 'device_config', 'site_config', 'missing']


class AudioFileRecord(BaseModel):
    """Canonical inventory record for a single WAV file."""

    file_path: Path
    file_name: str
    device_id: str | None = None
    device_label: str | None = None

    timestamp: datetime | None = None
    timestamp_source: TimestampSource = 'missing'
    filename_timestamp: datetime | None = None

    sample_rate_hz: int | None = None
    duration_s: float | None = None
    file_size_bytes: int | None = None

    latitude: float | None = None
    longitude: float | None = None
    location_source: LocationSource = 'missing'

    temperature_int_c: float | None = None

    guano_present: bool = False
    readable_wav: bool = True
    notes: str | None = None