"""Core Pydantic models for ingestion records."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

TimestampSource = Literal['guano', 'filename', 'missing']
LocationSource = Literal[
    'guano',
    'deployment_config',
    'device_config',
    'site_config',
    'missing',
]


class AudioFileRecord(BaseModel):
    """Canonical inventory record for a single WAV file."""

    file_path: Path
    file_name: str
    device_id: str | None = None
    device_label: str | None = None
    deployment_id: str | None = None
    habitat_label: str | None = None
    detection_targets: list[str] = Field(default_factory=list)

    timestamp: datetime | None = None
    timestamp_source: TimestampSource = 'missing'
    filename_timestamp: datetime | None = None
    sunrise_timestamp: datetime | None = None
    sunset_timestamp: datetime | None = None
    minutes_from_sunrise: float | None = None
    minutes_to_sunset: float | None = None
    is_daylight: bool | None = None

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


class AudioChunkRecord(BaseModel):
    """Canonical record for one derived analysis chunk."""

    parent_file_path: Path
    parent_file_name: str
    chunk_file_path: Path | None = None
    device_id: str | None = None
    device_label: str | None = None
    deployment_id: str | None = None
    habitat_label: str | None = None
    detection_targets: list[str] = Field(default_factory=list)

    chunk_index: int
    chunk_start_s: float
    chunk_end_s: float
    chunk_duration_s: float

    timestamp: datetime | None = None
    latitude: float | None = None
    longitude: float | None = None
    sunrise_timestamp: datetime | None = None
    sunset_timestamp: datetime | None = None
    minutes_from_sunrise: float | None = None
    minutes_to_sunset: float | None = None
    is_daylight: bool | None = None

    sample_rate_hz: int | None = None
    analysis_targets: list[str] = Field(default_factory=list)

    @property
    def chunk_timestamp(self) -> datetime | None:
        """Return absolute timestamp for the chunk start if available."""
        if self.timestamp is None:
            return None
        return self.timestamp + timedelta(seconds=self.chunk_start_s)


class BirdDetectionRecord(BaseModel):
    """Canonical record for one BirdNET bird detection."""

    file_path: Path
    file_name: str
    detection_start_s: float
    detection_end_s: float
    detection_duration_s: float

    timestamp: datetime | None = None
    latitude: float | None = None
    longitude: float | None = None
    sunrise_timestamp: datetime | None = None
    sunset_timestamp: datetime | None = None
    minutes_from_sunrise: float | None = None
    minutes_to_sunset: float | None = None
    is_daylight: bool | None = None
    temperature_int_c: float | None = None
    deployment_id: str | None = None
    habitat_label: str | None = None
    detection_targets: list[str] = Field(default_factory=list)

    scientific_name: str
    common_name: str
    confidence: float
    analysis_backend: str = 'birdnet'
    model_name: str | None = None
    source_result_path: Path | None = None

    @property
    def detection_timestamp(self) -> datetime | None:
        """Return absolute timestamp for the detection start if available."""
        if self.timestamp is None:
            return None
        return self.timestamp + timedelta(seconds=self.detection_start_s)
