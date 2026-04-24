"""Constants used across the AudioMoth ingestion pipeline."""

from __future__ import annotations

AUDIOMOTH_FILENAME_TIMESTAMP_FORMAT = '%Y%m%d_%H%M%S'
GUANO_TIMESTAMP_FORMAT = '%Y-%m-%dT%H:%M:%SZ'

GUANO_TIMESTAMP_FIELD = 'Timestamp'
GUANO_LOCATION_FIELD = 'Loc Position'
GUANO_TEMPERATURE_INT_FIELD = 'Temperature Int'

TIMESTAMP_SOURCE_GUANO = 'guano'
TIMESTAMP_SOURCE_FILENAME = 'filename'
TIMESTAMP_SOURCE_MISSING = 'missing'

LOCATION_SOURCE_GUANO = 'guano'
LOCATION_SOURCE_DEPLOYMENT_CONFIG = 'deployment_config'
LOCATION_SOURCE_DEVICE_CONFIG = 'device_config'
LOCATION_SOURCE_SITE_CONFIG = 'site_config'
LOCATION_SOURCE_MISSING = 'missing'

ALLOWED_DETECTION_TARGETS = (
    'bird',
    'bat',
    'insect',
    'other',
    'all',
)
