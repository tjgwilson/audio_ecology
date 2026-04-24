"""Configuration models and loading utilities."""

from __future__ import annotations

from datetime import datetime, timedelta
import logging
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from audio_ecology.constants import ALLOWED_DETECTION_TARGETS

logger = logging.getLogger(__name__)


def _validate_detection_target_labels(
    labels: list[str],
    field_name: str,
) -> list[str]:
    """Validate configured target labels against the supported vocabulary.

    :param labels: Configured target labels.
    :param field_name: Name of the config field being validated.
    :return: Validated labels unchanged.
    :raises ValueError: If any labels are unsupported.
    """
    invalid_labels = [
        label for label in labels if label not in ALLOWED_DETECTION_TARGETS
    ]
    if invalid_labels:
        allowed_values = ', '.join(ALLOWED_DETECTION_TARGETS)
        invalid_values = ', '.join(sorted(invalid_labels))
        raise ValueError(
            f'{field_name} contains unsupported values: {invalid_values}. '
            f'Allowed values are: {allowed_values}'
        )

    return labels


class LocationConfig(BaseModel):
    """Fallback location configuration."""

    latitude: float | None = None
    longitude: float | None = None


class DeviceConfig(BaseModel):
    """Configuration for a specific recorder device."""

    label: str | None = None
    fallback_location: LocationConfig | None = None


class DeploymentConfig(BaseModel):
    """Configuration for one active recorder deployment."""

    device_id: str
    habitat_label: str | None = None
    detection_targets: list[str] = Field(default_factory=list)
    fallback_location: LocationConfig | None = None

    @field_validator('detection_targets')
    @classmethod
    def validate_detection_targets(cls, labels: list[str]) -> list[str]:
        """Validate deployment detection target labels.

        :param labels: Configured detection targets.
        :return: Validated detection targets.
        """
        return _validate_detection_target_labels(
            labels,
            field_name='deployments.detection_targets',
        )


class ChunkingConfig(BaseModel):
    """Configuration for optional audio chunking."""

    enabled: bool = False
    duration_s: float = 3.0
    overlap_s: float = 0.0
    write_chunk_inventory: bool = True
    write_audio_files: bool = False
    output_dir: Path | None = None
    analysis_targets: list[str] = Field(default_factory=list)

    @field_validator('analysis_targets')
    @classmethod
    def validate_analysis_targets(cls, labels: list[str]) -> list[str]:
        """Validate chunking analysis target labels.

        :param labels: Configured analysis targets.
        :return: Validated analysis targets.
        """
        return _validate_detection_target_labels(
            labels,
            field_name='chunking.analysis_targets',
        )

    @model_validator(mode='after')
    def validate_chunking(self) -> 'ChunkingConfig':
        """Validate chunking settings."""
        if self.duration_s <= 0:
            raise ValueError('chunking.duration_s must be greater than 0')

        if self.overlap_s < 0:
            raise ValueError('chunking.overlap_s must be non-negative')

        if self.overlap_s >= self.duration_s:
            raise ValueError(
                'chunking.overlap_s must be smaller than chunking.duration_s'
            )

        return self


class BirdNETConfig(BaseModel):
    """Configuration for BirdNET bird detection."""

    output_dir: Path | None = None
    model_version: str = '2.4'
    model_backend: str = 'tf'
    min_confidence: float = 0.25
    use_location_filter: bool = True
    location_min_confidence: float = 0.03
    batch_size: int = 1
    fmin_hz: float = 0.0
    fmax_hz: float = 15000.0
    sensitivity: float = 1.0
    overlap_s: float = 0.0

    @model_validator(mode='after')
    def validate_birdnet(self) -> 'BirdNETConfig':
        """Validate BirdNET settings."""
        if not 0.00001 <= self.min_confidence <= 0.99:
            raise ValueError(
                'birdnet.min_confidence must be between 0.00001 and 0.99'
            )

        if not 0.0 <= self.location_min_confidence <= 0.99:
            raise ValueError(
                'birdnet.location_min_confidence must be between 0.0 and 0.99'
            )

        if self.batch_size <= 0:
            raise ValueError('birdnet.batch_size must be greater than 0')

        if self.fmin_hz < 0:
            raise ValueError('birdnet.fmin_hz must be non-negative')

        if self.fmax_hz <= self.fmin_hz:
            raise ValueError('birdnet.fmax_hz must be greater than fmin_hz')

        if not 0.0 <= self.overlap_s <= 2.9:
            raise ValueError('birdnet.overlap_s must be between 0.0 and 2.9')

        return self


class DetectionUncertaintyConfig(BaseModel):
    """Configuration for window-level detection uncertainty summaries."""

    detections_path: Path | None = None
    output_dir: Path | None = None
    output_stem: str = 'detection_window_evidence'
    start_time: datetime | None = None
    end_time: datetime | None = None
    duration_s: float | None = None
    event_gap_s: float = 30.0
    min_confidence: float = 0.25
    possible_threshold: float = 0.40
    probable_threshold: float = 0.70
    strong_threshold: float = 0.90

    @property
    def resolved_start_time(self) -> datetime | None:
        """Return the resolved window start time.

        :return: Window start time, or ``None`` when unset.
        """
        return self.start_time

    @property
    def resolved_end_time(self) -> datetime | None:
        """Return the resolved window end time.

        :return: Explicit end time or start time plus duration.
        """
        if self.end_time is not None:
            return self.end_time

        if self.start_time is not None and self.duration_s is not None:
            return self.start_time + timedelta(seconds=self.duration_s)

        return None

    @model_validator(mode='after')
    def validate_detection_uncertainty(self) -> 'DetectionUncertaintyConfig':
        """Validate detection uncertainty settings."""
        if self.start_time is not None and self.end_time is not None and (
            self.end_time <= self.start_time
        ):
            raise ValueError(
                'detection_uncertainty.end_time must be after start_time'
            )

        if self.duration_s is not None and self.duration_s <= 0:
            raise ValueError('detection_uncertainty.duration_s must be greater than 0')

        has_start_time = self.start_time is not None
        has_end_time = self.end_time is not None
        has_duration = self.duration_s is not None
        if has_end_time and has_duration:
            raise ValueError(
                'detection_uncertainty may use end_time or duration_s, but not both'
            )

        if has_start_time and not (has_end_time or has_duration):
            raise ValueError(
                'detection_uncertainty.start_time requires end_time or duration_s'
            )

        if (has_end_time or has_duration) and not has_start_time:
            raise ValueError(
                'detection_uncertainty.end_time and duration_s require start_time'
            )

        if self.event_gap_s < 0:
            raise ValueError('detection_uncertainty.event_gap_s must be non-negative')

        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError(
                'detection_uncertainty.min_confidence must be between 0.0 and 1.0'
            )

        thresholds = [
            self.possible_threshold,
            self.probable_threshold,
            self.strong_threshold,
        ]
        if any(not 0.0 <= threshold <= 1.0 for threshold in thresholds):
            raise ValueError(
                'detection_uncertainty thresholds must be between 0.0 and 1.0'
            )

        if thresholds != sorted(thresholds):
            raise ValueError(
                'detection_uncertainty thresholds must be ordered from '
                'possible to strong'
            )

        return self


class OutputConfig(BaseModel):
    """Configuration for shared pipeline output formats."""

    write_csv: bool = False


class LogConfig(BaseModel):
    """Configuration for optional log file output."""

    write_file: bool = False
    output_dir: Path | None = None


class PipelineConfig(BaseModel):
    """Top level pipeline configuration."""

    project_root: Path
    input_dir: Path
    output_dir: Path
    site_name: str
    fallback_location: LocationConfig | None = None
    devices: dict[str, DeviceConfig] = Field(default_factory=dict)
    deployments: dict[str, DeploymentConfig] = Field(default_factory=dict)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    birdnet: BirdNETConfig = Field(default_factory=BirdNETConfig)
    detection_uncertainty: DetectionUncertaintyConfig = Field(
        default_factory=DetectionUncertaintyConfig
    )
    outputs: OutputConfig = Field(default_factory=OutputConfig)
    logging: LogConfig = Field(default_factory=LogConfig)

    @model_validator(mode='after')
    def resolve_paths(self) -> 'PipelineConfig':
        """Resolve configured paths against the project root."""
        self.project_root = self.project_root.resolve()

        if not self.input_dir.is_absolute():
            self.input_dir = (self.project_root / self.input_dir).resolve()

        if not self.output_dir.is_absolute():
            self.output_dir = (self.project_root / self.output_dir).resolve()

        if (
            self.chunking.output_dir is not None
            and not self.chunking.output_dir.is_absolute()
        ):
            self.chunking.output_dir = (
                self.project_root / self.chunking.output_dir
            ).resolve()

        if (
            self.birdnet.output_dir is not None
            and not self.birdnet.output_dir.is_absolute()
        ):
            self.birdnet.output_dir = (
                self.project_root / self.birdnet.output_dir
            ).resolve()

        if (
            self.detection_uncertainty.detections_path is not None
            and not self.detection_uncertainty.detections_path.is_absolute()
        ):
            self.detection_uncertainty.detections_path = (
                self.project_root / self.detection_uncertainty.detections_path
            ).resolve()

        if (
            self.detection_uncertainty.output_dir is not None
            and not self.detection_uncertainty.output_dir.is_absolute()
        ):
            self.detection_uncertainty.output_dir = (
                self.project_root / self.detection_uncertainty.output_dir
            ).resolve()

        if (
            self.logging.output_dir is not None
            and not self.logging.output_dir.is_absolute()
        ):
            self.logging.output_dir = (
                self.project_root / self.logging.output_dir
            ).resolve()

        deployment_device_ids = [
            deployment.device_id for deployment in self.deployments.values()
        ]
        if len(deployment_device_ids) != len(set(deployment_device_ids)):
            raise ValueError(
                'deployments must contain at most one active deployment per device_id'
            )

        return self


def find_project_root(start_path: Path) -> Path:
    """Find the repository root by searching upwards.

    The search looks for a directory containing ``pyproject.toml`` or ``.git``.

    :param start_path: Starting path for the upward search.
    :return: Repository root path.
    :raises FileNotFoundError: If no project root marker is found.
    """
    current = start_path.resolve()

    if current.is_file():
        current = current.parent

    for candidate in [current, *current.parents]:
        if (candidate / 'pyproject.toml').exists():
            return candidate
        if (candidate / '.git').exists():
            return candidate

    raise FileNotFoundError(
        f'Could not find project root from starting path: {start_path}'
    )


def load_config(
    config_path: Path,
    project_root: Path | None = None,
) -> PipelineConfig:
    """Load and validate a YAML configuration file.

    Relative paths in the config are resolved against the project root.

    :param config_path: Path to the config YAML file.
    :param project_root: Optional explicit project root.
    :return: Validated pipeline configuration.
    :raises FileNotFoundError: If the config file does not exist.
    :raises ValueError: If the YAML root is not a mapping.
    """
    logger.info('Loading config from %s', config_path)
    if not config_path.exists():
        raise FileNotFoundError(f'Config file not found: {config_path}')

    with config_path.open('r', encoding='utf-8') as handle:
        raw_config = yaml.safe_load(handle)

    if not isinstance(raw_config, dict):
        raise ValueError('Config file must contain a top-level mapping.')

    if 'analyses' in raw_config:
        raise ValueError(
            'Top-level "analyses" is no longer used. Choose what runs with '
            'the CLI command, such as "inventory" or "birds". If you need '
            'labels on chunk records, use "chunking.analysis_targets".'
        )

    birdnet_config = raw_config.get('birdnet')
    if isinstance(birdnet_config, dict) and 'enabled' in birdnet_config:
        raise ValueError(
            '"birdnet.enabled" is no longer used. Run the "birds" command to '
            'perform BirdNET analysis; keep the birdnet config block for '
            'BirdNET parameters only.'
        )

    resolved_project_root = (
        project_root.resolve()
        if project_root is not None
        else find_project_root(config_path)
    )

    raw_config['project_root'] = resolved_project_root

    config = PipelineConfig.model_validate(raw_config)
    logger.info(
        'Loaded config for site %s: input=%s output=%s',
        config.site_name,
        config.input_dir,
        config.output_dir,
    )
    return config
