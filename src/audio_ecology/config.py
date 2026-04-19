"""Configuration models and loading utilities."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, model_validator


class LocationConfig(BaseModel):
    """Fallback location configuration."""

    latitude: float | None = None
    longitude: float | None = None


class DeviceConfig(BaseModel):
    """Configuration for a specific recorder device."""

    label: str | None = None
    fallback_location: LocationConfig | None = None


class ChunkingConfig(BaseModel):
    """Configuration for optional audio chunking."""

    enabled: bool = False
    duration_s: float = 3.0
    overlap_s: float = 0.0
    write_chunk_inventory: bool = True
    write_audio_files: bool = False
    output_dir: Path | None = None

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


class PipelineConfig(BaseModel):
    """Top level pipeline configuration."""

    project_root: Path
    input_dir: Path
    output_dir: Path
    site_name: str
    fallback_location: LocationConfig | None = None
    devices: dict[str, DeviceConfig] = Field(default_factory=dict)
    analyses: list[str] = Field(default_factory=list)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)

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
    if not config_path.exists():
        raise FileNotFoundError(f'Config file not found: {config_path}')

    with config_path.open('r', encoding='utf-8') as handle:
        raw_config = yaml.safe_load(handle)

    if not isinstance(raw_config, dict):
        raise ValueError('Config file must contain a top-level mapping.')

    resolved_project_root = (
        project_root.resolve()
        if project_root is not None
        else find_project_root(config_path)
    )

    raw_config['project_root'] = resolved_project_root

    return PipelineConfig.model_validate(raw_config)