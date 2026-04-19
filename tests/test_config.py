from __future__ import annotations

from pathlib import Path

import yaml

from audio_ecology.config import load_config


def test_load_config_resolves_relative_paths(tmp_path: Path) -> None:
    project_root = tmp_path / 'repo'
    config_dir = project_root / 'configs'
    config_dir.mkdir(parents=True)

    config_path = config_dir / 'site.yaml'
    config_data = {
        'input_dir': 'data/raw/site_a',
        'output_dir': 'data/processed/site_a',
        'site_name': 'Test Site',
        'analyses': ['inventory'],
    }
    config_path.write_text(yaml.safe_dump(config_data), encoding='utf-8')

    config = load_config(config_path, project_root=project_root)

    assert config.project_root == project_root.resolve()
    assert config.input_dir == (project_root / 'data/raw/site_a').resolve()
    assert config.output_dir == (
        project_root / 'data/processed/site_a'
    ).resolve()
    assert config.site_name == 'Test Site'


def test_load_config_keeps_absolute_paths(tmp_path: Path) -> None:
    project_root = tmp_path / 'repo'
    config_dir = project_root / 'configs'
    config_dir.mkdir(parents=True)

    external_input = tmp_path / 'external' / 'raw'
    external_output = tmp_path / 'external' / 'processed'

    config_path = config_dir / 'site.yaml'
    config_data = {
        'input_dir': str(external_input),
        'output_dir': str(external_output),
        'site_name': 'Test Site',
    }
    config_path.write_text(yaml.safe_dump(config_data), encoding='utf-8')

    config = load_config(config_path, project_root=project_root)

    assert config.input_dir == external_input.resolve()
    assert config.output_dir == external_output.resolve()


def test_load_config_resolves_birdnet_paths(tmp_path: Path) -> None:
    project_root = tmp_path / 'repo'
    config_dir = project_root / 'configs'
    config_dir.mkdir(parents=True)

    config_path = config_dir / 'site.yaml'
    config_data = {
        'input_dir': 'data/raw/site_a',
        'output_dir': 'data/processed/site_a',
        'site_name': 'Test Site',
        'birdnet': {
            'enabled': True,
            'output_dir': 'data/processed/site_a/birdnet',
        },
    }
    config_path.write_text(yaml.safe_dump(config_data), encoding='utf-8')

    config = load_config(config_path, project_root=project_root)

    assert config.birdnet.enabled is True
    assert config.birdnet.output_dir == (
        project_root / 'data/processed/site_a/birdnet'
    ).resolve()
