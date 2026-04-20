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
            'output_dir': 'data/processed/site_a/birdnet',
        },
    }
    config_path.write_text(yaml.safe_dump(config_data), encoding='utf-8')

    config = load_config(config_path, project_root=project_root)

    assert config.birdnet.output_dir == (
        project_root / 'data/processed/site_a/birdnet'
    ).resolve()


def test_load_config_keeps_chunking_analysis_targets(tmp_path: Path) -> None:
    project_root = tmp_path / 'repo'
    config_dir = project_root / 'configs'
    config_dir.mkdir(parents=True)

    config_path = config_dir / 'site.yaml'
    config_data = {
        'input_dir': 'data/raw/site_a',
        'output_dir': 'data/processed/site_a',
        'site_name': 'Test Site',
        'chunking': {
            'enabled': True,
            'analysis_targets': ['bird'],
        },
    }
    config_path.write_text(yaml.safe_dump(config_data), encoding='utf-8')

    config = load_config(config_path, project_root=project_root)

    assert config.chunking.analysis_targets == ['bird']


def test_load_config_loads_output_preferences(tmp_path: Path) -> None:
    project_root = tmp_path / 'repo'
    config_dir = project_root / 'configs'
    config_dir.mkdir(parents=True)

    config_path = config_dir / 'site.yaml'
    config_data = {
        'input_dir': 'data/raw/site_a',
        'output_dir': 'data/processed/site_a',
        'site_name': 'Test Site',
        'outputs': {
            'write_csv': True,
        },
    }
    config_path.write_text(yaml.safe_dump(config_data), encoding='utf-8')

    config = load_config(config_path, project_root=project_root)

    assert config.outputs.write_csv is True


def test_load_config_rejects_top_level_analyses(tmp_path: Path) -> None:
    project_root = tmp_path / 'repo'
    config_dir = project_root / 'configs'
    config_dir.mkdir(parents=True)

    config_path = config_dir / 'site.yaml'
    config_data = {
        'input_dir': 'data/raw/site_a',
        'output_dir': 'data/processed/site_a',
        'site_name': 'Test Site',
        'analyses': ['bird'],
    }
    config_path.write_text(yaml.safe_dump(config_data), encoding='utf-8')

    try:
        load_config(config_path, project_root=project_root)
    except ValueError as exc:
        assert 'Top-level "analyses" is no longer used' in str(exc)
    else:
        raise AssertionError('Expected ValueError')


def test_load_config_rejects_birdnet_enabled(tmp_path: Path) -> None:
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
        },
    }
    config_path.write_text(yaml.safe_dump(config_data), encoding='utf-8')

    try:
        load_config(config_path, project_root=project_root)
    except ValueError as exc:
        assert '"birdnet.enabled" is no longer used' in str(exc)
    else:
        raise AssertionError('Expected ValueError')
