from __future__ import annotations

from datetime import datetime
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


def test_load_config_resolves_detection_uncertainty_paths(tmp_path: Path) -> None:
    project_root = tmp_path / 'repo'
    config_dir = project_root / 'configs'
    config_dir.mkdir(parents=True)

    config_path = config_dir / 'site.yaml'
    config_data = {
        'input_dir': 'data/raw/site_a',
        'output_dir': 'data/processed/site_a',
        'site_name': 'Test Site',
        'detection_uncertainty': {
            'detections_path': 'data/processed/site_a/detections.parquet',
            'output_dir': 'data/processed/site_a/detection_uncertainty',
        },
    }
    config_path.write_text(yaml.safe_dump(config_data), encoding='utf-8')

    config = load_config(config_path, project_root=project_root)

    assert config.detection_uncertainty.detections_path == (
        project_root / 'data/processed/site_a/detections.parquet'
    ).resolve()
    assert config.detection_uncertainty.output_dir == (
        project_root / 'data/processed/site_a/detection_uncertainty'
    ).resolve()


def test_load_config_accepts_detection_uncertainty_duration(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / 'repo'
    config_dir = project_root / 'configs'
    config_dir.mkdir(parents=True)

    config_path = config_dir / 'site.yaml'
    config_data = {
        'input_dir': 'data/raw/site_a',
        'output_dir': 'data/processed/site_a',
        'site_name': 'Test Site',
        'detection_uncertainty': {
            'start_time': '2026-04-20T12:35:00+00:00',
            'duration_s': 3600,
        },
    }
    config_path.write_text(yaml.safe_dump(config_data), encoding='utf-8')

    config = load_config(config_path, project_root=project_root)

    assert config.detection_uncertainty.resolved_start_time == datetime.fromisoformat(
        '2026-04-20T12:35:00+00:00'
    )
    assert config.detection_uncertainty.resolved_end_time == datetime.fromisoformat(
        '2026-04-20T13:35:00+00:00'
    )


def test_load_config_rejects_detection_uncertainty_end_time_and_duration(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / 'repo'
    config_dir = project_root / 'configs'
    config_dir.mkdir(parents=True)

    config_path = config_dir / 'site.yaml'
    config_data = {
        'input_dir': 'data/raw/site_a',
        'output_dir': 'data/processed/site_a',
        'site_name': 'Test Site',
        'detection_uncertainty': {
            'start_time': '2026-04-20T12:35:00+00:00',
            'end_time': '2026-04-20T13:35:00+00:00',
            'duration_s': 3600,
        },
    }
    config_path.write_text(yaml.safe_dump(config_data), encoding='utf-8')

    try:
        load_config(config_path, project_root=project_root)
    except ValueError as exc:
        assert 'end_time or duration_s, but not both' in str(exc)
    else:
        raise AssertionError('Expected mixed end_time and duration_s to fail')


def test_load_config_rejects_detection_uncertainty_duration_without_start_time(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / 'repo'
    config_dir = project_root / 'configs'
    config_dir.mkdir(parents=True)

    config_path = config_dir / 'site.yaml'
    config_data = {
        'input_dir': 'data/raw/site_a',
        'output_dir': 'data/processed/site_a',
        'site_name': 'Test Site',
        'detection_uncertainty': {
            'duration_s': 3600,
        },
    }
    config_path.write_text(yaml.safe_dump(config_data), encoding='utf-8')

    try:
        load_config(config_path, project_root=project_root)
    except ValueError as exc:
        assert 'require start_time' in str(exc)
    else:
        raise AssertionError('Expected duration_s without start_time to fail')


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


def test_load_config_rejects_invalid_chunking_analysis_targets(
    tmp_path: Path,
) -> None:
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
            'analysis_targets': ['birds'],
        },
    }
    config_path.write_text(yaml.safe_dump(config_data), encoding='utf-8')

    try:
        load_config(config_path, project_root=project_root)
    except ValueError as exc:
        assert 'chunking.analysis_targets' in str(exc)
        assert 'birds' in str(exc)
    else:
        raise AssertionError('Expected invalid chunking analysis_targets to fail')


def test_load_config_rejects_invalid_deployment_detection_targets(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / 'repo'
    config_dir = project_root / 'configs'
    config_dir.mkdir(parents=True)

    config_path = config_dir / 'site.yaml'
    config_data = {
        'input_dir': 'data/raw/site_a',
        'output_dir': 'data/processed/site_a',
        'site_name': 'Test Site',
        'deployments': {
            'deploy_a': {
                'device_id': '24F319046907737B',
                'detection_targets': ['birds'],
            }
        },
    }
    config_path.write_text(yaml.safe_dump(config_data), encoding='utf-8')

    try:
        load_config(config_path, project_root=project_root)
    except ValueError as exc:
        assert 'deployments.detection_targets' in str(exc)
        assert 'birds' in str(exc)
    else:
        raise AssertionError(
            'Expected invalid deployment detection_targets to fail'
        )


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


def test_load_config_loads_logging_preferences(tmp_path: Path) -> None:
    project_root = tmp_path / 'repo'
    config_dir = project_root / 'configs'
    config_dir.mkdir(parents=True)

    config_path = config_dir / 'site.yaml'
    config_data = {
        'input_dir': 'data/raw/site_a',
        'output_dir': 'data/processed/site_a',
        'site_name': 'Test Site',
        'logging': {
            'write_file': True,
            'output_dir': 'data/processed/site_a/logs',
        },
    }
    config_path.write_text(yaml.safe_dump(config_data), encoding='utf-8')

    config = load_config(config_path, project_root=project_root)

    assert config.logging.write_file is True
    assert config.logging.output_dir == (
        project_root / 'data/processed/site_a/logs'
    ).resolve()


def test_load_config_rejects_multiple_deployments_for_one_device(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / 'repo'
    config_dir = project_root / 'configs'
    config_dir.mkdir(parents=True)

    config_path = config_dir / 'site.yaml'
    config_data = {
        'input_dir': 'data/raw/site_a',
        'output_dir': 'data/processed/site_a',
        'site_name': 'Test Site',
        'deployments': {
            'deploy_a': {'device_id': '24F319046907737B'},
            'deploy_b': {'device_id': '24F319046907737B'},
        },
    }
    config_path.write_text(yaml.safe_dump(config_data), encoding='utf-8')

    try:
        load_config(config_path, project_root=project_root)
    except ValueError as exc:
        assert 'at most one active deployment per device_id' in str(exc)
    else:
        raise AssertionError('Expected duplicate deployment device_id to fail')


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
