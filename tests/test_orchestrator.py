from __future__ import annotations

from pathlib import Path

from audio_ecology.config import LocationConfig, PipelineConfig
from audio_ecology.orchestrator import (
    format_inventory_summary,
    run_inventory_pipeline,
)
from conftest import create_test_wav


def make_config(tmp_path: Path) -> PipelineConfig:
    """Create a simple config for orchestrator tests."""
    return PipelineConfig(
        project_root=tmp_path,
        input_dir=tmp_path / 'raw',
        output_dir=tmp_path / 'processed',
        site_name='Test Site',
        fallback_location=LocationConfig(
            latitude=51.5,
            longitude=-2.1,
        ),
        devices={},
    )


def test_run_inventory_pipeline_returns_summary(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    create_test_wav(
        tmp_path / 'raw' / '24F319046907737B_20260417_223541.WAV',
        guano_fields={
            'Timestamp': '2026-04-16T21:43:00Z',
            'Loc Position': '50.432584 -3.672039',
            'Temperature Int': '18.5',
        },
    )

    inventory_df, chunk_df, summary = run_inventory_pipeline(config)

    assert inventory_df.height == 1
    assert chunk_df is None
    assert summary['n_files'] == 1
    assert summary['n_unreadable_wav'] == 0
    assert summary['n_missing_timestamp'] == 0
    assert summary['n_missing_location'] == 0
    assert summary['n_missing_temperature'] == 0
    assert summary['n_with_guano'] == 1


def test_format_inventory_summary_contains_expected_lines() -> None:
    summary = {
        'n_files': 2,
        'n_unreadable_wav': 1,
        'n_missing_timestamp': 0,
        'n_missing_location': 1,
        'n_missing_temperature': 2,
        'n_with_guano': 1,
        'device_ids': ['A', 'B'],
        'sample_rate_counts': {'8000': 1, '16000': 1},
    }

    formatted = format_inventory_summary(summary)

    assert 'Files: 2' in formatted
    assert 'Unreadable WAVs: 1' in formatted
    assert 'Device IDs: A, B' in formatted
    assert '8000: 1' in formatted
    assert '16000: 1' in formatted
