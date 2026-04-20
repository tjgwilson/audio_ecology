from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import polars as pl
import pytest

from audio_ecology.analysis.evidence import (
    build_noisy_or_species_time_period,
    build_noisy_or_species_windows,
    write_noisy_or_species_windows,
)
from audio_ecology.config import DetectionUncertaintyConfig


def make_detection_rows(tmp_path: Path) -> pl.DataFrame:
    """Create detections with one clustered event and one separated event."""
    return pl.DataFrame(
        [
            {
                'file_path': str(tmp_path / 'raw' / 'sample.wav'),
                'file_name': 'sample.wav',
                'detection_start_s': 3.0,
                'detection_end_s': 6.0,
                'timestamp': datetime(
                    2026,
                    4,
                    17,
                    22,
                    35,
                    41,
                    tzinfo=timezone.utc,
                ).isoformat(),
                'scientific_name': 'Erithacus rubecula',
                'common_name': 'European Robin',
                'confidence': 0.42,
            },
            {
                'file_path': str(tmp_path / 'raw' / 'sample.wav'),
                'file_name': 'sample.wav',
                'detection_start_s': 12.0,
                'detection_end_s': 15.0,
                'timestamp': datetime(
                    2026,
                    4,
                    17,
                    22,
                    35,
                    41,
                    tzinfo=timezone.utc,
                ).isoformat(),
                'scientific_name': 'Erithacus rubecula',
                'common_name': 'European Robin',
                'confidence': 0.55,
            },
            {
                'file_path': str(tmp_path / 'raw' / 'sample.wav'),
                'file_name': 'sample.wav',
                'detection_start_s': 70.0,
                'detection_end_s': 73.0,
                'timestamp': datetime(
                    2026,
                    4,
                    17,
                    22,
                    35,
                    41,
                    tzinfo=timezone.utc,
                ).isoformat(),
                'scientific_name': 'Erithacus rubecula',
                'common_name': 'European Robin',
                'confidence': 0.81,
            },
            {
                'file_path': str(tmp_path / 'raw' / 'sample.wav'),
                'file_name': 'sample.wav',
                'detection_start_s': 75.0,
                'detection_end_s': 78.0,
                'timestamp': datetime(
                    2026,
                    4,
                    17,
                    22,
                    35,
                    41,
                    tzinfo=timezone.utc,
                ).isoformat(),
                'scientific_name': 'Turdus merula',
                'common_name': 'Eurasian Blackbird',
                'confidence': 0.24,
            },
        ]
    )


def test_build_noisy_or_species_windows_collapses_events(
    tmp_path: Path,
) -> None:
    detections_df = make_detection_rows(tmp_path)

    evidence_df = build_noisy_or_species_windows(
        detections_df=detections_df,
        window_s=600.0,
        config=DetectionUncertaintyConfig(
            event_gap_s=30.0,
            min_confidence=0.25,
        ),
    )

    assert evidence_df.height == 1
    evidence = evidence_df.row(0, named=True)

    assert evidence['scientific_name'] == 'Erithacus rubecula'
    assert evidence['n_detections'] == 3
    assert evidence['n_events'] == 2
    assert evidence['max_confidence'] == 0.81
    assert evidence['mean_event_confidence'] == pytest.approx(0.68)
    assert evidence['noisy_or_evidence'] == pytest.approx(0.9145)
    assert evidence['detection_uncertainty'] == pytest.approx(0.0855)
    assert evidence['evidence_class'] == 'strong'
    assert evidence['window_start_timestamp'] == '2026-04-17T22:35:41+00:00'
    assert evidence['window_end_timestamp'] == '2026-04-17T22:45:41+00:00'


def test_build_noisy_or_species_windows_splits_windows(tmp_path: Path) -> None:
    detections_df = pl.DataFrame(
        [
            {
                'file_path': str(tmp_path / 'raw' / 'sample.wav'),
                'file_name': 'sample.wav',
                'detection_start_s': 3.0,
                'detection_end_s': 6.0,
                'scientific_name': 'Erithacus rubecula',
                'common_name': 'European Robin',
                'confidence': 0.6,
            },
            {
                'file_path': str(tmp_path / 'raw' / 'sample.wav'),
                'file_name': 'sample.wav',
                'detection_start_s': 610.0,
                'detection_end_s': 613.0,
                'scientific_name': 'Erithacus rubecula',
                'common_name': 'European Robin',
                'confidence': 0.7,
            },
        ]
    )

    evidence_df = build_noisy_or_species_windows(
        detections_df=detections_df,
        window_s=600.0,
    )

    assert evidence_df['window_index'].to_list() == [0, 1]
    assert evidence_df['noisy_or_evidence'].to_list() == [0.6, 0.7]


def test_build_noisy_or_species_time_period_groups_across_files(
    tmp_path: Path,
) -> None:
    detections_df = pl.DataFrame(
        [
            {
                'file_path': str(tmp_path / 'raw' / 'sample_a.wav'),
                'file_name': 'sample_a.wav',
                'detection_start_s': 3.0,
                'detection_end_s': 6.0,
                'timestamp': datetime(
                    2026,
                    4,
                    17,
                    22,
                    0,
                    0,
                    tzinfo=timezone.utc,
                ).isoformat(),
                'scientific_name': 'Erithacus rubecula',
                'common_name': 'European Robin',
                'confidence': 0.6,
            },
            {
                'file_path': str(tmp_path / 'raw' / 'sample_b.wav'),
                'file_name': 'sample_b.wav',
                'detection_start_s': 10.0,
                'detection_end_s': 13.0,
                'timestamp': datetime(
                    2026,
                    4,
                    17,
                    22,
                    1,
                    0,
                    tzinfo=timezone.utc,
                ).isoformat(),
                'scientific_name': 'Erithacus rubecula',
                'common_name': 'European Robin',
                'confidence': 0.8,
            },
            {
                'file_path': str(tmp_path / 'raw' / 'sample_c.wav'),
                'file_name': 'sample_c.wav',
                'detection_start_s': 10.0,
                'detection_end_s': 13.0,
                'timestamp': datetime(
                    2026,
                    4,
                    17,
                    23,
                    1,
                    0,
                    tzinfo=timezone.utc,
                ).isoformat(),
                'scientific_name': 'Erithacus rubecula',
                'common_name': 'European Robin',
                'confidence': 0.95,
            },
        ]
    )

    evidence_df = build_noisy_or_species_time_period(
        detections_df=detections_df,
        start_time=datetime(2026, 4, 17, 22, 0, 0, tzinfo=timezone.utc),
        end_time=datetime(2026, 4, 17, 23, 0, 0, tzinfo=timezone.utc),
        config=DetectionUncertaintyConfig(event_gap_s=30.0),
    )

    assert evidence_df.height == 1
    evidence = evidence_df.row(0, named=True)
    assert evidence['file_path'] is None
    assert evidence['window_start_timestamp'] == '2026-04-17T22:00:00+00:00'
    assert evidence['window_end_timestamp'] == '2026-04-17T23:00:00+00:00'
    assert evidence['n_detections'] == 2
    assert evidence['n_events'] == 2
    assert evidence['period_duration_s'] == 3600.0
    assert evidence['noisy_or_evidence'] == pytest.approx(0.92)


def test_build_noisy_or_species_time_period_keeps_models_separate(
    tmp_path: Path,
) -> None:
    detections_df = pl.DataFrame(
        [
            {
                'file_path': str(tmp_path / 'raw' / 'sample.wav'),
                'file_name': 'sample.wav',
                'analysis_backend': 'birdnet',
                'model_name': 'acoustic-2.4-tf',
                'detection_start_s': 3.0,
                'detection_end_s': 6.0,
                'timestamp': datetime(
                    2026,
                    4,
                    17,
                    22,
                    0,
                    0,
                    tzinfo=timezone.utc,
                ).isoformat(),
                'scientific_name': 'Erithacus rubecula',
                'common_name': 'European Robin',
                'confidence': 0.6,
            },
            {
                'file_path': str(tmp_path / 'raw' / 'sample.wav'),
                'file_name': 'sample.wav',
                'analysis_backend': 'other-model',
                'model_name': 'other-1',
                'detection_start_s': 8.0,
                'detection_end_s': 11.0,
                'timestamp': datetime(
                    2026,
                    4,
                    17,
                    22,
                    0,
                    0,
                    tzinfo=timezone.utc,
                ).isoformat(),
                'scientific_name': 'Erithacus rubecula',
                'common_name': 'European Robin',
                'confidence': 0.8,
            },
        ]
    )

    evidence_df = build_noisy_or_species_time_period(
        detections_df=detections_df,
        start_time=datetime(2026, 4, 17, 22, 0, 0, tzinfo=timezone.utc),
        end_time=datetime(2026, 4, 17, 22, 5, 0, tzinfo=timezone.utc),
    )

    assert evidence_df.height == 2
    assert evidence_df.select(
        ['analysis_backend', 'model_name', 'noisy_or_evidence']
    ).to_dicts() == [
        {
            'analysis_backend': 'birdnet',
            'model_name': 'acoustic-2.4-tf',
            'noisy_or_evidence': 0.6,
        },
        {
            'analysis_backend': 'other-model',
            'model_name': 'other-1',
            'noisy_or_evidence': 0.8,
        },
    ]


def test_build_noisy_or_species_windows_requires_detection_columns() -> None:
    with pytest.raises(ValueError, match='missing required columns'):
        build_noisy_or_species_windows(pl.DataFrame({'confidence': [0.5]}))


def test_write_noisy_or_species_windows_writes_outputs(tmp_path: Path) -> None:
    evidence_df = build_noisy_or_species_windows(make_detection_rows(tmp_path))

    parquet_path, csv_path = write_noisy_or_species_windows(
        evidence_df=evidence_df,
        output_dir=tmp_path / 'out',
        write_csv=True,
    )

    assert parquet_path.exists()
    assert csv_path is not None
    assert csv_path.exists()
    assert pl.read_parquet(parquet_path).to_dicts() == evidence_df.to_dicts()
