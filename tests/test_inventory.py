from __future__ import annotations

from pathlib import Path

import polars as pl

from audio_ecology.config import LocationConfig, OutputConfig, PipelineConfig
from audio_ecology.ingest.inventory import (
    build_and_write_inventory,
    build_inventory_records,
    merge_chunk_inventory_dataframes,
    merge_inventory_dataframes,
    records_to_polars,
    write_inventory_outputs,
)
from conftest import create_test_wav


def make_config(tmp_path: Path) -> PipelineConfig:
    """Create a simple config for inventory tests."""
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


def test_records_to_polars_creates_expected_columns(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    create_test_wav(
        tmp_path / 'raw' / '24F319046907737B_20260417_223541.WAV',
        guano_fields={
            'Timestamp': '2026-04-16T21:43:00Z',
            'Loc Position': '50.432584 -3.672039',
            'Temperature Int': '18.5',
        },
    )

    records = build_inventory_records(config)
    inventory_df = records_to_polars(records)

    assert isinstance(inventory_df, pl.DataFrame)
    assert inventory_df.height == 1
    assert 'device_id' in inventory_df.columns
    assert 'timestamp' in inventory_df.columns
    assert 'latitude' in inventory_df.columns
    assert 'temperature_int_c' in inventory_df.columns


def test_build_and_write_inventory_writes_parquet_by_default(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    create_test_wav(
        tmp_path / 'raw' / '24F319046907737B_20260417_223541.WAV',
        guano_fields={
            'Timestamp': '2026-04-16T21:43:00Z',
            'Loc Position': '50.432584 -3.672039',
        },
    )

    inventory_df = build_and_write_inventory(config)

    parquet_path = config.output_dir / 'audio_inventory.parquet'
    csv_path = config.output_dir / 'audio_inventory.csv'

    assert inventory_df.height == 1
    assert parquet_path.exists()
    assert not csv_path.exists()

    loaded_df = pl.read_parquet(parquet_path)
    assert loaded_df.height == 1
    assert loaded_df['device_id'].to_list() == ['24F319046907737B']


def test_write_inventory_outputs_writes_csv_when_enabled(tmp_path: Path) -> None:
    config = PipelineConfig(
        project_root=tmp_path,
        input_dir=tmp_path / 'raw',
        output_dir=tmp_path / 'processed',
        site_name='Test Site',
        outputs=OutputConfig(write_csv=True),
    )
    inventory_df = pl.DataFrame(
        [
            {
                'file_path': 'example.wav',
                'file_name': 'example.wav',
                'device_id': 'device',
            }
        ]
    )

    parquet_path, csv_path = write_inventory_outputs(
        inventory_df=inventory_df,
        output_dir=config.output_dir,
        write_csv=config.outputs.write_csv,
    )

    assert parquet_path.exists()
    assert csv_path is not None
    assert csv_path.exists()


def test_write_inventory_outputs_merges_with_existing_inventory(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / 'processed'
    first_df = pl.DataFrame(
        [
            {
                'file_path': 'raw/device_a_20260417_220000.WAV',
                'file_name': 'device_a_20260417_220000.WAV',
                'deployment_id': 'north',
                'detection_targets': ['bird'],
                'latitude': 51.5001,
                'longitude': -2.1001,
            }
        ]
    )
    second_df = pl.DataFrame(
        [
            {
                'file_path': 'raw/device_b_20260417_221000.WAV',
                'file_name': 'device_b_20260417_221000.WAV',
                'deployment_id': 'south',
                'detection_targets': ['bat'],
                'latitude': 51.6001,
                'longitude': -2.2001,
            }
        ]
    )

    write_inventory_outputs(first_df, output_dir)
    write_inventory_outputs(second_df, output_dir)

    merged_df = pl.read_parquet(output_dir / 'audio_inventory.parquet')

    assert merged_df.height == 2
    assert set(merged_df['deployment_id'].to_list()) == {'north', 'south'}
    assert merged_df.select(['file_name', 'latitude', 'longitude']).to_dicts() == [
        {
            'file_name': 'device_a_20260417_220000.WAV',
            'latitude': 51.5001,
            'longitude': -2.1001,
        },
        {
            'file_name': 'device_b_20260417_221000.WAV',
            'latitude': 51.6001,
            'longitude': -2.2001,
        },
    ]


def test_merge_inventory_dataframes_keeps_latest_duplicate_file() -> None:
    existing_df = pl.DataFrame(
        [
            {
                'file_path': 'raw/device_a_20260417_220000.WAV',
                'file_name': 'device_a_20260417_220000.WAV',
                'deployment_id': 'north',
                'detection_targets': ['bird'],
                'latitude': 51.5001,
                'longitude': -2.1001,
            }
        ]
    )
    incoming_df = pl.DataFrame(
        [
            {
                'file_path': 'raw/device_a_20260417_220000.WAV',
                'file_name': 'device_a_20260417_220000.WAV',
                'deployment_id': 'north_updated',
                'detection_targets': ['bird', 'bat'],
                'latitude': 51.5002,
                'longitude': -2.1002,
            }
        ]
    )

    merged_df = merge_inventory_dataframes(existing_df, incoming_df)

    assert merged_df.height == 1
    assert merged_df.to_dicts() == [
        {
            'file_path': 'raw/device_a_20260417_220000.WAV',
            'file_name': 'device_a_20260417_220000.WAV',
            'deployment_id': 'north_updated',
            'detection_targets': ['bird', 'bat'],
            'latitude': 51.5002,
            'longitude': -2.1002,
        }
    ]


def test_merge_chunk_inventory_dataframes_keeps_latest_duplicate_chunk() -> None:
    existing_df = pl.DataFrame(
        [
            {
                'parent_file_path': 'raw/device_a_20260417_220000.WAV',
                'parent_file_name': 'device_a_20260417_220000.WAV',
                'chunk_file_path': 'chunks/chunk0.wav',
                'chunk_index': 0,
                'chunk_start_s': 0.0,
                'chunk_end_s': 3.0,
                'chunk_duration_s': 3.0,
            }
        ]
    )
    incoming_df = pl.DataFrame(
        [
            {
                'parent_file_path': 'raw/device_a_20260417_220000.WAV',
                'parent_file_name': 'device_a_20260417_220000.WAV',
                'chunk_file_path': 'chunks/chunk0_rebuilt.wav',
                'chunk_index': 0,
                'chunk_start_s': 0.0,
                'chunk_end_s': 3.0,
                'chunk_duration_s': 3.0,
            }
        ]
    )

    merged_df = merge_chunk_inventory_dataframes(existing_df, incoming_df)

    assert merged_df.height == 1
    assert merged_df['chunk_file_path'].to_list() == ['chunks/chunk0_rebuilt.wav']
