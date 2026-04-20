from __future__ import annotations

from pathlib import Path

import polars as pl

from audio_ecology.config import LocationConfig, OutputConfig, PipelineConfig
from audio_ecology.ingest.inventory import (
    build_and_write_inventory,
    build_inventory_records,
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
