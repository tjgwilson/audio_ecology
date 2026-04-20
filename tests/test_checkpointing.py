from __future__ import annotations

from pathlib import Path

import polars as pl

from audio_ecology.analysis.checkpointing import AnalysisCheckpointStore


def test_analysis_checkpoint_store_writes_and_reads_checkpoint(
    tmp_path: Path,
) -> None:
    store = AnalysisCheckpointStore(
        output_dir=tmp_path,
        backend_name='test_backend',
        schema={
            'file_path': pl.Utf8,
            'confidence': pl.Float64,
        },
    )
    input_path = tmp_path / 'raw' / 'example file.wav'
    results_df = pl.DataFrame(
        [
            {
                'file_path': str(input_path),
                'confidence': 0.9,
            }
        ]
    )

    checkpoint_path = store.write(input_path, results_df)
    loaded_df = store.read(input_path)

    assert checkpoint_path.exists()
    assert store.exists(input_path) is True
    assert checkpoint_path.parent == tmp_path / 'checkpoints' / 'test_backend'
    assert loaded_df.to_dicts() == results_df.to_dicts()


def test_analysis_checkpoint_store_reads_empty_schema_when_no_checkpoints(
    tmp_path: Path,
) -> None:
    store = AnalysisCheckpointStore(
        output_dir=tmp_path,
        backend_name='test_backend',
        schema={
            'file_path': pl.Utf8,
            'confidence': pl.Float64,
        },
    )

    checkpoint_df = store.read_all()

    assert checkpoint_df.is_empty()
    assert checkpoint_df.schema == store.schema
