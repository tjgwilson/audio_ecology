"""Reusable checkpoint helpers for long-running analysis stages."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import logging
from pathlib import Path

import polars as pl

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AnalysisCheckpointStore:
    """Store per-input analysis checkpoints for resumable model runs."""

    output_dir: Path
    backend_name: str
    schema: dict[str, pl.DataType]

    @property
    def checkpoint_dir(self) -> Path:
        """Return the checkpoint directory for this backend."""
        return self.output_dir / 'checkpoints' / self.backend_name

    def checkpoint_path(self, input_path: Path) -> Path:
        """Return the checkpoint path for one input file."""
        resolved_path = input_path.resolve()
        path_hash = hashlib.sha1(
            str(resolved_path).encode('utf-8')
        ).hexdigest()[:12]
        file_stem = _safe_file_stem(input_path)
        return self.checkpoint_dir / f'{file_stem}__{path_hash}.parquet'

    def exists(self, input_path: Path) -> bool:
        """Return whether a checkpoint exists for one input file."""
        return self.checkpoint_path(input_path).exists()

    def read(self, input_path: Path) -> pl.DataFrame:
        """Read one input file checkpoint."""
        checkpoint_path = self.checkpoint_path(input_path)
        logger.info('Loading checkpoint %s', checkpoint_path)
        return pl.read_parquet(checkpoint_path)

    def write(self, input_path: Path, results_df: pl.DataFrame) -> Path:
        """Write one input file checkpoint."""
        checkpoint_path = self.checkpoint_path(input_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.write_parquet(checkpoint_path)
        logger.info(
            'Wrote checkpoint %s with %d rows',
            checkpoint_path,
            results_df.height,
        )
        return checkpoint_path

    def read_all(self) -> pl.DataFrame:
        """Read and combine all checkpoints for this backend."""
        checkpoint_paths = sorted(self.checkpoint_dir.glob('*.parquet'))
        if not checkpoint_paths:
            return pl.DataFrame(schema=self.schema)

        checkpoint_dfs = [
            pl.read_parquet(checkpoint_path)
            for checkpoint_path in checkpoint_paths
        ]
        non_empty_dfs = [
            checkpoint_df
            for checkpoint_df in checkpoint_dfs
            if not checkpoint_df.is_empty()
        ]
        if not non_empty_dfs:
            return pl.DataFrame(schema=self.schema)

        return pl.concat(non_empty_dfs, how='diagonal_relaxed')


def _safe_file_stem(input_path: Path) -> str:
    """Return a conservative file stem for checkpoint file names."""
    safe_characters = []
    for character in input_path.stem:
        if character.isalnum() or character in ('-', '_'):
            safe_characters.append(character)
        else:
            safe_characters.append('_')

    safe_stem = ''.join(safe_characters).strip('_')
    return safe_stem or 'input'
