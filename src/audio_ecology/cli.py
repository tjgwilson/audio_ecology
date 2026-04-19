"""Command line interface for the audio ecology pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from audio_ecology.config import load_config
from audio_ecology.orchestrator import (
    format_inventory_summary,
    run_inventory_pipeline_with_chunks,
)

app = typer.Typer(help='Passive acoustic monitoring pipeline.')


@app.command()
def inventory(
    config_path: Annotated[
        Path,
        typer.Argument(help='Path to the YAML configuration file.'),
    ],
    stem: Annotated[
        str,
        typer.Option(
            '--stem',
            help='Base file stem for inventory outputs.',
        ),
    ] = 'audio_inventory',
) -> None:
    """Build an inventory of WAV files from a config file."""
    config = load_config(config_path.resolve())
    inventory_df, chunk_df, summary = run_inventory_pipeline_with_chunks(
        config=config,
        stem=stem,
    )

    typer.echo(
        f'Wrote inventory with {inventory_df.height} files to '
        f'{config.output_dir}'
    )

    if chunk_df is not None:
        typer.echo(f'Wrote chunk inventory with {chunk_df.height} chunks')

    typer.echo('')
    typer.echo(format_inventory_summary(summary))


if __name__ == '__main__':
    app()
