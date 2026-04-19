"""Command line interface for the audio ecology pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from audio_ecology.analysis.birdnet import (
    get_birdnet_output_dir,
    run_birdnet_analysis,
)
from audio_ecology.config import load_config
from audio_ecology.logging_config import configure_logging
from audio_ecology.orchestrator import (
    format_inventory_summary,
    run_inventory_pipeline,
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
    log_level: Annotated[
        str,
        typer.Option(
            '--log-level',
            help='Logging level: INFO or DEBUG.',
        ),
    ] = 'INFO',
) -> None:
    """Build an inventory of WAV files from a config file."""
    configure_logging(log_level)
    config = load_config(config_path.resolve())
    inventory_df, chunk_df, summary = run_inventory_pipeline(
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


@app.command()
def birds(
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
    log_level: Annotated[
        str,
        typer.Option(
            '--log-level',
            help='Logging level: INFO or DEBUG.',
        ),
    ] = 'INFO',
) -> None:
    """Run inventory and BirdNET bird detection."""
    configure_logging(log_level)
    config = load_config(config_path.resolve())
    inventory_df, _, summary = run_inventory_pipeline(
        config=config,
        stem=stem,
    )
    detections_df = run_birdnet_analysis(
        config=config,
        inventory_df=inventory_df,
    )

    typer.echo(
        f'Wrote BirdNET detections with {detections_df.height} rows to '
        f'{get_birdnet_output_dir(config)}'
    )
    typer.echo('')
    typer.echo(format_inventory_summary(summary))


if __name__ == '__main__':
    app()
