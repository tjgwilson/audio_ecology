"""Command line interface for the audio ecology pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import polars as pl
import typer

from audio_ecology.analysis.birdnet import (
    get_birdnet_output_dir,
    run_birdnet_analysis,
)
from audio_ecology.config import load_config
from audio_ecology.ingest.inventory import (
    build_inventory_records,
    records_to_polars,
    write_inventory_outputs,
)
from audio_ecology.logging_config import configure_logging
from audio_ecology.orchestrator import format_inventory_summary, summarise_inventory

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
    records = build_inventory_records(config)
    inventory_df = records_to_polars(records)
    write_inventory_outputs(
        inventory_df=inventory_df,
        output_dir=config.output_dir,
        stem=stem,
    )
    summary = summarise_inventory(inventory_df)

    typer.echo(
        f'Wrote inventory with {inventory_df.height} files to '
        f'{config.output_dir}'
    )

    typer.echo('')
    typer.echo(format_inventory_summary(summary))


@app.command()
def birds(
    config_path: Annotated[
        Path,
        typer.Argument(help='Path to the YAML configuration file.'),
    ],
    inventory_stem: Annotated[
        str,
        typer.Option(
            '--inventory-stem',
            help='Inventory file stem in the configured output directory.',
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
    """Run BirdNET bird detection from an existing inventory."""
    configure_logging(log_level)
    config = load_config(config_path.resolve())
    inventory_path = config.output_dir / f'{inventory_stem}.parquet'
    if not inventory_path.exists():
        raise typer.BadParameter(
            f'Inventory parquet not found: {inventory_path}. '
            'Run the inventory stage first.'
        )

    inventory_df = pl.read_parquet(inventory_path)
    detections_df = run_birdnet_analysis(
        config=config,
        inventory_df=inventory_df,
    )

    typer.echo(
        f'Wrote BirdNET detections with {detections_df.height} rows to '
        f'{get_birdnet_output_dir(config)}'
    )


if __name__ == '__main__':
    app()
