"""Command line interface for the audio ecology pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import polars as pl
import typer

from audio_ecology.analysis.birdnet import (
    BIRDNET_DETECTIONS_STEM,
    get_birdnet_output_dir,
    run_birdnet_analysis,
)
from audio_ecology.analysis.evidence import (
    DETECTION_WINDOW_EVIDENCE_STEM,
    build_noisy_or_species_time_period,
    write_noisy_or_species_windows,
)
from audio_ecology.config import load_config
from audio_ecology.ingest.inventory import (
    build_inventory_records,
    records_to_polars,
    write_inventory_outputs,
)
from audio_ecology.logging_config import configure_pipeline_logging
from audio_ecology.orchestrator import (
    format_inventory_summary,
    summarise_inventory,
)
from audio_ecology.profiling import ProfileRecorder

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
    profile: Annotated[
        bool,
        typer.Option(
            '--profile/--no-profile',
            help='Write profiling reports for this run.',
        ),
    ] = True,
) -> None:
    """Build an inventory of WAV files from a config file."""
    config = load_config(config_path.resolve())
    log_file_path = configure_pipeline_logging(
        config=config,
        level=log_level,
        run_name='inventory',
    )
    profiler = ProfileRecorder(
        output_dir=config.output_dir,
        run_name='inventory',
        enabled=profile,
    )
    with profiler.profile('inventory'):
        records = build_inventory_records(config)
        inventory_df = records_to_polars(records)
        write_inventory_outputs(
            inventory_df=inventory_df,
            output_dir=config.output_dir,
            stem=stem,
            write_csv=config.outputs.write_csv,
        )
    summary = summarise_inventory(inventory_df)
    profile_paths = profiler.write()

    typer.echo(
        f'Wrote inventory with {inventory_df.height} files to '
        f'{config.output_dir}'
    )

    if profile_paths is not None:
        typer.echo(f'Wrote profile reports to {profile_paths[0].parent}')
    if log_file_path is not None:
        typer.echo(f'Wrote log to {log_file_path}')

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
    overwrite_checkpoints: Annotated[
        bool,
        typer.Option(
            '--overwrite-checkpoints',
            help='Re-run files even when BirdNET checkpoints already exist.',
        ),
    ] = False,
    profile: Annotated[
        bool,
        typer.Option(
            '--profile/--no-profile',
            help='Write profiling reports for this run.',
        ),
    ] = True,
) -> None:
    """Run BirdNET bird detection from an existing inventory."""
    config = load_config(config_path.resolve())
    log_file_path = configure_pipeline_logging(
        config=config,
        level=log_level,
        run_name='birds',
    )
    inventory_path = config.output_dir / f'{inventory_stem}.parquet'
    if not inventory_path.exists():
        raise typer.BadParameter(
            f'Inventory parquet not found: {inventory_path}. '
            'Run the inventory stage first.'
        )

    profiler = ProfileRecorder(
        output_dir=config.output_dir,
        run_name='birds',
        enabled=profile,
    )
    with profiler.profile('read_inventory'):
        inventory_df = pl.read_parquet(inventory_path)
    with profiler.profile('birdnet_analysis'):
        detections_df = run_birdnet_analysis(
            config=config,
            inventory_df=inventory_df,
            overwrite_checkpoints=overwrite_checkpoints,
        )
    profile_paths = profiler.write()

    typer.echo(
        f'Wrote BirdNET detections with {detections_df.height} rows to '
        f'{get_birdnet_output_dir(config)}'
    )
    if profile_paths is not None:
        typer.echo(f'Wrote profile reports to {profile_paths[0].parent}')
    if log_file_path is not None:
        typer.echo(f'Wrote log to {log_file_path}')


@app.command('detection-windows')
def detection_windows(
    config_path: Annotated[
        Path,
        typer.Argument(help='Path to the YAML configuration file.'),
    ],
    detections_stem: Annotated[
        str,
        typer.Option(
            '--detections-stem',
            help='Detection file stem used only when no generic path is configured.',
        ),
    ] = BIRDNET_DETECTIONS_STEM,
    output_stem: Annotated[
        str,
        typer.Option(
            '--output-stem',
            help='Base file stem for window evidence outputs.',
        ),
    ] = DETECTION_WINDOW_EVIDENCE_STEM,
    log_level: Annotated[
        str,
        typer.Option(
            '--log-level',
            help='Logging level: INFO or DEBUG.',
        ),
    ] = 'INFO',
) -> None:
    """Aggregate detections into window-level noisy-OR evidence."""
    config = load_config(config_path.resolve())
    if (
        config.detection_uncertainty.start_time is None
        or config.detection_uncertainty.end_time is None
    ):
        raise typer.BadParameter(
            'Set detection_uncertainty.start_time and '
            'detection_uncertainty.end_time in the config.'
        )

    log_file_path = configure_pipeline_logging(
        config=config,
        level=log_level,
        run_name='detection_windows',
    )
    birdnet_output_dir = get_birdnet_output_dir(config)
    detections_path = (
        config.detection_uncertainty.detections_path
        if config.detection_uncertainty.detections_path is not None
        else birdnet_output_dir / f'{detections_stem}.parquet'
    )
    if not detections_path.exists():
        raise typer.BadParameter(
            f'Detections parquet not found: {detections_path}. '
            'Run the relevant detection stage first.'
        )

    detections_df = pl.read_parquet(detections_path)
    evidence_df = build_noisy_or_species_time_period(
        detections_df=detections_df,
        start_time=config.detection_uncertainty.start_time,
        end_time=config.detection_uncertainty.end_time,
        config=config.detection_uncertainty,
    )
    write_noisy_or_species_windows(
        evidence_df=evidence_df,
        output_dir=config.detection_uncertainty.output_dir or detections_path.parent,
        stem=output_stem,
        write_csv=config.outputs.write_csv,
    )

    typer.echo(
        f'Wrote window evidence with {evidence_df.height} rows to '
        f'{birdnet_output_dir}'
    )
    if log_file_path is not None:
        typer.echo(f'Wrote log to {log_file_path}')


if __name__ == '__main__':
    app()
