"""Run window-level detection uncertainty summaries."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import polars as pl

from audio_ecology.analysis.birdnet import (
    BIRDNET_DETECTION_SCHEMA,
    get_birdnet_detection_dataset_dir,
)
from audio_ecology.analysis.evidence import (
    DETECTION_WINDOW_EVIDENCE_STEM,
    build_noisy_or_species_time_period,
    write_noisy_or_species_windows,
)
from audio_ecology.analysis.storage import load_detection_dataframe
from audio_ecology.config import load_config
from audio_ecology.logging_config import configure_pipeline_logging

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = SCRIPT_DIR / 'config_files' / 'wyke_lodge.yaml'


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    :return: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description='Aggregate detections into window-level evidence.'
    )
    parser.add_argument(
        'config_path',
        nargs='?',
        default=DEFAULT_CONFIG_PATH,
        type=Path,
        help='Path to the pipeline config YAML.',
    )
    parser.add_argument(
        '--detections-path',
        default=None,
        type=Path,
        help=(
            'Path to a canonical detections parquet. Defaults to '
            'detection_uncertainty.detections_path, then BirdNET detections.'
        ),
    )
    parser.add_argument(
        '--output-dir',
        default=None,
        type=Path,
        help=(
            'Directory for evidence outputs. Defaults to '
            'detection_uncertainty.output_dir, then the detections directory.'
        ),
    )
    parser.add_argument(
        '--output-stem',
        default=None,
        help='Window evidence file stem. Defaults to config, then generic default.',
    )
    parser.add_argument(
        '--log-level',
        default='INFO',
        help='Logging level: INFO or DEBUG.',
    )
    return parser.parse_args()


def default_birdnet_detections_path(config) -> Path:
    """Return the canonical BirdNET detections dataset path.

    :param config: Loaded pipeline configuration.
    :return: BirdNET detection dataset directory.
    """
    return get_birdnet_detection_dataset_dir(config)


def resolve_detections_path(config, args: argparse.Namespace) -> Path:
    """Resolve the detections dataset path for this evidence run.

    :param config: Loaded pipeline configuration.
    :param args: Parsed command-line arguments.
    :return: Detection dataset path to read.
    """
    if args.detections_path is not None:
        return args.detections_path.resolve()

    if config.detection_uncertainty.detections_path is not None:
        return config.detection_uncertainty.detections_path

    detections_path = default_birdnet_detections_path(config)
    logging.getLogger(__name__).info(
        'No generic detection_uncertainty.detections_path configured; '
        'falling back to BirdNET detections at %s',
        detections_path,
    )
    return detections_path


def resolve_output_dir(detections_path: Path, config, args: argparse.Namespace) -> Path:
    """Resolve the output directory for evidence files.

    :param detections_path: Detection dataset path used for this run.
    :param config: Loaded pipeline configuration.
    :param args: Parsed command-line arguments.
    :return: Output directory for evidence files.
    """
    if args.output_dir is not None:
        return args.output_dir.resolve()

    if config.detection_uncertainty.output_dir is not None:
        return config.detection_uncertainty.output_dir

    return detections_path.parent


def resolve_output_stem(config, args: argparse.Namespace) -> str:
    """Resolve the output stem for evidence files.

    :param config: Loaded pipeline configuration.
    :param args: Parsed command-line arguments.
    :return: Output file stem for evidence files.
    """
    if args.output_stem is not None:
        return str(args.output_stem)

    if config.detection_uncertainty.output_stem:
        return config.detection_uncertainty.output_stem

    return DETECTION_WINDOW_EVIDENCE_STEM


def main() -> None:
    """Run window-level detection uncertainty summaries.

    :return: ``None``.
    :raises ValueError: If the configured window bounds are incomplete.
    :raises FileNotFoundError: If the configured detection dataset does not exist.
    """
    args = parse_args()
    config = load_config(args.config_path.resolve())
    uncertainty_config = config.detection_uncertainty
    resolved_start_time = uncertainty_config.resolved_start_time
    resolved_end_time = uncertainty_config.resolved_end_time
    if resolved_start_time is None or resolved_end_time is None:
        raise ValueError(
            'Set detection_uncertainty.start_time together with end_time '
            'or duration_s in the config.'
        )

    log_file_path = configure_pipeline_logging(
        config=config,
        level=args.log_level,
        run_name='detection_windows',
    )

    logger = logging.getLogger(__name__)
    detections_path = resolve_detections_path(config=config, args=args)
    output_dir = resolve_output_dir(
        detections_path=detections_path,
        config=config,
        args=args,
    )
    output_stem = resolve_output_stem(config=config, args=args)

    logger.info('Reading detections from %s', detections_path)
    if not detections_path.exists():
        raise FileNotFoundError(
            f'Detections parquet not found: {detections_path}. '
            'Run the relevant detection stage first or set '
            'detection_uncertainty.detections_path.'
        )

    detections_df = load_detection_dataframe(
        detections_path,
        schema=BIRDNET_DETECTION_SCHEMA,
    )
    evidence_df = build_noisy_or_species_time_period(
        detections_df=detections_df,
        start_time=resolved_start_time,
        end_time=resolved_end_time,
        config=uncertainty_config,
    )
    parquet_path, csv_path = write_noisy_or_species_windows(
        evidence_df=evidence_df,
        output_dir=output_dir,
        stem=output_stem,
        write_csv=config.outputs.write_csv,
    )

    message = (
        f'Wrote {evidence_df.height} window evidence rows to {parquet_path} '
        f'for {resolved_start_time.isoformat()} to '
        f'{resolved_end_time.isoformat()} '
        f'event_gap_s={uncertainty_config.event_gap_s:g}'
    )
    if csv_path is not None:
        message = f'{message} and {csv_path}'
    print(message)
    if log_file_path is not None:
        print(f'Wrote log to {log_file_path}')


if __name__ == '__main__':
    main()
