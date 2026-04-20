"""Lightweight process profiling for pipeline stages."""

from __future__ import annotations

from contextlib import contextmanager
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import platform
import resource
import sys
import time
from typing import Iterator

logger = logging.getLogger(__name__)

try:
    import psutil
except ImportError:  # pragma: no cover - exercised when psutil is absent.
    psutil = None


@dataclass
class ProfileRecord:
    """Measurements for one profiled stage."""

    stage: str
    wall_seconds: float
    cpu_seconds: float
    cpu_percent: float
    rss_start_mb: float | None
    rss_end_mb: float | None
    rss_delta_mb: float | None
    peak_rss_start_mb: float
    peak_rss_end_mb: float
    peak_rss_delta_mb: float


def _current_rss_mb() -> float | None:
    """Return current resident memory in MiB when psutil is available."""
    if psutil is None:
        return None

    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def _peak_rss_mb() -> float:
    """Return peak resident memory in MiB.

    macOS reports ru_maxrss in bytes; Linux reports it in KiB.
    """
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == 'darwin':
        return peak_rss / (1024 * 1024)
    return peak_rss / 1024


class ProfileRecorder:
    """Record and write process profile measurements for pipeline stages."""

    def __init__(
        self,
        output_dir: Path,
        run_name: str,
        enabled: bool = True,
    ) -> None:
        """Create a recorder.

        :param output_dir: Directory where profile reports should be written.
        :param run_name: Human-readable run name, used in output file names.
        :param enabled: Whether to record and write profile output.
        """
        self.output_dir = output_dir
        self.run_name = run_name
        self.enabled = enabled
        self.records: list[ProfileRecord] = []
        self.started_at = datetime.now(timezone.utc)

    @contextmanager
    def profile(self, stage: str) -> Iterator[None]:
        """Measure one stage of work."""
        if not self.enabled:
            yield
            return

        logger.info('Profiling stage started: %s', stage)
        wall_start = time.perf_counter()
        cpu_start = time.process_time()
        rss_start = _current_rss_mb()
        peak_rss_start = _peak_rss_mb()

        try:
            yield
        finally:
            wall_seconds = time.perf_counter() - wall_start
            cpu_seconds = time.process_time() - cpu_start
            rss_end = _current_rss_mb()
            peak_rss_end = _peak_rss_mb()
            cpu_percent = (
                (cpu_seconds / wall_seconds) * 100
                if wall_seconds > 0
                else 0.0
            )
            rss_delta = (
                rss_end - rss_start
                if rss_start is not None and rss_end is not None
                else None
            )

            record = ProfileRecord(
                stage=stage,
                wall_seconds=wall_seconds,
                cpu_seconds=cpu_seconds,
                cpu_percent=cpu_percent,
                rss_start_mb=rss_start,
                rss_end_mb=rss_end,
                rss_delta_mb=rss_delta,
                peak_rss_start_mb=peak_rss_start,
                peak_rss_end_mb=peak_rss_end,
                peak_rss_delta_mb=peak_rss_end - peak_rss_start,
            )
            self.records.append(record)
            logger.info(
                'Profiling stage finished: %s wall=%.3fs cpu=%.3fs '
                'cpu=%.1f%% rss_delta=%s peak_rss=%.1f MiB',
                stage,
                record.wall_seconds,
                record.cpu_seconds,
                record.cpu_percent,
                _format_optional_mb(record.rss_delta_mb),
                record.peak_rss_end_mb,
            )

    def write(self) -> tuple[Path, Path] | None:
        """Write profile records to JSON and CSV."""
        if not self.enabled:
            return None

        profile_dir = self.output_dir / 'profiles'
        profile_dir.mkdir(parents=True, exist_ok=True)
        timestamp = self.started_at.strftime('%Y%m%dT%H%M%SZ')
        file_stem = f'{timestamp}_{self.run_name}_profile'

        json_path = profile_dir / f'{file_stem}.json'
        csv_path = profile_dir / f'{file_stem}.csv'

        payload = {
            'run_name': self.run_name,
            'started_at': self.started_at.isoformat(),
            'platform': {
                'system': platform.system(),
                'release': platform.release(),
                'machine': platform.machine(),
                'processor': platform.processor(),
                'python_version': platform.python_version(),
            },
            'records': [asdict(record) for record in self.records],
        }

        json_path.write_text(
            json.dumps(payload, indent=2),
            encoding='utf-8',
        )

        with csv_path.open('w', encoding='utf-8', newline='') as handle:
            field_names = list(ProfileRecord.__dataclass_fields__)
            writer = csv.DictWriter(handle, fieldnames=field_names)
            writer.writeheader()
            for record in self.records:
                writer.writerow(asdict(record))

        logger.info('Wrote profile reports to %s and %s', json_path, csv_path)
        return json_path, csv_path


def _format_optional_mb(value: float | None) -> str:
    """Format an optional MiB value for logs."""
    if value is None:
        return 'n/a'
    return f'{value:.1f} MiB'
