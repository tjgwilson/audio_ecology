from __future__ import annotations

import csv
import json
from pathlib import Path

from audio_ecology.profiling import ProfileRecorder


def test_profile_recorder_writes_json_and_csv(tmp_path: Path) -> None:
    recorder = ProfileRecorder(
        output_dir=tmp_path,
        run_name='test_run',
    )

    with recorder.profile('test_stage'):
        _ = sum(range(10))

    profile_paths = recorder.write()

    assert profile_paths is not None
    json_path, csv_path = profile_paths
    assert json_path.exists()
    assert csv_path.exists()

    payload = json.loads(json_path.read_text(encoding='utf-8'))
    assert payload['run_name'] == 'test_run'
    assert payload['records'][0]['stage'] == 'test_stage'
    assert payload['records'][0]['wall_seconds'] >= 0
    assert payload['records'][0]['cpu_seconds'] >= 0
    assert payload['records'][0]['peak_rss_end_mb'] >= 0

    with csv_path.open('r', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))

    assert rows[0]['stage'] == 'test_stage'


def test_profile_recorder_can_be_disabled(tmp_path: Path) -> None:
    recorder = ProfileRecorder(
        output_dir=tmp_path,
        run_name='test_run',
        enabled=False,
    )

    with recorder.profile('test_stage'):
        _ = sum(range(10))

    assert recorder.records == []
    assert recorder.write() is None
    assert not (tmp_path / 'profiles').exists()
