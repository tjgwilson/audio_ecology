from __future__ import annotations

from pathlib import Path

from audio_ecology.ingest.discovery import discover_wav_files


def test_discover_wav_files_finds_and_sorts_files(tmp_path: Path) -> None:
    input_dir = tmp_path / 'raw'
    input_dir.mkdir()

    wav_b = input_dir / 'b.WAV'
    wav_a = input_dir / 'a.wav'
    txt_file = input_dir / 'notes.txt'

    wav_b.write_bytes(b'test')
    wav_a.write_bytes(b'test')
    txt_file.write_text('ignore me', encoding='utf-8')

    result = discover_wav_files(input_dir)

    assert result == sorted([wav_a, wav_b])


def test_discover_wav_files_searches_recursively(tmp_path: Path) -> None:
    input_dir = tmp_path / 'raw'
    nested_dir = input_dir / 'nested'
    nested_dir.mkdir(parents=True)

    wav_file = nested_dir / 'x.wav'
    wav_file.write_bytes(b'test')

    result = discover_wav_files(input_dir)

    assert result == [wav_file]