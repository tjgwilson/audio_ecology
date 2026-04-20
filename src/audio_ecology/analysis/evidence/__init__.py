"""Window-level detection evidence summaries."""

from audio_ecology.analysis.evidence.noisy_or import (
    BIRD_WINDOW_EVIDENCE_STEM,
    BIRD_WINDOW_EVIDENCE_SCHEMA,
    DETECTION_WINDOW_EVIDENCE_SCHEMA,
    DETECTION_WINDOW_EVIDENCE_STEM,
    build_noisy_or_species_time_period,
    build_noisy_or_species_windows,
    write_noisy_or_species_windows,
)

__all__ = [
    'BIRD_WINDOW_EVIDENCE_SCHEMA',
    'BIRD_WINDOW_EVIDENCE_STEM',
    'DETECTION_WINDOW_EVIDENCE_SCHEMA',
    'DETECTION_WINDOW_EVIDENCE_STEM',
    'build_noisy_or_species_time_period',
    'build_noisy_or_species_windows',
    'write_noisy_or_species_windows',
]
