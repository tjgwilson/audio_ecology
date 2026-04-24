from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

from audio_ecology.solar import calculate_solar_metadata


def _minutes_between(left: datetime, right: datetime) -> float:
    """Return the absolute difference between two datetimes in minutes.

    :param left: First datetime.
    :param right: Second datetime.
    :return: Absolute difference in minutes.
    """
    return abs((left - right).total_seconds()) / 60.0


def test_calculate_solar_metadata_matches_exeter_reference_times() -> None:
    """Check sunrise/sunset against published Exeter values.

    Reference values from Time and Date for Exeter, UK:
    https://www.timeanddate.com/sun/uk/exeter?month=4&year=2026
    Apr 17 2026 sunrise 6:15 am, sunset 8:12 pm local time (BST).
    """
    bst = timezone(timedelta(hours=1))
    timestamp = datetime(2026, 4, 17, 13, 0, 0, tzinfo=bst)

    solar_metadata = calculate_solar_metadata(
        timestamp=timestamp,
        latitude=50.7260,
        longitude=-3.5275,
    )

    assert solar_metadata.sunrise_timestamp is not None
    assert solar_metadata.sunset_timestamp is not None

    expected_sunrise = datetime(2026, 4, 17, 6, 15, 0, tzinfo=bst)
    expected_sunset = datetime(2026, 4, 17, 20, 12, 0, tzinfo=bst)

    assert _minutes_between(solar_metadata.sunrise_timestamp, expected_sunrise) < 10
    assert _minutes_between(solar_metadata.sunset_timestamp, expected_sunset) < 10


def test_calculate_solar_metadata_daylight_fields_are_consistent() -> None:
    """Check day/night classification and relative offsets for a daytime record."""
    bst = timezone(timedelta(hours=1))
    timestamp = datetime(2026, 4, 20, 13, 35, 0, tzinfo=bst)

    solar_metadata = calculate_solar_metadata(
        timestamp=timestamp,
        latitude=50.7260,
        longitude=-3.5275,
    )

    assert solar_metadata.is_daylight is True
    assert solar_metadata.minutes_from_sunrise is not None
    assert solar_metadata.minutes_from_sunrise > 0
    assert solar_metadata.minutes_to_sunset is not None
    assert solar_metadata.minutes_to_sunset > 0


def test_calculate_solar_metadata_returns_none_without_required_inputs() -> None:
    """Check that missing timestamp or location yields empty solar metadata."""
    assert calculate_solar_metadata(
        timestamp=None,
        latitude=50.7260,
        longitude=-3.5275,
    ).sunrise_timestamp is None
    assert calculate_solar_metadata(
        timestamp=datetime(2026, 4, 17, 13, 0, 0, tzinfo=timezone.utc),
        latitude=None,
        longitude=-3.5275,
    ).sunrise_timestamp is None
    assert calculate_solar_metadata(
        timestamp=datetime(2026, 4, 17, 13, 0, 0, tzinfo=timezone.utc),
        latitude=50.7260,
        longitude=None,
    ).sunrise_timestamp is None


def test_calculate_solar_metadata_handles_night_record() -> None:
    """Check that night-time records are classified as not daylight."""
    bst = timezone(timedelta(hours=1))
    timestamp = datetime(2026, 4, 17, 23, 0, 0, tzinfo=bst)

    solar_metadata = calculate_solar_metadata(
        timestamp=timestamp,
        latitude=50.7260,
        longitude=-3.5275,
    )

    assert solar_metadata.is_daylight is False
    assert solar_metadata.minutes_to_sunset is not None
    assert solar_metadata.minutes_to_sunset < 0
