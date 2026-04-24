"""Solar event helpers for ecological timestamp enrichment."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
import math


ZENITH_DEGREES = 90.833


@dataclass(frozen=True)
class SolarMetadata:
    """Solar timing metadata for one timestamp and location."""

    sunrise_timestamp: datetime | None = None
    sunset_timestamp: datetime | None = None
    minutes_from_sunrise: float | None = None
    minutes_to_sunset: float | None = None
    is_daylight: bool | None = None


def _normalise_degrees(value: float) -> float:
    """Wrap an angle into the 0-360 degree range.

    :param value: Angle in degrees.
    :return: Normalized angle in degrees.
    """
    return value % 360.0


def _sun_event_utc(
    event_date: date,
    latitude: float,
    longitude: float,
    *,
    is_sunrise: bool,
) -> datetime | None:
    """Return a sunrise or sunset datetime in UTC using the NOAA approximation.

    :param event_date: Local calendar date for the event.
    :param latitude: Latitude in decimal degrees.
    :param longitude: Longitude in decimal degrees.
    :param is_sunrise: Whether to calculate sunrise instead of sunset.
    :return: Event datetime in UTC, or ``None`` when the event does not occur.
    """
    day_of_year = event_date.timetuple().tm_yday
    lng_hour = longitude / 15.0
    approximate_time = day_of_year + (
        ((6.0 if is_sunrise else 18.0) - lng_hour) / 24.0
    )

    mean_anomaly = (0.9856 * approximate_time) - 3.289
    true_longitude = _normalise_degrees(
        mean_anomaly
        + (1.916 * math.sin(math.radians(mean_anomaly)))
        + (0.020 * math.sin(math.radians(2 * mean_anomaly)))
        + 282.634
    )

    right_ascension = math.degrees(
        math.atan(0.91764 * math.tan(math.radians(true_longitude)))
    )
    right_ascension = _normalise_degrees(right_ascension)

    true_longitude_quadrant = math.floor(true_longitude / 90.0) * 90.0
    right_ascension_quadrant = math.floor(right_ascension / 90.0) * 90.0
    right_ascension += true_longitude_quadrant - right_ascension_quadrant
    right_ascension_hours = right_ascension / 15.0

    sin_declination = 0.39782 * math.sin(math.radians(true_longitude))
    cos_declination = math.cos(math.asin(sin_declination))

    latitude_radians = math.radians(latitude)
    cos_local_hour_angle = (
        math.cos(math.radians(ZENITH_DEGREES))
        - (sin_declination * math.sin(latitude_radians))
    ) / (cos_declination * math.cos(latitude_radians))

    if cos_local_hour_angle > 1.0 or cos_local_hour_angle < -1.0:
        return None

    local_hour_angle = math.degrees(math.acos(cos_local_hour_angle))
    if is_sunrise:
        local_hour_angle = 360.0 - local_hour_angle
    local_hour_angle /= 15.0

    local_mean_time = (
        local_hour_angle
        + right_ascension_hours
        - (0.06571 * approximate_time)
        - 6.622
    )
    universal_time_hours = (local_mean_time - lng_hour) % 24.0

    base_datetime = datetime.combine(event_date, time(0, 0), tzinfo=UTC)
    return base_datetime + timedelta(hours=universal_time_hours)


def calculate_solar_metadata(
    *,
    timestamp: datetime | None,
    latitude: float | None,
    longitude: float | None,
) -> SolarMetadata:
    """Calculate sunrise/sunset-derived fields for a timestamp and location.

    :param timestamp: Reference timestamp for the record.
    :param latitude: Latitude in decimal degrees.
    :param longitude: Longitude in decimal degrees.
    :return: Derived solar metadata with ``None`` values when unavailable.
    """
    if timestamp is None or latitude is None or longitude is None:
        return SolarMetadata()

    sunrise_utc = _sun_event_utc(
        timestamp.date(),
        latitude,
        longitude,
        is_sunrise=True,
    )
    sunset_utc = _sun_event_utc(
        timestamp.date(),
        latitude,
        longitude,
        is_sunrise=False,
    )
    if sunrise_utc is None or sunset_utc is None:
        return SolarMetadata()

    if timestamp.tzinfo is not None:
        sunrise_timestamp = sunrise_utc.astimezone(timestamp.tzinfo)
        sunset_timestamp = sunset_utc.astimezone(timestamp.tzinfo)
    else:
        sunrise_timestamp = sunrise_utc.replace(tzinfo=None)
        sunset_timestamp = sunset_utc.replace(tzinfo=None)

    minutes_from_sunrise = (timestamp - sunrise_timestamp).total_seconds() / 60.0
    minutes_to_sunset = (sunset_timestamp - timestamp).total_seconds() / 60.0

    return SolarMetadata(
        sunrise_timestamp=sunrise_timestamp,
        sunset_timestamp=sunset_timestamp,
        minutes_from_sunrise=minutes_from_sunrise,
        minutes_to_sunset=minutes_to_sunset,
        is_daylight=sunrise_timestamp <= timestamp <= sunset_timestamp,
    )
