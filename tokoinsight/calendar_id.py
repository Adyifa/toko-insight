"""Indonesian retail calendar: Lebaran (Idul Fitri) moves about 11 days earlier every year.

Lebaran is the biggest demand event for an Indonesian minimarket, and it follows the Hijri
calendar, so a plain "same month last year" model puts the peak in the wrong month. The
forecasting code uses these dates to adjust for it (see forecasting.MovingHoliday).

Dates are the first day of Idul Fitri as set by the Indonesian government (SKB 3 Menteri).
2027 and 2028 are the announced / astronomical estimates and may shift by a day.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

LEBARAN = {
    2021: date(2021, 5, 13),
    2022: date(2022, 5, 2),
    2023: date(2023, 4, 22),
    2024: date(2024, 4, 10),
    2025: date(2025, 3, 31),
    2026: date(2026, 3, 20),
    2027: date(2027, 3, 10),
    2028: date(2028, 2, 27),
}

RAMADAN_DAYS = 30  # the fasting month is 29 or 30 days; 30 is close enough for demand modelling


def lebaran(year: int) -> date:
    try:
        return LEBARAN[year]
    except KeyError:
        raise ValueError(f"No Lebaran date for {year}; add it to calendar_id.LEBARAN") from None


def ramadan_window(year: int) -> tuple[date, date]:
    """Inclusive (first, last) day of Ramadan that ends with Lebaran in `year`."""
    end = lebaran(year) - timedelta(days=1)
    return end - timedelta(days=RAMADAN_DAYS - 1), end


def lebaran_week(year: int) -> tuple[date, date]:
    """Lebaran day plus the following mudik/holiday days, when minimarket traffic drops."""
    start = lebaran(year)
    return start, start + timedelta(days=6)


def school_window(year: int) -> tuple[date, date]:
    """Back-to-school shopping before the new school year in mid July."""
    return date(year, 6, 20), date(year, 7, 20)


def year_end_window(year: int) -> tuple[date, date]:
    """Christmas and New Year's Eve parties."""
    return date(year, 12, 10), date(year, 12, 31)


def in_window(day: date, window: tuple[date, date]) -> bool:
    return window[0] <= day <= window[1]


def is_rainy_season(day: date) -> bool:
    return day.month in (11, 12, 1, 2)


def is_dry_season(day: date) -> bool:
    return day.month in (6, 7, 8, 9)


@dataclass(frozen=True)
class Event:
    name: str
    start: date
    end: date


def upcoming_events(after: date, within_days: int = 150) -> list[Event]:
    """Seasonal events that start (or are running) between `after` and `after + within_days`."""
    horizon = after + timedelta(days=within_days)
    events: list[Event] = []
    for year in (after.year, after.year + 1):
        windows = [("school", school_window(year)), ("year_end", year_end_window(year))]
        if year in LEBARAN:
            windows += [("ramadan", ramadan_window(year)), ("lebaran", lebaran_week(year))]
        for name, (start, end) in windows:
            if end >= after and start <= horizon:
                events.append(Event(name, start, end))
    return sorted(events, key=lambda e: e.start)


def day_labels(day: date) -> dict[str, int]:
    """Calendar flags stored in the `calendar` table so SQL questions can filter on them."""
    year = day.year
    ram = ramadan_window(year) if year in LEBARAN else None
    leb = lebaran_week(year) if year in LEBARAN else None
    days_to = (LEBARAN[year] - day).days if year in LEBARAN else None
    if days_to is not None and days_to < -180 and (year + 1) in LEBARAN:
        days_to = (LEBARAN[year + 1] - day).days
    return {
        "is_ramadan": int(bool(ram and in_window(day, ram))),
        "is_lebaran_week": int(bool(leb and in_window(day, leb))),
        "is_school_season": int(in_window(day, school_window(year))),
        "is_year_end": int(in_window(day, year_end_window(year))),
        "is_rainy_season": int(is_rainy_season(day)),
        "days_to_lebaran": days_to if days_to is not None else 9999,
    }
