"""Small formatting helpers shared across pages."""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

TZ_CHOICES = {
    "US/Eastern": "America/New_York",
    "US/Central": "America/Chicago",
    "US/Mountain": "America/Denver",
    "US/Pacific": "America/Los_Angeles",
    "UTC": "UTC",
}


def pct(x: float, digits: int = 0) -> str:
    if x is None or pd.isna(x):
        return "—"
    return f"{x * 100:.{digits}f}%"


def signed(x: float, digits: int = 1) -> str:
    if x is None or pd.isna(x):
        return "—"
    return f"{x:+.{digits}f}"


def fmt_dt(iso_or_dt, tz_name: str = "America/New_York") -> str:
    if iso_or_dt is None:
        return "—"
    if isinstance(iso_or_dt, str):
        try:
            dt = datetime.fromisoformat(iso_or_dt)
        except ValueError:
            return iso_or_dt
    else:
        dt = iso_or_dt
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    try:
        local = dt.astimezone(_zone(tz_name))
    except Exception:
        local = dt
    return local.strftime("%b %d, %Y  %I:%M %p %Z")


def _zone(name: str):
    from zoneinfo import ZoneInfo

    return ZoneInfo(TZ_CHOICES.get(name, name))


def humanize_hours(h: float | None) -> str:
    if h is None:
        return "—"
    if h < 1:
        return f"{int(h * 60)} min ago"
    if h < 48:
        return f"{h:.1f} hours ago"
    return f"{h / 24:.1f} days ago"


def team_record(row: pd.Series) -> str:
    return f"{int(row.get('wins', 0))}-{int(row.get('losses', 0))}"
