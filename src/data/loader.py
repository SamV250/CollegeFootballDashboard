"""High-level data access used by every other module.

``refresh()`` walks the configured source priority list, takes the first
source that returns valid data, and upserts it into the local store.
``load_games()`` / ``load_teams()`` read the store.  If the store is
empty (fresh checkout, no credentials) they transparently trigger a demo
refresh so the dashboard always has something to show.
"""

from __future__ import annotations

import logging

import pandas as pd

from src.config import Settings, get_settings
from src.data.sources import SourceError, get_source
from src.data.store import DataStore

log = logging.getLogger(__name__)


def refresh(
    settings: Settings | None = None,
    seasons: list[int] | None = None,
    source: str | None = None,
) -> dict:
    """Fetch fresh data and merge it into the local store.

    Parameters
    ----------
    source:
        Force a specific adapter.  Otherwise the config priority list is
        tried in order and the first success wins.
    """

    settings = settings or get_settings()
    store = DataStore(settings)
    cfg = settings.config
    seasons = seasons or sorted(
        set(cfg["season"]["backtest_seasons"] + [cfg["season"]["current_season"]])
    )
    order = [source] if source else settings.source_priority

    errors: list[str] = []
    for name in order:
        try:
            adapter = get_source(name, settings)
            payload = adapter.fetch(seasons)
            summary = store.upsert(payload, source=name)
            log.info("refresh: source=%s %s", name, summary)
            return {"source": name, "status": "ok", **summary}
        except (SourceError, KeyError, Exception) as exc:  # noqa: BLE001
            errors.append(f"{name}: {exc}")
            store.log_refresh(source=name, status="failed", detail={"error": str(exc)})
            log.warning("refresh: source=%s failed: %s", name, exc)
            continue

    # Every configured source failed. Keep whatever we already had.
    raise RuntimeError(
        "All data sources failed; previous data (if any) left intact.\n  "
        + "\n  ".join(errors)
    )


def _ensure_data(settings: Settings) -> DataStore:
    store = DataStore(settings)
    if not store.has_data():
        log.info("no local data found -- running first-time demo refresh")
        refresh(settings, source="demo")
    return store


def load_games(settings: Settings | None = None) -> pd.DataFrame:
    settings = settings or get_settings()
    return _ensure_data(settings).load_games()


def load_teams(settings: Settings | None = None) -> pd.DataFrame:
    settings = settings or get_settings()
    return _ensure_data(settings).load_teams()


def freshness(settings: Settings | None = None) -> dict:
    settings = settings or get_settings()
    return DataStore(settings).freshness()
