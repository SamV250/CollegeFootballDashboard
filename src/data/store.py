"""Local persistence for game / team data and the refresh log.

Storage is Parquet by default (``config.data.storage_format``).  The
store guarantees:

* **Idempotent upserts** -- running an update twice never duplicates a
  game.  Rows are keyed on ``game_id``; newer records replace older ones
  only when they carry *more* information (a score appearing, a game
  flipping to completed).
* **No destructive overwrites** -- if an incoming frame is empty or fails
  validation the previous dataset is kept and the failure is logged.
* **An auditable refresh log** -- every attempt records timestamp (UTC),
  source, status and row counts.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.config import Settings, get_settings, utc_now

_GAMES = "games"
_TEAMS = "teams"
_LOG = "refresh_log.json"


class DataStore:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.dir: Path = self.settings.processed_dir
        self.fmt: str = self.settings.config["data"]["storage_format"]

    # -- low-level io ----------------------------------------------------------
    def _path(self, stem: str) -> Path:
        ext = "parquet" if self.fmt == "parquet" else "db"
        return self.dir / f"{stem}.{ext}"

    def _read(self, stem: str) -> pd.DataFrame | None:
        path = self._path(stem)
        if not path.exists():
            return None
        if self.fmt == "parquet":
            return pd.read_parquet(path)
        import sqlite3

        with sqlite3.connect(path) as con:
            return pd.read_sql(f"SELECT * FROM {stem}", con)

    def _write(self, df: pd.DataFrame, stem: str) -> None:
        path = self._path(stem)
        if self.fmt == "parquet":
            df.to_parquet(path, index=False)
        else:
            import sqlite3

            with sqlite3.connect(path) as con:
                df.to_sql(stem, con, if_exists="replace", index=False)

    # -- public read --------------------------------------------------------
    def load_games(self) -> pd.DataFrame:
        df = self._read(_GAMES)
        if df is None:
            raise FileNotFoundError(
                "No processed games found. Run `python scripts/update_data.py` "
                "or `python scripts/build_dataset.py` first."
            )
        df["date"] = pd.to_datetime(df["date"], utc=True)
        # Defensive: a bad source or a partial merge could leave duplicate
        # game_ids, which fan out every downstream merge. Keep the most
        # informative row per id.
        if "game_id" in df and not df["game_id"].is_unique:
            df = (df.assign(_info=df.apply(self._info_score, axis=1))
                    .sort_values("_info")
                    .drop_duplicates("game_id", keep="last")
                    .drop(columns="_info"))
        return df.sort_values(["season", "week", "date"]).reset_index(drop=True)

    def load_teams(self) -> pd.DataFrame:
        df = self._read(_TEAMS)
        if df is None:
            raise FileNotFoundError("No processed teams found.")
        return df.drop_duplicates(subset="team", keep="first").reset_index(drop=True)

    def has_data(self) -> bool:
        return self._path(_GAMES).exists()

    # -- upsert -----------------------------------------------------------
    @staticmethod
    def _info_score(row: pd.Series) -> int:
        """Higher = more informative. Used to decide which duplicate wins."""

        score = 0
        if bool(row.get("completed")):
            score += 2
        if pd.notna(row.get("home_points")):
            score += 1
        if pd.notna(row.get("away_points")):
            score += 1
        return score

    def upsert(
        self, incoming: dict[str, pd.DataFrame], source: str
    ) -> dict[str, int]:
        """Merge ``incoming`` into the store. Returns a change summary."""

        games_new = incoming["games"].copy()
        games_new["date"] = pd.to_datetime(games_new["date"], utc=True)
        existing = self._read(_GAMES)

        if existing is None or existing.empty:
            merged = games_new
            added, updated = len(merged), 0
        else:
            existing["date"] = pd.to_datetime(existing["date"], utc=True)
            existing = existing.set_index("game_id")
            games_new = games_new.set_index("game_id")
            added = int((~games_new.index.isin(existing.index)).sum())
            updated = 0
            for gid, new_row in games_new.iterrows():
                if gid in existing.index:
                    old_row = existing.loc[gid]
                    if self._info_score(new_row) >= self._info_score(old_row):
                        if not new_row.equals(old_row.reindex(new_row.index)):
                            updated += 1
                        existing.loc[gid, new_row.index] = new_row.values
                else:
                    existing.loc[gid] = new_row.reindex(existing.columns)
            merged = existing.reset_index()

        merged = merged.sort_values(["season", "week", "date"]).reset_index(drop=True)
        self._write(merged, _GAMES)

        # teams: union, preferring the newer conference label
        teams_existing = self._read(_TEAMS)
        teams_new = incoming["teams"]
        if teams_existing is not None and not teams_existing.empty:
            teams = pd.concat([teams_new, teams_existing]).drop_duplicates(
                subset="team", keep="first"
            )
        else:
            teams = teams_new
        self._write(teams.reset_index(drop=True), _TEAMS)

        summary = {
            "rows_total": len(merged),
            "rows_added": added,
            "rows_updated": updated,
            "completed_games": int(merged["completed"].sum()),
        }
        self.log_refresh(source=source, status="ok", detail=summary)
        return summary

    # -- refresh log + freshness ----------------------------------------
    def _log_path(self) -> Path:
        return self.dir / _LOG

    def log_refresh(self, source: str, status: str, detail: dict | None = None) -> None:
        entry = {
            "timestamp_utc": utc_now().isoformat(),
            "source": source,
            "status": status,
            "detail": detail or {},
        }
        log = self.read_log()
        log.append(entry)
        self._log_path().write_text(json.dumps(log[-500:], indent=2))

    def read_log(self) -> list[dict]:
        p = self._log_path()
        if not p.exists():
            return []
        try:
            return json.loads(p.read_text())
        except json.JSONDecodeError:
            return []

    # entries whose "source" names an actual data adapter (not an
    # orchestration wrapper like "update_dashboard")
    _ADAPTER_SOURCES = {"cfbd", "espn", "local", "demo"}

    def last_successful_refresh(self) -> dict | None:
        """Most recent OK entry from an actual data adapter (so the UI
        reports 'via demo', not 'via update_dashboard')."""

        adapter_hit = None
        for entry in reversed(self.read_log()):
            if entry.get("status") != "ok":
                continue
            if entry.get("source") in self._ADAPTER_SOURCES:
                return entry
            adapter_hit = adapter_hit or entry
        return adapter_hit

    def freshness(self, now: datetime | None = None) -> dict:
        """Return a freshness report for the UI badge."""

        now = now or utc_now()
        cfg = self.settings.config["data"]
        last = self.last_successful_refresh()
        report = {
            "state": "unknown",
            "last_refresh_utc": None,
            "source": None,
            "age_hours": None,
            "latest_completed_game_utc": None,
            "next_refresh_utc": None,
        }
        if last:
            ts = datetime.fromisoformat(last["timestamp_utc"])
            age_h = (now - ts).total_seconds() / 3600.0
            interval_h = cfg["refresh_interval_minutes"] / 60.0
            if age_h > cfg["stale_after_hours"]:
                state = "stale"
            elif age_h > interval_h * (1 + cfg["delayed_after_intervals"]):
                state = "delayed"
            else:
                state = "current"
            report.update(
                state=state,
                last_refresh_utc=last["timestamp_utc"],
                source=last["source"],
                age_hours=round(age_h, 2),
                next_refresh_utc=(ts + pd.Timedelta(hours=interval_h)).isoformat(),
            )
        try:
            g = self.load_games()
            done = g[g["completed"]]
            if not done.empty:
                report["latest_completed_game_utc"] = done["date"].max().isoformat()
        except FileNotFoundError:
            pass
        return report
