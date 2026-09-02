"""Pluggable data-source adapters.

Each adapter implements :meth:`DataSource.fetch` and returns a dict with
two DataFrames, ``teams`` and ``games``, in the canonical schema
documented in ``DATA_DICTIONARY.md``.  The modeling and UI layers never
import an adapter directly -- they go through :mod:`src.data.loader`,
which walks the configured source priority list.

Design rule: an adapter must *raise* on failure (network error, empty
payload, schema mismatch).  It must never return partial or malformed
data, because the loader treats "raised" as "try the next source" and
"returned" as "trust this".
"""

from __future__ import annotations

import abc
from typing import Any

import pandas as pd
import requests

from src.config import Settings, get_settings

CANONICAL_GAME_COLUMNS = [
    "game_id", "season", "week", "date", "season_type", "home_team",
    "away_team", "home_conference", "away_conference", "neutral_site",
    "completed", "home_points", "away_points",
]


class SourceError(RuntimeError):
    """Raised when a source cannot produce valid data."""


class DataSource(abc.ABC):
    name: str = "base"

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    @abc.abstractmethod
    def fetch(self, seasons: list[int]) -> dict[str, pd.DataFrame]:
        ...

    # -- shared validation -------------------------------------------------
    def _validate(self, payload: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        for key in ("teams", "games"):
            if key not in payload or not isinstance(payload[key], pd.DataFrame):
                raise SourceError(f"{self.name}: missing '{key}' frame")
            if payload[key].empty:
                raise SourceError(f"{self.name}: '{key}' frame is empty")
        missing = set(CANONICAL_GAME_COLUMNS) - set(payload["games"].columns)
        if missing:
            raise SourceError(f"{self.name}: games missing columns {sorted(missing)}")
        games = payload["games"].copy()
        games["date"] = pd.to_datetime(games["date"], utc=True)
        payload["games"] = games
        return payload


class DemoSource(DataSource):
    """Deterministic synthetic universe -- always available."""

    name = "demo"

    def fetch(self, seasons: list[int]) -> dict[str, pd.DataFrame]:
        from src.data.demo import generate_dataset

        payload = generate_dataset(self.settings, seasons=seasons)
        return self._validate(payload)


class LocalSource(DataSource):
    """Read user-supplied CSV/Parquet files from ``data/raw``.

    Expected files: ``data/raw/games.(csv|parquet)`` and
    ``data/raw/teams.(csv|parquet)`` in the canonical schema.
    """

    name = "local"

    def _read(self, stem: str) -> pd.DataFrame | None:
        raw = self.settings.raw_dir
        for ext, reader in ((".parquet", pd.read_parquet), (".csv", pd.read_csv)):
            path = raw / f"{stem}{ext}"
            if path.exists():
                return reader(path)
        return None

    def fetch(self, seasons: list[int]) -> dict[str, pd.DataFrame]:
        games = self._read("games")
        teams = self._read("teams")
        if games is None or teams is None:
            raise SourceError("local: no data/raw/{games,teams}.{parquet,csv} found")
        games = games[games["season"].isin(seasons)]
        return self._validate({"teams": teams, "games": games})


# CFBD advanced-stat path  ->  our canonical per-game metric column.
# Offense keys sit under the "offense" block, "def_*" under "defense".
_CFBD_ADV_MAP = {
    "off_epa_per_play": ("offense", "ppa"),
    "success_rate": ("offense", "successRate"),
    "explosiveness": ("offense", "explosiveness"),
    "line_yards": ("offense", "lineYards"),
    "pass_epa_per_play": ("offense", ("passingPlays", "ppa")),
    "rush_epa_per_play": ("offense", ("rushingPlays", "ppa")),
    "def_epa_per_play": ("defense", "ppa"),
    "def_success_rate": ("defense", "successRate"),
    "def_explosiveness": ("defense", "explosiveness"),
    "def_line_yards": ("defense", "lineYards"),
}


class CFBDSource(DataSource):
    """CollegeFootballData.com REST API adapter (api.collegefootballdata.com).

    Requires ``CFBD_API_KEY``.  Pulls the FBS schedule + results and,
    where available, per-game advanced team stats (PPA/EPA, success rate,
    explosiveness, line yards, on offense and defense).  Any metric the
    API does not return is left as NaN and the feature layer degrades to
    results-derived features.

    Scope: **FBS vs. FBS games only.**  Games against FCS / lower-division
    opponents are excluded (this matches how efficiency systems such as
    SP+/FEI are built); a handful of "cupcake" wins are therefore not
    reflected in records.
    """

    name = "cfbd"
    BASE = "https://api.collegefootballdata.com"
    MAX_WEEK = 20

    def _get(self, path: str, params: dict[str, Any]) -> Any:
        key = self.settings.cfbd_api_key
        if not key:
            raise SourceError("cfbd: CFBD_API_KEY is not set")
        try:
            resp = requests.get(
                f"{self.BASE}{path}", params=params,
                headers={"Authorization": f"Bearer {key}"}, timeout=45,
            )
        except requests.RequestException as exc:
            raise SourceError(f"cfbd: {path} request failed ({exc})") from exc
        if resp.status_code == 401:
            raise SourceError("cfbd: HTTP 401 - CFBD_API_KEY is missing or invalid")
        if resp.status_code != 200:
            raise SourceError(f"cfbd: {path} -> HTTP {resp.status_code}")
        return resp.json()

    # -- games -----------------------------------------------------------
    def _fetch_games(self, season: int) -> tuple[list[dict], dict[str, dict]]:
        raw = self._get("/games", {"year": season, "seasonType": "both"})
        if not raw:
            raise SourceError(f"cfbd: no games returned for {season}")
        rows, teams = [], {}
        for g in raw:
            if g.get("homeClassification") != "fbs" or g.get("awayClassification") != "fbs":
                continue
            home, away = g["homeTeam"], g["awayTeam"]
            hc = g.get("homeConference") or "FBS Independents"
            ac = g.get("awayConference") or "FBS Independents"
            teams.setdefault(home, {"team": home, "conference": hc,
                                    "tier": "unknown", "fbs": True})
            teams.setdefault(away, {"team": away, "conference": ac,
                                    "tier": "unknown", "fbs": True})
            rows.append({
                "game_id": str(g["id"]),
                "season": int(g["season"]),
                "week": int(g["week"]),
                "date": g["startDate"],
                "season_type": g.get("seasonType", "regular"),
                "home_team": home,
                "away_team": away,
                "home_conference": hc,
                "away_conference": ac,
                "neutral_site": bool(g.get("neutralSite", False)),
                "completed": bool(g.get("completed", False)),
                "home_points": g.get("homePoints"),
                "away_points": g.get("awayPoints"),
            })
        return rows, teams

    # -- advanced per-game stats --------------------------------------
    @staticmethod
    def _dig(block: dict, spec) -> Any:
        if isinstance(spec, tuple):
            for k in spec:
                block = (block or {}).get(k, {}) if isinstance(block, dict) else None
            return block if not isinstance(block, dict) else None
        return (block or {}).get(spec)

    def _fetch_advanced(self, season: int, weeks: list[int]) -> dict[str, dict]:
        """Return {(game_id, team_name): {metric: value}}."""

        out: dict[tuple[str, str], dict[str, float]] = {}
        for wk in weeks:
            try:
                rows = self._get("/stats/game/advanced",
                                 {"year": season, "week": wk})
            except SourceError:
                continue
            for r in rows or []:
                gid, team = str(r.get("gameId")), r.get("team")
                if not team:
                    continue
                rec: dict[str, float] = {}
                for metric, (side, spec) in _CFBD_ADV_MAP.items():
                    val = self._dig(r.get(side, {}), spec)
                    if isinstance(val, (int, float)):
                        rec[metric] = float(val)
                if rec:
                    out[(gid, team)] = rec
        return out

    def fetch(self, seasons: list[int]) -> dict[str, pd.DataFrame]:
        game_rows: list[dict] = []
        team_rows: dict[str, dict] = {}
        for season in seasons:
            rows, teams = self._fetch_games(season)
            team_rows.update({k: v for k, v in teams.items() if k not in team_rows})

            weeks = sorted({r["week"] for r in rows}) or list(range(1, self.MAX_WEEK))
            adv = self._fetch_advanced(season, weeks)
            for r in rows:
                for side, col in (("home", "home_team"), ("away", "away_team")):
                    rec = adv.get((r["game_id"], r[col]))
                    if rec:
                        for metric, val in rec.items():
                            r[f"{side}_{metric}"] = val
            game_rows.extend(rows)

        if not game_rows:
            raise SourceError("cfbd: no FBS-vs-FBS games found for requested seasons")
        payload = {
            "teams": pd.DataFrame(list(team_rows.values())),
            "games": pd.DataFrame(game_rows),
        }
        return self._validate(payload)


class ESPNSource(DataSource):
    """Public ESPN scoreboard endpoint -- schedule + results only.

    No advanced metrics.  Used as a fallback so the dashboard keeps
    running when CFBD is unavailable.
    """

    name = "espn"
    SCOREBOARD = ("https://site.api.espn.com/apis/site/v2/sports/"
                  "football/college-football/scoreboard")

    def fetch(self, seasons: list[int]) -> dict[str, pd.DataFrame]:
        rows: list[dict] = []
        teams: dict[str, dict] = {}
        for season in seasons:
            for week in range(1, 16):
                try:
                    data = requests.get(
                        self.SCOREBOARD,
                        params={"year": season, "week": week,
                                "seasontype": 2, "groups": 80, "limit": 400},
                        timeout=30,
                    ).json()
                except Exception as exc:  # noqa: BLE001
                    raise SourceError(f"espn: request failed ({exc})") from exc
                try:
                    for ev in data.get("events", []):
                        comp = ev["competitions"][0]
                        cs = comp["competitors"]
                        home = next(c for c in cs if c["homeAway"] == "home")
                        away = next(c for c in cs if c["homeAway"] == "away")
                        hn = home["team"]["displayName"]
                        an = away["team"]["displayName"]
                        hc = home["team"].get("conferenceId") or "FBS Independents"
                        ac = away["team"].get("conferenceId") or "FBS Independents"
                        for t, c in ((hn, str(hc)), (an, str(ac))):
                            teams.setdefault(t, {"team": t, "conference": c,
                                                 "tier": "unknown", "fbs": True})
                        status = comp["status"]["type"]["completed"]
                        rows.append({
                            "game_id": ev["id"],
                            "season": season,
                            "week": week,
                            "date": ev["date"],
                            "season_type": "regular",
                            "home_team": hn,
                            "away_team": an,
                            "home_conference": str(hc),
                            "away_conference": str(ac),
                            "neutral_site": bool(comp.get("neutralSite", False)),
                            "completed": bool(status),
                            "home_points": float(home["score"]) if status else None,
                            "away_points": float(away["score"]) if status else None,
                        })
                except (KeyError, IndexError, TypeError, StopIteration) as exc:
                    raise SourceError(f"espn: unexpected payload shape ({exc})") from exc
        if not rows:
            raise SourceError("espn: no events parsed")
        return self._validate({
            "teams": pd.DataFrame(list(teams.values())),
            "games": pd.DataFrame(rows),
        })


REGISTRY: dict[str, type[DataSource]] = {
    "demo": DemoSource,
    "local": LocalSource,
    "cfbd": CFBDSource,
    "espn": ESPNSource,
}


def get_source(name: str, settings: Settings | None = None) -> DataSource:
    if name not in REGISTRY:
        raise KeyError(f"Unknown data source '{name}'. Known: {sorted(REGISTRY)}")
    return REGISTRY[name](settings)
