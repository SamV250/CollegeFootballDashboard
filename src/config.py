"""Central configuration loader.

Reads the YAML files in ``config/`` and the process environment (via a
``.env`` file when present) and exposes them through a single cached
:class:`Settings` object.  Nothing else in the codebase should read the
YAML or environment directly -- import :func:`get_settings` instead.
"""

from __future__ import annotations

import functools
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

try:  # optional, only used to load a local .env during development
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - dotenv is a convenience only
    pass

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "config"


def _read_yaml(name: str) -> dict[str, Any]:
    path = CONFIG_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing config file: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


@dataclass(frozen=True)
class Settings:
    """Immutable snapshot of all configuration for one run."""

    config: dict[str, Any]
    conferences: dict[str, Any]
    playoff: dict[str, Any]
    env: dict[str, str] = field(default_factory=dict)

    # -- convenience accessors ------------------------------------------------
    @property
    def project_root(self) -> Path:
        return PROJECT_ROOT

    @property
    def processed_dir(self) -> Path:
        d = PROJECT_ROOT / self.config["data"]["processed_dir"]
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def raw_dir(self) -> Path:
        d = PROJECT_ROOT / self.config["data"]["raw_dir"]
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def models_dir(self) -> Path:
        d = PROJECT_ROOT / "models"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def cfbd_api_key(self) -> str | None:
        return self.env.get("CFBD_API_KEY") or os.environ.get("CFBD_API_KEY")

    @property
    def source_priority(self) -> list[str]:
        forced = self.env.get("DATA_SOURCE")
        if forced:
            return [forced]
        return list(self.config["data"]["source_priority"])

    def active_season(self, now: datetime | None = None) -> int:
        """Return the season year that is currently in progress.

        A new college-football season is considered to begin in the month
        given by ``season.season_start_month``.  August 2026 through July
        2027 is therefore the "2026" season.  ``force_season`` overrides
        the calculation entirely.
        """

        forced = self.config["season"].get("force_season")
        if forced:
            return int(forced)
        now = now or datetime.now(UTC)
        start_month = int(self.config["season"]["season_start_month"])
        return now.year if now.month >= start_month else now.year - 1

    def conference_of(self, team: str) -> str:
        for name, meta in self.conferences["conferences"].items():
            if team in meta["teams"]:
                return name
        return "FBS Independents"

    def all_fbs_teams(self) -> list[str]:
        teams: list[str] = []
        for meta in self.conferences["conferences"].values():
            teams.extend(meta["teams"])
        return sorted(teams)

    def team_conference_map(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for name, meta in self.conferences["conferences"].items():
            for team in meta["teams"]:
                out[team] = name
        return out


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load and cache configuration.  Cheap to call repeatedly."""

    env = {k: v for k, v in os.environ.items() if k.isupper()}
    return Settings(
        config=_read_yaml("config.yaml"),
        conferences=_read_yaml("conferences.yaml"),
        playoff=_read_yaml("playoff.yaml"),
        env=env,
    )


def utc_now() -> datetime:
    return datetime.now(UTC)
