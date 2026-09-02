"""A transparent Elo rating system for college football.

Elo is our first baseline and also a *feature* for the primary model
(``elo_diff``).  Properties:

* Home-field advantage added as a rating bump before each game.
* Optional margin-of-victory multiplier (autocorrelation-corrected, the
  standard 538 form) so blowouts move ratings more than one-score wins.
* Between-season regression toward the mean so last year's rating carries
  over only partially.
* New / low-history FBS teams and non-FBS opponents get configurable
  starting ratings, satisfying the "reasonable preseason prior" rule.

``fit`` walks games in chronological order and records the pre-game
rating of each side, which is exactly the leakage-safe quantity the
feature layer needs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.config import Settings, get_settings


@dataclass
class EloModel:
    settings: Settings = field(default_factory=get_settings)
    ratings: dict[str, float] = field(default_factory=dict)
    history: pd.DataFrame | None = None

    def __post_init__(self) -> None:
        e = self.settings.config["elo"]
        self.k = float(e["k_factor"])
        self.hfa = float(e["home_advantage"])
        self.use_mov = bool(e["mov_multiplier"])
        self.regression = float(e["preseason_regression"])
        self.mean = float(e["mean_rating"])
        self.new_team = float(e["new_team_rating"])
        self.fcs = float(e["fcs_opponent_rating"])

    # -- helpers --------------------------------------------------------------
    def _get(self, team: str, fbs_teams: set[str]) -> float:
        if team in self.ratings:
            return self.ratings[team]
        self.ratings[team] = self.new_team if team in fbs_teams else self.fcs
        return self.ratings[team]

    @staticmethod
    def expected(r_a: float, r_b: float) -> float:
        return 1.0 / (1.0 + 10 ** ((r_b - r_a) / 400.0))

    def _mov_mult(self, margin: float, elo_diff_winner: float) -> float:
        if not self.use_mov:
            return 1.0
        return math.log(abs(margin) + 1.0) * (2.2 / ((elo_diff_winner * 0.001) + 2.2))

    # -- fitting -----------------------------------------------------------
    def fit(self, games: pd.DataFrame, teams: pd.DataFrame | None = None) -> "EloModel":
        fbs = set(teams["team"]) if teams is not None else set(games["home_team"]) | set(
            games["away_team"]
        )
        games = games.sort_values(["date"]).reset_index(drop=True)
        rows = []
        last_season: int | None = None
        for g in games.itertuples():
            if last_season is not None and g.season != last_season:
                self._new_season()
            last_season = g.season

            home, away = g.home_team, g.away_team
            r_home = self._get(home, fbs)
            r_away = self._get(away, fbs)
            hfa = 0.0 if g.neutral_site else self.hfa
            p_home = self.expected(r_home + hfa, r_away)

            rows.append({
                "game_id": g.game_id,
                "season": g.season,
                "week": g.week,
                "home_elo_pre": r_home,
                "away_elo_pre": r_away,
                "elo_diff": (r_home + hfa) - r_away,
                "elo_home_win_prob": p_home,
            })

            if getattr(g, "completed", False) and not pd.isna(g.home_points):
                margin = g.home_points - g.away_points
                s_home = 1.0 if margin > 0 else 0.0
                winner_diff = (r_home - r_away) if margin > 0 else (r_away - r_home)
                mult = self._mov_mult(margin, winner_diff)
                delta = self.k * mult * (s_home - p_home)
                self.ratings[home] = r_home + delta
                self.ratings[away] = r_away - delta

        self.history = pd.DataFrame(rows)
        return self

    def _new_season(self) -> None:
        for t, r in list(self.ratings.items()):
            self.ratings[t] = r + self.regression * (self.mean - r)

    # -- inference -------------------------------------------------------
    def win_probability(
        self, home: str, away: str, neutral: bool = False
    ) -> float:
        r_home = self.ratings.get(home, self.new_team)
        r_away = self.ratings.get(away, self.new_team)
        hfa = 0.0 if neutral else self.hfa
        return float(self.expected(r_home + hfa, r_away))

    def rating(self, team: str) -> float:
        return float(self.ratings.get(team, self.new_team))

    def rating_table(self) -> pd.DataFrame:
        return (
            pd.DataFrame({"team": list(self.ratings), "elo": list(self.ratings.values())})
            .sort_values("elo", ascending=False)
            .reset_index(drop=True)
        )

    def predicted_margin(self, home: str, away: str, neutral: bool = False) -> float:
        """Rough points spread implied by the Elo gap (~25 Elo == 1 pt)."""

        r_home = self.ratings.get(home, self.new_team)
        r_away = self.ratings.get(away, self.new_team)
        hfa = 0.0 if neutral else self.hfa
        return ((r_home + hfa) - r_away) / 25.0
