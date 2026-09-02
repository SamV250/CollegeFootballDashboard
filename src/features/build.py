"""Assemble the model-ready matchup feature matrix.

For every game (played or upcoming) we join:

* the **home** team's pre-game rolling ratings,
* the **away** team's pre-game rolling ratings,
* the pre-game **Elo** gap,
* schedule context (neutral site, rest days),

then express most features as *home-minus-away* differences plus a few
explicit matchup cross-terms (e.g. home offense vs away defense).  A
positive value always favours the home team.

Preseason priors: when a team has fewer than
``config.features.min_games_for_stats`` completed games in the current
season, its rolling ratings are blended toward a prior derived from the
prior season's opponent-adjusted rating (regressed toward the mean) or,
for brand-new teams, a tier baseline.  The blend weight moves smoothly
from "all prior" to "all in-season" as games accumulate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import Settings, get_settings
from src.features.team_stats import (
    rolling_team_features,
    team_game_log,
)
from src.models.elo import EloModel

# rolling rating columns (all "higher == better for that team")
RATING_COLS = [
    "exp_net_margin", "ewm_net_margin", "exp_points_for", "exp_points_against",
    "exp_won", "exp_turnover_margin", "adj_net_rating", "sos_to_date",
    "exp_off_epa_per_play", "exp_def_epa_prevention",
    "exp_success_rate", "exp_def_success_prevention",
    "exp_explosiveness", "exp_def_explosiveness_prevention",
    "exp_pass_epa_per_play", "exp_rush_epa_per_play",
    "exp_havoc", "exp_def_havoc_prevention",
    "exp_sack_rate", "exp_pass_pro_prevention",
    "exp_finish_pts_per_opp", "exp_def_finish_prevention",
    "exp_line_yards", "exp_def_line_prevention",
    "exp_st_epa", "exp_pace",
]

# feature name -> (home col, away col, "diff" | "cross")
# "diff"  -> home_value - away_value
# "cross" -> already directional; used for O-vs-D matchup terms
DIFF_FEATURES = {
    "elo_diff": None,  # supplied directly by Elo
    "adj_rating_diff": ("adj_net_rating", "adj_net_rating"),
    "recent_form_diff": ("ewm_net_margin", "ewm_net_margin"),
    "season_margin_diff": ("exp_net_margin", "exp_net_margin"),
    "win_pct_diff": ("exp_won", "exp_won"),
    "sos_diff": ("sos_to_date", "sos_to_date"),
    "turnover_margin_diff": ("exp_turnover_margin", "exp_turnover_margin"),
    "off_epa_diff": ("exp_off_epa_per_play", "exp_off_epa_per_play"),
    "def_epa_prevention_diff": ("exp_def_epa_prevention", "exp_def_epa_prevention"),
    "success_rate_diff": ("exp_success_rate", "exp_success_rate"),
    "def_success_prevention_diff": ("exp_def_success_prevention", "exp_def_success_prevention"),
    "explosiveness_diff": ("exp_explosiveness", "exp_explosiveness"),
    "def_explosiveness_prevention_diff": ("exp_def_explosiveness_prevention", "exp_def_explosiveness_prevention"),
    "pass_epa_diff": ("exp_pass_epa_per_play", "exp_pass_epa_per_play"),
    "rush_epa_diff": ("exp_rush_epa_per_play", "exp_rush_epa_per_play"),
    "havoc_diff": ("exp_havoc", "exp_havoc"),
    "pass_protection_diff": ("exp_pass_pro_prevention", "exp_pass_pro_prevention"),
    "pass_rush_diff": ("exp_sack_rate", "exp_sack_rate"),
    "red_zone_diff": ("exp_finish_pts_per_opp", "exp_finish_pts_per_opp"),
    "red_zone_defense_diff": ("exp_def_finish_prevention", "exp_def_finish_prevention"),
    "line_yards_diff": ("exp_line_yards", "exp_line_yards"),
    "special_teams_diff": ("exp_st_epa", "exp_st_epa"),
}
CROSS_FEATURES = {
    # home offense EPA vs away defense EPA prevention (both higher==better)
    "home_off_vs_away_def": ("exp_off_epa_per_play", "exp_def_epa_prevention"),
    "away_off_vs_home_def": ("exp_off_epa_per_play", "exp_def_epa_prevention"),
}
CONTEXT_FEATURES = ["home_indicator", "is_neutral", "rest_days_diff",
                    "home_games_played", "away_games_played"]

TARGETS = ["home_win", "home_points", "away_points", "point_margin", "total_points"]


def _prior_rating(row: pd.Series, tier_prior: dict[str, float]) -> float:
    return tier_prior.get(row.get("tier", "group_of_five"), 0.0)


def build_feature_matrix(
    games: pd.DataFrame,
    teams: pd.DataFrame,
    settings: Settings | None = None,
    elo: EloModel | None = None,
) -> pd.DataFrame:
    """Return one row per game with feature columns + (nullable) targets."""

    settings = settings or get_settings()
    fcfg = settings.config["features"]
    min_games = int(fcfg["min_games_for_stats"])

    games = games.sort_values(["season", "week", "date"]).reset_index(drop=True)

    # 1) Elo pre-game ratings (leakage-safe by construction)
    if elo is None:
        elo = EloModel(settings=settings).fit(games, teams)
    elo_hist = elo.history[["game_id", "home_elo_pre", "away_elo_pre",
                            "elo_diff", "elo_home_win_prob"]]

    # 2) rolling team ratings
    log = team_game_log(games)
    rolled = rolling_team_features(log, settings)
    per_team = rolled[["game_id", "team", "games_played_before"] + RATING_COLS]

    home = per_team.add_prefix("h_").rename(
        columns={"h_game_id": "game_id", "h_team": "home_team"}
    )
    away = per_team.add_prefix("a_").rename(
        columns={"a_game_id": "game_id", "a_team": "away_team"}
    )

    df = games.merge(home, on=["game_id", "home_team"], how="left")
    df = df.merge(away, on=["game_id", "away_team"], how="left")
    df = df.merge(elo_hist, on="game_id", how="left")

    # 3) preseason-prior blend for thin-history teams.
    # The prior for a team's current season is its *previous* season's
    # final opponent-adjusted rating, regressed toward the mean.  Brand-new
    # / no-history teams fall back to a conference-tier constant.  The blend
    # weight moves from all-prior to all-in-season over `min_games` games,
    # exactly the "reduce preseason influence as games accumulate" rule.
    tier_map = dict(zip(teams["team"], teams.get("tier", pd.Series(index=teams.index))))
    tier_prior = {"power": 4.0, "independent": 1.0, "group_of_five": -7.0,
                  "unknown": 0.0}
    regress = 1.0 - float(fcfg.get("preseason_regression", 0.35))
    season_final = (
        rolled.sort_values("date")
        .groupby(["team", "season"]).tail(1)
        .set_index(["team", "season"])["adj_net_rating"]
        .to_dict()
    )

    def prior_for(team_series: pd.Series, season_series: pd.Series) -> pd.Series:
        out = []
        for tm, sn in zip(team_series, season_series):
            pv = season_final.get((tm, sn - 1))
            if pv is None or pd.isna(pv):
                out.append(tier_prior.get(tier_map.get(tm), 0.0))
            else:
                out.append(regress * float(pv))
        return pd.Series(out, index=team_series.index)

    for side in ("h", "a"):
        team_col = "home_team" if side == "h" else "away_team"
        gp = df[f"{side}_games_played_before"].fillna(0)
        w = (gp / max(min_games, 1)).clip(upper=1.0)  # 0 -> prior, 1 -> in-season
        prior = prior_for(df[team_col], df["season"])
        for col in RATING_COLS:
            margin_like = col in ("exp_net_margin", "ewm_net_margin", "adj_net_rating")
            filled = df[f"{side}_{col}"].fillna(prior if margin_like else 0.0)
            if margin_like:
                df[f"{side}_{col}"] = w * filled + (1 - w) * prior
            else:
                df[f"{side}_{col}"] = w * filled

    # 4) context features
    df["home_indicator"] = (~df["neutral_site"]).astype(int)
    df["is_neutral"] = df["neutral_site"].astype(int)
    df["home_games_played"] = df["h_games_played_before"].fillna(0)
    df["away_games_played"] = df["a_games_played_before"].fillna(0)
    rest = _rest_days(games)
    df = df.merge(rest, on="game_id", how="left")
    df["rest_days_diff"] = df["rest_days_diff"].fillna(0.0)

    # 5) diff + cross features
    df["elo_diff"] = df["elo_diff"].fillna(0.0)
    _apply_diff_features(df)

    # 6) targets (NaN for upcoming games)
    df["home_win"] = np.where(
        df["completed"] & df["home_points"].notna(),
        (df["home_points"] > df["away_points"]).astype("float"),
        np.nan,
    )
    df["point_margin"] = df["home_points"] - df["away_points"]
    df["total_points"] = df["home_points"] + df["away_points"]

    return df


def _apply_diff_features(df: pd.DataFrame) -> None:
    """Fill DIFF_FEATURES + cross terms from ``h_*`` / ``a_*`` columns."""

    for name, spec in DIFF_FEATURES.items():
        if spec is None:
            continue
        hcol, acol = spec
        df[name] = df[f"h_{hcol}"].fillna(0) - df[f"a_{acol}"].fillna(0)
    df["home_off_vs_away_def"] = (
        df["h_exp_off_epa_per_play"].fillna(0) - df["a_exp_def_epa_prevention"].fillna(0)
    )
    df["away_off_vs_home_def"] = (
        df["a_exp_off_epa_per_play"].fillna(0) - df["h_exp_def_epa_prevention"].fillna(0)
    )
    df["oline_matchup"] = df["line_yards_diff"]


def synthesize_matchup(
    home: str,
    away: str,
    latest_ratings: pd.DataFrame,
    elo,
    neutral: bool = False,
    adjustments: dict | None = None,
) -> pd.DataFrame:
    """Build a single feature row for a hypothetical matchup from each
    team's most recent rating snapshot.

    ``adjustments`` (all optional) let the user run "what-if" scenarios:
        home_rating_delta / away_rating_delta : points added to a team's
            Elo-scaled rating (injuries, personnel);
        home_off_delta / away_off_delta       : offensive-efficiency nudge;
        home_def_delta / away_def_delta       : defensive-efficiency nudge;
        turnover_margin_shift                 : points added to home
            turnover-margin feature;
        pace_shift                            : seconds/play adjustment.

    These are applied on top of the model's own ratings and are reported
    separately in the UI so model output and user assumptions never blur.
    """

    adj = adjustments or {}
    lr = latest_ratings.set_index("team")

    def snap(team: str) -> dict:
        if team in lr.index:
            return lr.loc[team].to_dict()
        return {}

    hs, as_ = snap(home), snap(away)
    row = {"game_id": "hypothetical", "home_team": home, "away_team": away,
           "neutral_site": neutral, "completed": False}

    for col in RATING_COLS:
        row[f"h_{col}"] = float(hs.get(col, 0.0) or 0.0)
        row[f"a_{col}"] = float(as_.get(col, 0.0) or 0.0)

    # apply scenario adjustments
    row["h_adj_net_rating"] += float(adj.get("home_rating_delta", 0.0))
    row["a_adj_net_rating"] += float(adj.get("away_rating_delta", 0.0))
    row["h_exp_net_margin"] += float(adj.get("home_rating_delta", 0.0))
    row["a_exp_net_margin"] += float(adj.get("away_rating_delta", 0.0))
    row["h_exp_off_epa_per_play"] += float(adj.get("home_off_delta", 0.0))
    row["a_exp_off_epa_per_play"] += float(adj.get("away_off_delta", 0.0))
    row["h_exp_def_epa_prevention"] += float(adj.get("home_def_delta", 0.0))
    row["a_exp_def_epa_prevention"] += float(adj.get("away_def_delta", 0.0))
    row["h_exp_turnover_margin"] += float(adj.get("turnover_margin_shift", 0.0))
    row["h_exp_pace"] += float(adj.get("pace_shift", 0.0))
    row["a_exp_pace"] += float(adj.get("pace_shift", 0.0))

    df = pd.DataFrame([row])
    df["home_indicator"] = 0 if neutral else 1
    df["is_neutral"] = 1 if neutral else 0
    df["home_games_played"] = float(hs.get("games_played_before", 6) or 6)
    df["away_games_played"] = float(as_.get("games_played_before", 6) or 6)
    df["rest_days_diff"] = 0.0

    elo_hfa = 0.0 if neutral else elo.hfa
    r_home = elo.rating(home) + float(adj.get("home_rating_delta", 0.0)) * 25.0
    r_away = elo.rating(away) + float(adj.get("away_rating_delta", 0.0)) * 25.0
    df["elo_diff"] = (r_home + elo_hfa) - r_away
    _apply_diff_features(df)
    for tgt in ("home_win", "point_margin", "total_points"):
        df[tgt] = np.nan
    return df


def feature_columns() -> list[str]:
    cols = [c for c in DIFF_FEATURES] + [
        "home_off_vs_away_def", "away_off_vs_home_def", "oline_matchup"
    ] + CONTEXT_FEATURES
    # de-dupe, keep order
    seen: set[str] = set()
    ordered = []
    for c in cols:
        if c not in seen:
            ordered.append(c)
            seen.add(c)
    return ordered


def _rest_days(games: pd.DataFrame) -> pd.DataFrame:
    """Days since each team's previous game; return home-minus-away diff."""

    long = []
    for side in ("home", "away"):
        long.append(pd.DataFrame({
            "game_id": games["game_id"],
            "team": games[f"{side}_team"],
            "date": games["date"],
            "side": side,
        }))
    lf = pd.concat(long).sort_values(["team", "date"])
    lf["prev"] = lf.groupby("team")["date"].shift(1)
    lf["rest"] = (lf["date"] - lf["prev"]).dt.days.fillna(7).clip(3, 21)
    wide = lf.pivot_table(index="game_id", columns="side", values="rest")
    wide["rest_days_diff"] = wide.get("home", 7) - wide.get("away", 7)
    return wide[["rest_days_diff"]].reset_index()
