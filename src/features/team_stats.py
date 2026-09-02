"""Leakage-safe rolling team statistics.

The single most important rule in this file: **every value attached to a
game is computed only from games that kicked off strictly earlier.**  We
enforce that by sorting each team's games by date and calling
``.shift(1)`` on every rolling aggregate, so a game never contributes to
its own features.

Two views of form are produced for each metric:

* ``exp_*``  -- season-to-date expanding mean (stable, season-long form)
* ``ewm_*``  -- exponentially weighted mean (recent form, half-life from
  ``config.features.recent_form_halflife``)

A single-pass opponent adjustment (``adj_net_rating``) folds in the
quality of opponents already played.  It is intentionally simple and
transparent rather than a full simultaneous solve.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import Settings, get_settings

# metric name -> True if a higher raw value is better for the team that
# "owns" the metric.  Defensive metrics are stored from the defense's
# perspective (points/EPA allowed) so higher == worse; we flip them into
# "prevention" features (higher == better) in :func:`team_game_log`.
OWN_METRICS_HIGHER_BETTER = {
    "off_epa_per_play": True,
    "success_rate": True,
    "explosiveness": True,
    "pass_epa_per_play": True,
    "rush_epa_per_play": True,
    "sack_rate": True,
    "finish_pts_per_opp": True,
    "line_yards": True,
    "st_epa": True,
}
OWN_METRICS_LOWER_BETTER = {  # defense allowed -- flipped to prevention
    "def_epa_per_play": "def_epa_prevention",
    "def_success_rate": "def_success_prevention",
    "def_explosiveness": "def_explosiveness_prevention",
    "def_havoc": "def_havoc_prevention",
    "sack_rate_allowed": "pass_pro_prevention",
    "def_finish_pts_per_opp": "def_finish_prevention",
    "def_line_yards": "def_line_prevention",
}
NEUTRAL_METRICS = {"pace_sec_per_play": "pace", "havoc": "havoc"}


def team_game_log(games: pd.DataFrame) -> pd.DataFrame:
    """Explode one game row into two team-perspective rows.

    Only *completed* games are returned -- upcoming games carry no stats.
    """

    done = games[games["completed"] & games["home_points"].notna()].copy()
    frames = []
    for side, opp in (("home", "away"), ("away", "home")):
        f = pd.DataFrame({
            "game_id": done["game_id"].values,
            "season": done["season"].values,
            "week": done["week"].values,
            "date": done["date"].values,
            "team": done[f"{side}_team"].values,
            "opponent": done[f"{opp}_team"].values,
            "is_home": (side == "home") & (~done["neutral_site"].values),
            "neutral": done["neutral_site"].values,
            "points_for": done[f"{side}_points"].values.astype(float),
            "points_against": done[f"{opp}_points"].values.astype(float),
        })
        f["won"] = (f["points_for"] > f["points_against"]).astype(int)
        f["net_margin"] = f["points_for"] - f["points_against"]

        for metric in OWN_METRICS_HIGHER_BETTER:
            col = f"{side}_{metric}"
            f[metric] = done[col].values if col in done else np.nan
        for metric, pretty in OWN_METRICS_LOWER_BETTER.items():
            col = f"{side}_{metric}"
            f[pretty] = -done[col].values if col in done else np.nan
        for metric, pretty in NEUTRAL_METRICS.items():
            col = f"{side}_{metric}"
            f[pretty] = done[col].values if col in done else np.nan

        to_col, ta_col = f"{side}_turnovers", f"{side}_takeaways"
        if to_col in done:
            f["turnover_margin"] = (done[ta_col].values - done[to_col].values)
        else:
            f["turnover_margin"] = np.nan
        frames.append(f)

    log = pd.concat(frames, ignore_index=True)
    return log.sort_values(["team", "date"]).reset_index(drop=True)


ROLLED_METRICS = (
    ["net_margin", "points_for", "points_against", "won", "turnover_margin"]
    + list(OWN_METRICS_HIGHER_BETTER)
    + list(OWN_METRICS_LOWER_BETTER.values())
    + list(NEUTRAL_METRICS.values())
)


def rolling_team_features(
    log: pd.DataFrame, settings: Settings | None = None
) -> pd.DataFrame:
    """Attach ``exp_*`` and ``ewm_*`` pre-game aggregates to each row."""

    settings = settings or get_settings()
    hl = float(settings.config["features"]["recent_form_halflife"])
    out = log.copy()

    grp = out.groupby(["team", "season"], group_keys=False)
    out["games_played_before"] = grp.cumcount()

    for metric in ROLLED_METRICS:
        if metric not in out:
            continue
        g = out.groupby(["team", "season"])[metric]
        # expanding: season-to-date, current game excluded
        out[f"exp_{metric}"] = g.transform(
            lambda s: s.expanding().mean().shift(1)
        )
        # exponentially weighted: recent form, current game excluded
        out[f"ewm_{metric}"] = g.transform(
            lambda s: s.ewm(halflife=hl).mean().shift(1)
        )

    # -- single-pass opponent adjustment --------------------------------
    # opponent's pre-game season net-margin, aligned to this game
    pre = out[["game_id", "team", "exp_net_margin"]].rename(
        columns={"team": "opponent", "exp_net_margin": "opp_exp_net_margin"}
    )
    out = out.merge(pre, on=["game_id", "opponent"], how="left")
    og = out.groupby(["team", "season"])["opp_exp_net_margin"]
    out["sos_to_date"] = og.transform(lambda s: s.expanding().mean().shift(1))
    # adjusted rating = own scoring margin + strength of opponents faced
    out["adj_net_rating"] = out["exp_net_margin"].fillna(0) + out["sos_to_date"].fillna(0)
    return out


def latest_team_ratings(
    rolled: pd.DataFrame, season: int
) -> pd.DataFrame:
    """One row per team: their most recent pre-game rating snapshot in
    ``season`` (used for team profiles and hypothetical matchups)."""

    cur = rolled[rolled["season"] == season]
    if cur.empty:
        cur = rolled
    latest = cur.sort_values("date").groupby("team").tail(1)
    keep = [c for c in latest.columns if c.startswith(("exp_", "ewm_"))]
    keep += ["team", "games_played_before", "adj_net_rating", "sos_to_date"]
    return latest[keep].reset_index(drop=True)
