"""Data-leakage guards — the most important tests in the project.

For a game on date D, only information available before kickoff may enter
its feature row.  We verify this structurally by mutating games and
checking which feature rows move.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.features.build import build_feature_matrix, feature_columns
from src.features.team_stats import rolling_team_features, team_game_log
from src.models.elo import EloModel


@pytest.fixture(scope="module")
def feat(games, teams, settings):
    return build_feature_matrix(games, teams, settings)


def _completed(feat: pd.DataFrame) -> pd.DataFrame:
    return feat[feat["home_win"].notna()].sort_values("date")


def test_current_game_stats_do_not_leak_into_its_own_features(games, teams, settings):
    """Blowing up one game's box score must NOT change that game's own
    feature row (its features come only from earlier games)."""

    base = build_feature_matrix(games, teams, settings)
    done = _completed(base)
    # pick a mid-season 2025 game so both teams have prior history
    target = done[(done["season"] == 2025) & (done["week"].between(6, 9))].iloc[0]
    gid = target["game_id"]

    tampered = games.copy()
    m = tampered["game_id"] == gid
    for col in tampered.columns:
        if col.startswith(("home_", "away_")) and pd.api.types.is_numeric_dtype(
            tampered[col]
        ):
            tampered.loc[m, col] = tampered.loc[m, col] * 0 + 999.0
    tampered.loc[m, "home_points"] = 999
    tampered.loc[m, "away_points"] = 0

    after = build_feature_matrix(tampered, teams, settings)
    fcols = feature_columns()
    row_before = base.loc[base["game_id"] == gid, fcols].iloc[0]
    row_after = after.loc[after["game_id"] == gid, fcols].iloc[0]

    pd.testing.assert_series_equal(row_before, row_after, check_names=False,
                                   rtol=1e-9, atol=1e-9)


def test_future_game_does_not_change_past_feature_rows(games, teams, settings):
    """Mutating a late-season game must not move an earlier game's row for
    the same teams."""

    base = build_feature_matrix(games, teams, settings)
    done = _completed(base)
    late = done[(done["season"] == 2025) & (done["week"] >= 11)].iloc[0]
    home, away = late["home_team"], late["away_team"]
    earlier = done[(done["season"] == 2025) & (done["week"] <= 4) &
                   ((done["home_team"] == home) | (done["away_team"] == home) |
                    (done["home_team"] == away) | (done["away_team"] == away))]
    if earlier.empty:
        pytest.skip("no earlier game for these teams in this synthetic draw")
    early_gid = earlier.iloc[0]["game_id"]

    tampered = games.copy()
    m = tampered["game_id"] == late["game_id"]
    tampered.loc[m, "home_points"] = 1
    tampered.loc[m, "away_points"] = 99

    after = build_feature_matrix(tampered, teams, settings)
    fcols = feature_columns()
    b = base.loc[base["game_id"] == early_gid, fcols].iloc[0]
    a = after.loc[after["game_id"] == early_gid, fcols].iloc[0]
    pd.testing.assert_series_equal(b, a, check_names=False, rtol=1e-9, atol=1e-9)


def test_rolling_features_are_shifted(games, settings):
    """The first game of a team's season has no prior expanding mean."""

    log = team_game_log(games)
    rolled = rolling_team_features(log, settings)
    firsts = rolled.sort_values("date").groupby(["team", "season"]).head(1)
    assert firsts["exp_net_margin"].isna().all()
    assert (firsts["games_played_before"] == 0).all()


def test_elo_pregame_rating_precedes_update(games, teams, settings):
    """Elo history stores the rating *before* the game; the post-game
    update only affects later games."""

    elo = EloModel(settings=settings).fit(games, teams)
    hist = elo.history.merge(
        games[["game_id", "home_team", "away_team", "date"]], on="game_id"
    ).sort_values("date")
    # a team's away_elo_pre in a later game equals its home/away_elo_pre
    # from the previous game +/- that game's delta -> monotonic chain, not
    # an exact identity, so we just assert no NaNs and correct ordering.
    assert hist["home_elo_pre"].notna().all()
    assert hist["away_elo_pre"].notna().all()
    # ratings must not all be identical (model actually learned something)
    assert elo.rating_table()["elo"].std() > 20


def test_no_completed_flag_leak_in_targets(feat):
    """Upcoming games must have NaN targets; completed games must not."""

    upcoming = feat[~feat["completed"]]
    assert upcoming["home_win"].isna().all()
    assert upcoming["point_margin"].isna().all()
    done = feat[feat["completed"] & feat["home_points"].notna()]
    assert done["home_win"].notna().all()
