"""Preseason priors for new / low-history teams must be reasonable."""

from __future__ import annotations

import numpy as np

from src.features.build import build_feature_matrix, feature_columns
from src.models.elo import EloModel


def test_new_team_gets_reasonable_elo_prior(settings):
    elo = EloModel(settings=settings)
    r = elo._get("Some Brand New FBS Team", {"Some Brand New FBS Team"})
    assert 1200 <= r <= 1500
    # a non-FBS opponent is lower still
    r_fcs = elo._get("Tiny FCS School", set())
    assert r_fcs < r


def test_low_history_team_features_are_finite_and_bounded(games, teams, settings):
    """A team appearing for the first time in 2026 (no prior season) must
    still get finite, sane feature values via the tier prior."""

    # Force a team to have essentially no history: keep only its first
    # 2026 game and drop all its earlier games.
    victim = "UConn"
    trimmed = games[~(
        ((games["home_team"] == victim) | (games["away_team"] == victim))
        & (games["season"] < 2026)
    )].copy()

    feat = build_feature_matrix(trimmed, teams, settings)
    fcols = feature_columns()
    rows = feat[(feat["home_team"] == victim) | (feat["away_team"] == victim)]
    vals = rows[fcols].to_numpy()
    assert np.isfinite(vals).all()
    assert np.abs(vals).max() < 500  # nothing exploded


def test_prior_influence_decreases_with_games_played(games, teams, settings):
    """Feature blend weight -> in-season data as games accumulate."""

    feat = build_feature_matrix(games, teams, settings)
    cur = feat[feat["season"] == 2026].copy()
    early = cur[cur["home_games_played"] <= 1]
    late = cur[cur["home_games_played"] >= 5]
    # variance of the adjusted-rating diff should grow once real games
    # replace the shrunk-to-prior values
    if len(early) > 10 and len(late) > 10:
        assert late["adj_rating_diff"].std() >= early["adj_rating_diff"].std()
