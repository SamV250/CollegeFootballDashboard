"""Data-layer integrity: schema, name mappings, neutral sites, idempotency,
missing-data resilience."""

from __future__ import annotations

import numpy as np

from src.data.sources import CANONICAL_GAME_COLUMNS
from src.data.store import DataStore
from src.features.build import build_feature_matrix, feature_columns


def test_demo_source_schema(demo_data):
    games = demo_data["games"]
    for col in CANONICAL_GAME_COLUMNS:
        assert col in games.columns, f"missing {col}"
    assert games["game_id"].is_unique
    assert games["date"].dt.tz is not None  # tz-aware UTC


def test_team_name_mappings_consistent(games, teams, settings):
    """Every team in games has exactly one conference, matching config."""

    team_conf = settings.team_conference_map()
    appearing = set(games["home_team"]) | set(games["away_team"])
    for t in appearing:
        assert t in team_conf, f"{t} not in conference config"
    # home_conference column agrees with the map
    bad = games[games["home_team"].map(team_conf) != games["home_conference"]]
    assert bad.empty, f"conference mismatch for {bad['home_team'].unique()[:5]}"


def test_neutral_site_games_have_no_home_indicator(games, teams, settings):
    feat = build_feature_matrix(games, teams, settings)
    neutral = feat[feat["neutral_site"]]
    assert (neutral := neutral)["home_indicator"].eq(0).all()
    assert neutral["is_neutral"].eq(1).all()
    non_neutral = feat[~feat["neutral_site"]]
    assert non_neutral["home_indicator"].eq(1).all()


def test_neutral_site_elo_has_no_home_bump(games, teams, settings):
    from src.models.elo import EloModel

    elo = EloModel(settings=settings)
    p_home = elo.win_probability("Georgia", "Georgia", neutral=True)
    assert abs(p_home - 0.5) < 1e-9  # identical teams, neutral -> 50/50
    p_home_field = elo.win_probability("Georgia", "Georgia", neutral=False)
    assert p_home_field > 0.5  # home bump applies


def test_missing_advanced_metrics_do_not_crash(games, teams, settings):
    """Drop every advanced-stat column; pipeline must still produce features."""

    stripped = games[CANONICAL_GAME_COLUMNS].copy()
    feat = build_feature_matrix(stripped, teams, settings)
    fcols = feature_columns()
    assert not feat[fcols].isna().any().any()
    assert len(feat) == len(stripped)


def test_missing_scores_do_not_crash(games, teams, settings):
    holed = games.copy()
    # randomly null 30 completed scores
    done_idx = holed[holed["completed"]].sample(30, random_state=0).index
    holed.loc[done_idx, ["home_points", "away_points"]] = np.nan
    feat = build_feature_matrix(holed, teams, settings)
    assert len(feat) == len(holed)


def test_store_upsert_is_idempotent(tmp_path, settings, demo_data, monkeypatch):
    monkeypatch.setattr(type(settings), "processed_dir",
                        property(lambda self: tmp_path))
    store = DataStore(settings)
    s1 = store.upsert(demo_data, source="demo")
    s2 = store.upsert(demo_data, source="demo")
    assert s2["rows_added"] == 0
    assert s1["rows_total"] == s2["rows_total"]
    assert store.load_games()["game_id"].is_unique


def test_store_upsert_adds_new_results_without_dupes(tmp_path, settings, demo_data,
                                                     monkeypatch):
    monkeypatch.setattr(type(settings), "processed_dir",
                        property(lambda self: tmp_path))
    store = DataStore(settings)
    early = {"teams": demo_data["teams"],
             "games": demo_data["games"].iloc[:400].copy()}
    store.upsert(early, source="demo")
    full = store.upsert(demo_data, source="demo")
    assert full["rows_total"] == len(demo_data["games"])
    assert store.load_games()["game_id"].is_unique
