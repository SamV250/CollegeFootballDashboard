"""Explanations must be well-formed and in football language."""

from __future__ import annotations

import pytest

from src.explainability.narrative import (
    confidence_label,
    describe_contribution,
    magnitude_label,
)
from src.explainability.shap_explain import explain_game
from src.features.build import build_feature_matrix, feature_columns
from src.models.elo import EloModel
from src.models.game_model import GamePredictor


def test_magnitude_and_confidence_buckets():
    assert magnitude_label(0.9) == "major advantage"
    assert magnitude_label(0.4) == "moderate advantage"
    assert magnitude_label(0.15) == "slight advantage"
    assert magnitude_label(0.01) == "essentially even"
    assert confidence_label(0.95) == "High"
    assert confidence_label(0.52) == "Toss-up"


def test_describe_contribution_returns_football_sentence():
    d = describe_contribution("off_epa_diff", 0.4, 1.0, "Georgia", "Tennessee")
    assert d["favors"] == "home"
    assert "Georgia" in d["sentence"]
    assert d["magnitude"] in {"major advantage", "moderate advantage",
                              "slight advantage", "essentially even"}
    assert d["tooltip"]


@pytest.fixture(scope="module")
def predictor(games, teams, settings):
    elo = EloModel(settings=settings).fit(games, teams)
    feat = build_feature_matrix(games, teams, settings, elo=elo)
    train = feat[feat["home_win"].notna() & (feat["season"] <= 2025)]
    p = GamePredictor(features=feature_columns(), settings=settings)
    p.fit(train)
    return p, feat


def test_explain_game_structure(predictor):
    p, feat = predictor
    upcoming = feat[~feat["completed"]]
    row = upcoming.iloc[[0]]
    pred = p.predict(row).iloc[0]
    exp = explain_game(p, row, pred)
    assert 0 < exp["home_win_prob"] < 1
    assert exp["favorite"] in (exp["home_team"], exp["away_team"])
    assert len(exp["favorite_factors"]) <= 5
    assert len(exp["underdog_factors"]) <= 3
    assert isinstance(exp["summary"], str) and len(exp["summary"]) > 40
    lo, hi = exp["plausible_margin_range"]
    assert lo < hi
    # no raw shap numbers leak into the prose
    assert "shap" not in exp["summary"].lower()
