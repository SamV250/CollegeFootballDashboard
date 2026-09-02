"""Model behaviour: Elo, calibration, score coherence, baseline comparison."""

from __future__ import annotations

import numpy as np
import pytest

from src.features.build import build_feature_matrix, feature_columns
from src.models.baselines import HomeWinRateBaseline, LogisticBaseline
from src.models.elo import EloModel
from src.models.evaluate import expected_calibration_error, win_metrics
from src.models.game_model import GamePredictor


@pytest.fixture(scope="module")
def trained(games, teams, settings):
    elo = EloModel(settings=settings).fit(games, teams)
    feat = build_feature_matrix(games, teams, settings, elo=elo)
    labeled = feat[feat["home_win"].notna()]
    train = labeled[labeled["season"] <= 2025]
    test = labeled[labeled["season"] == 2026]
    predictor = GamePredictor(features=feature_columns(), settings=settings)
    predictor.fit(train)
    return {"elo": elo, "predictor": predictor, "train": train, "test": test,
            "feat": feat}


def test_elo_probabilities_are_valid(trained):
    hist = trained["elo"].history
    assert hist["elo_home_win_prob"].between(0, 1).all()


def test_win_probabilities_in_unit_interval(trained):
    p = trained["predictor"].win_probability(trained["test"])
    assert np.all((p > 0) & (p < 1))


def test_scores_are_coherent(trained):
    pred = trained["predictor"].predict(trained["test"])
    # home + away == total ; home - away == margin
    assert np.allclose(pred["pred_home_points"] + pred["pred_away_points"],
                       pred["pred_total"], atol=1e-6)
    assert np.allclose(pred["pred_home_points"] - pred["pred_away_points"],
                       pred["pred_margin"], atol=1e-6)
    assert (pred[["pred_home_points", "pred_away_points"]] >= 0).all().all()


def test_calibration_improves_or_matches_raw(trained):
    test = trained["test"]
    y = test["home_win"].astype(int).values
    raw = trained["predictor"]._raw_prob(test)
    cal = trained["predictor"].win_probability(test)
    ece_raw = expected_calibration_error(y, raw)
    ece_cal = expected_calibration_error(y, cal)
    # calibration should not make things materially worse
    assert ece_cal <= ece_raw + 0.03


def test_primary_model_beats_naive_home_rate(trained):
    test = trained["test"]
    y = test["home_win"].astype(int).values
    gbm = win_metrics(y, trained["predictor"].win_probability(test))
    hwr = HomeWinRateBaseline().fit(trained["train"]["home_win"])
    base = win_metrics(y, hwr.predict_proba(len(test)))
    assert gbm["log_loss"] < base["log_loss"]


def test_logistic_baseline_runs_and_is_reasonable(trained):
    logit = LogisticBaseline().fit(trained["train"], trained["train"]["home_win"])
    p = logit.predict_proba(trained["test"])
    assert np.all((p > 0) & (p < 1))
    coefs = logit.coefficients()
    # Elo gap should carry a positive coefficient (favours home)
    assert coefs.set_index("feature").loc["elo_diff", "coefficient"] > 0


def test_predictions_are_deterministic(trained):
    a = trained["predictor"].predict(trained["test"])["home_win_prob"].values
    b = trained["predictor"].predict(trained["test"])["home_win_prob"].values
    assert np.array_equal(a, b)
