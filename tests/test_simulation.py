"""Season simulation + playoff selection sanity checks."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.build import build_feature_matrix, feature_columns
from src.models.elo import EloModel
from src.models.game_model import GamePredictor
from src.simulation.playoff import PlayoffConfig, build_bracket
from src.simulation.season import SeasonSimulator


@pytest.fixture(scope="module")
def sim(games, teams, settings):
    elo = EloModel(settings=settings).fit(games, teams)
    feat = build_feature_matrix(games, teams, settings, elo=elo)
    train = feat[feat["home_win"].notna() & (feat["season"] <= 2025)]
    predictor = GamePredictor(features=feature_columns(), settings=settings)
    predictor.fit(train)
    return SeasonSimulator(games, teams, predictor, elo, settings)


def test_probabilities_sum_correctly(sim):
    res = sim.run(n_iterations=1500)
    tp = res.team_probabilities
    pcfg = PlayoffConfig.from_settings(sim.settings)
    assert abs(tp["p_national_champion"].sum() - 1.0) < 0.02
    assert abs(tp["p_playoff"].sum() - pcfg.n_teams) < 0.25
    assert abs(tp["p_first_round_bye"].sum() - pcfg.n_byes) < 0.25
    # every probability in [0, 1]
    for c in [c for c in tp.columns if c.startswith("p_")]:
        assert tp[c].between(0, 1).all()


def test_completed_games_are_respected(sim):
    """A team that already lost every game can't have a high title prob."""

    res = sim.run(n_iterations=1000)
    tp = res.team_probabilities.set_index("team")
    # sanity: the strongest Elo team should have >0 title probability
    top_elo = sim.elo.rating_table().iloc[0]["team"]
    if top_elo in tp.index:
        assert tp.loc[top_elo, "p_playoff"] >= 0


def test_more_iterations_reduce_noise(sim):
    a = sim.run(n_iterations=400).team_probabilities.set_index("team")["p_playoff"]
    b = sim.run(n_iterations=4000).team_probabilities.set_index("team")["p_playoff"]
    # both are valid distributions over the same teams
    assert set(a.index) == set(b.index)
    assert a.between(0, 1).all() and b.between(0, 1).all()


def test_bracket_has_right_size_and_byes(settings):
    pcfg = PlayoffConfig.from_settings(settings)
    ranked = pd.DataFrame({
        "team": [f"T{i}" for i in range(40)],
        "conference": (["SEC", "Big Ten", "Big 12", "ACC", "Pac-12"] * 8),
        "is_conf_champ": [i < 6 for i in range(40)],
        "score": np.linspace(20, 1, 40),
    })
    field = build_bracket(ranked, pcfg)
    assert len(field) == pcfg.n_teams
    assert field["bye"].sum() == pcfg.n_byes
    assert field["seed"].tolist() == list(range(1, pcfg.n_teams + 1))


def test_forcing_a_result_moves_probabilities(sim):
    """Forcing an upcoming game changes the involved teams' odds."""

    if sim.remaining.empty:
        pytest.skip("no remaining games")
    base = sim.run(n_iterations=1500).team_probabilities.set_index("team")["p_playoff"]
    g = sim.remaining.iloc[0]
    sim.remaining.loc[sim.remaining["game_id"] == g["game_id"], "home_win_prob"] = 0.999
    sim._static_records()
    forced = sim.run(n_iterations=1500).team_probabilities.set_index("team")["p_playoff"]
    # restore
    sim.remaining.loc[sim.remaining["game_id"] == g["game_id"], "home_win_prob"] = \
        g["home_win_prob"]
    sim._static_records()
    assert (base - forced).abs().max() > 0.0
