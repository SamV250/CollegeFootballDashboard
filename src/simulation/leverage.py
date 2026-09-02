"""Playoff leverage: which upcoming results move the field the most.

For a shortlist of high-relevance upcoming games we run two conditional
simulations -- one forcing a home win, one forcing an away win -- and
measure how far every team's playoff probability moves between the two
worlds.  A game's leverage is the total absolute swing (summed over
teams); we also report the single most-affected team.

Runs on a shortlist with a reduced iteration count and is cached in the
dashboard artifact.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.simulation.season import SeasonSimulator

log = logging.getLogger(__name__)


def _playoff_vector(sim: SeasonSimulator, n_iter: int, seed: int) -> np.ndarray:
    sim.settings.config["simulation"]["random_seed"] = seed
    res = sim.run(n_iterations=n_iter)
    vec = np.zeros(sim.n)
    for t, p in zip(res.team_probabilities["team"], res.team_probabilities["p_playoff"]):
        vec[sim.tidx[t]] = p
    return vec


def compute_leverage(
    sim: SeasonSimulator,
    shortlist_size: int = 16,
    n_iter: int = 4000,
) -> pd.DataFrame:
    """Return a leverage table for the highest-impact upcoming games."""

    rem = sim.remaining.copy()
    cols = ["game_id", "matchup", "week", "leverage",
            "most_affected_team", "swing_for_team"]
    if rem.empty:
        return pd.DataFrame(columns=cols)

    strength = {t: float(sim.elo.rating(t)) for t in sim.T}
    rem = rem.assign(relevance=[
        (1 - abs(p - 0.5) * 2) * (strength.get(h, 1500) + strength.get(a, 1500))
        for h, a, p in rem[["home_team", "away_team", "home_win_prob"]]
        .itertuples(index=False)
    ])
    shortlist = rem.sort_values("relevance", ascending=False).head(shortlist_size)
    base_p = dict(zip(sim.remaining["game_id"], sim.remaining["home_win_prob"]))
    saved_seed = sim.settings.config["simulation"]["random_seed"]

    rows = []
    for r in shortlist.itertuples():
        gid = r.game_id
        m = sim.remaining["game_id"] == gid
        sim.remaining.loc[m, "home_win_prob"] = 0.999
        sim._static_records()
        pa = _playoff_vector(sim, n_iter, 101)

        sim.remaining.loc[m, "home_win_prob"] = 0.001
        sim._static_records()
        pb = _playoff_vector(sim, n_iter, 202)

        sim.remaining.loc[m, "home_win_prob"] = base_p[gid]
        sim._static_records()

        swings = np.abs(pa - pb)
        top = int(np.argmax(swings))
        rows.append({
            "game_id": gid,
            "matchup": (f"{r.away_team} vs {r.home_team}" if r.neutral_site
                        else f"{r.away_team} at {r.home_team}"),
            "week": int(r.week),
            "leverage": round(float(swings.sum()), 3),
            "most_affected_team": sim.T[top],
            "swing_for_team": round(float(swings[top]), 3),
        })

    sim.settings.config["simulation"]["random_seed"] = saved_seed
    return pd.DataFrame(rows).sort_values("leverage", ascending=False).reset_index(drop=True)
