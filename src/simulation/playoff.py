"""Transparent, configurable playoff selection + bracket model.

The real selection committee cannot be reproduced exactly, so this module
implements an **explicit, tunable proxy** (weights live in
``config/playoff.yaml``).  Every probability derived from it is labelled
an ESTIMATE in the UI.

``selection_score`` combines: winning percentage, opponent-adjusted
efficiency, strength of schedule, quality wins, bad losses, a
conference-championship bonus, conference strength, a head-to-head nudge
and a "committee prior" (preseason + model blend).  ``build_bracket``
turns a ranking into a seeded 12-team field with byes, honouring the
"N highest-ranked conference champions are guaranteed a bid" rule.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import Settings, get_settings


@dataclass
class PlayoffConfig:
    n_teams: int
    n_byes: int
    guaranteed_conf_champs: int
    autobid_conferences: list[str]
    weights: dict[str, float]
    quality_win_cutoff: int
    bad_loss_cutoff: int

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> "PlayoffConfig":
        s = settings or get_settings()
        p = s.playoff
        fmt, sel = p["format"], p["selection_score"]
        return cls(
            n_teams=int(fmt["n_teams"]),
            n_byes=int(fmt["n_first_round_byes"]),
            guaranteed_conf_champs=int(fmt["highest_ranked_conf_champs"]),
            autobid_conferences=list(sel["autobid_conferences"]),
            weights=dict(sel["weights"]),
            quality_win_cutoff=int(sel["quality_win_rank_cutoff"]),
            bad_loss_cutoff=int(sel["bad_loss_rank_cutoff"]),
        )


def selection_score(
    table: pd.DataFrame, cfg: PlayoffConfig
) -> pd.Series:
    """Compute the committee-style score for every team.

    ``table`` columns required:
        win_pct, adj_efficiency_z, sos_z, quality_wins, bad_losses,
        is_conf_champ, conf_strength_z, committee_prior_z, h2h_bonus
    """

    w = cfg.weights
    score = (
        w["win_pct"] * table["win_pct"]
        + w["adjusted_efficiency"] * table["adj_efficiency_z"]
        + w["strength_of_schedule"] * table["sos_z"]
        + w["quality_wins"] * table["quality_wins"]
        + w["bad_losses"] * table["bad_losses"]
        + w["conference_champion"] * table["is_conf_champ"].astype(float)
        + w["conference_strength"] * table["conf_strength_z"]
        + w["committee_prior"] * table["committee_prior_z"]
        + w["head_to_head"] * table.get("h2h_bonus", 0.0)
    )
    return score


def build_bracket(
    ranked: pd.DataFrame, cfg: PlayoffConfig
) -> pd.DataFrame:
    """Given teams sorted best-first with columns ['team','conference',
    'is_conf_champ','score'], return the selected + seeded field.

    Rules implemented:
    * The ``guaranteed_conf_champs`` highest-ranked conference champions
      are guaranteed a spot even if ranked outside the at-large cut.
    * Remaining spots go to the next-highest-ranked teams at large.
    * Seeds 1..n_byes are byes.  (Committee re-seeding of champions into
      the top seeds is configurable via ``byes_go_to_conf_champs`` but we
      keep it simple: seed strictly by score among the selected field.)
    """

    ranked = ranked.sort_values("score", ascending=False).reset_index(drop=True)
    champs = ranked[ranked["is_conf_champ"]].head(cfg.guaranteed_conf_champs)
    selected = list(champs["team"])
    for team in ranked["team"]:
        if len(selected) >= cfg.n_teams:
            break
        if team not in selected:
            selected.append(team)

    field = ranked[ranked["team"].isin(selected)].copy()
    field = field.sort_values("score", ascending=False).reset_index(drop=True)
    field["seed"] = np.arange(1, len(field) + 1)
    field["bye"] = field["seed"] <= cfg.n_byes
    return field


def simulate_bracket(
    field: pd.DataFrame,
    neutral_win_prob,
    rng: np.random.Generator,
) -> dict[str, list[str]]:
    """Simulate a 12-team bracket (4 byes). Returns round reach lists.

    ``neutral_win_prob(a, b)`` -> P(a beats b) on a neutral field.
    """

    seeds = {int(r.seed): r.team for r in field.itertuples()}
    n = len(field)
    reached = {"quarterfinal": [], "semifinal": [], "final": [], "champion": []}

    if n < 8:  # degenerate; just rank
        order = list(field["team"])
        reached["champion"] = order[:1]
        return reached

    # First round: 5v12, 6v11, 7v10, 8v9  (seeds 1-4 bye)
    fr_pairs = [(5, 12), (6, 11), (7, 10), (8, 9)]
    fr_winners = {}
    for hi, lo in fr_pairs:
        if lo not in seeds:
            fr_winners[hi] = seeds.get(hi)
            continue
        a, b = seeds[hi], seeds[lo]
        fr_winners[hi] = a if rng.random() < neutral_win_prob(a, b) else b

    # Quarterfinals: 1 v W(8/9), 2 v W(7/10), 3 v W(6/11), 4 v W(5/12)
    qf_map = [(1, 8), (2, 7), (3, 6), (4, 5)]
    qf_winners = []
    for top, fr_slot in qf_map:
        a = seeds[top]
        b = fr_winners[fr_slot]
        reached["quarterfinal"].extend([a, b])
        qf_winners.append(a if rng.random() < neutral_win_prob(a, b) else b)

    # Semifinals: QF0 vs QF3, QF1 vs QF2
    sf_pairs = [(qf_winners[0], qf_winners[3]), (qf_winners[1], qf_winners[2])]
    finalists = []
    for a, b in sf_pairs:
        reached["semifinal"].extend([a, b])
        finalists.append(a if rng.random() < neutral_win_prob(a, b) else b)

    reached["final"] = finalists
    a, b = finalists
    reached["champion"] = [a if rng.random() < neutral_win_prob(a, b) else b]
    return reached
