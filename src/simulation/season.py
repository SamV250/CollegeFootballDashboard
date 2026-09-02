"""Monte Carlo simulation of the rest of the season (vectorised).

For each of ``config.simulation.n_iterations`` iterations we:

1. keep every completed game as-is;
2. simulate every remaining regular-season game from the primary model's
   calibrated home win probability;
3. compute conference standings and simulate each conference
   championship game on a neutral field;
4. score every team with the transparent, configurable selection model
   (:mod:`src.simulation.playoff`);
5. select + seed a 12-team field (5 highest-ranked champions guaranteed,
   top 4 seeds get byes);
6. simulate every playoff round on neutral fields;
7. record conference-title / playoff / bye / semifinal / title-game /
   national-championship outcomes.

Neutral-field probabilities for hypothetical matchups (title games and
the bracket) come from a pre-computed Elo win-probability matrix, so
10k iterations run in a couple of seconds.  Regular-season game
probabilities are the full calibrated model.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.config import Settings, get_settings, utc_now
from src.features.build import build_feature_matrix
from src.models.elo import EloModel
from src.models.game_model import GamePredictor
from src.simulation.playoff import PlayoffConfig

log = logging.getLogger(__name__)


@dataclass
class SeasonSimResult:
    team_probabilities: pd.DataFrame
    game_predictions: pd.DataFrame
    generated_at_utc: str
    n_iterations: int
    season: int
    meta: dict = field(default_factory=dict)


def _z(a: np.ndarray) -> np.ndarray:
    sd = a.std()
    return (a - a.mean()) / sd if sd > 1e-9 else np.zeros_like(a)


class SeasonSimulator:
    def __init__(
        self,
        games: pd.DataFrame,
        teams: pd.DataFrame,
        predictor: GamePredictor,
        elo: EloModel,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.teams = teams
        self.predictor = predictor
        self.elo = elo
        self.season = self.settings.active_season()
        self.games = games[games["season"] == self.season].copy()
        self.pcfg = PlayoffConfig.from_settings(self.settings)
        self.conf_map = dict(zip(teams["team"], teams["conference"]))
        self._prepare()

    # ------------------------------------------------------------------ setup
    def _prepare(self) -> None:
        feat = build_feature_matrix(self.games, self.teams, self.settings, self.elo)
        feat = feat[feat["season"] == self.season]
        self.completed = feat[feat["completed"] & feat["home_win"].notna()].copy()
        self.remaining = feat[~feat["completed"]].copy()

        if len(self.remaining):
            preds = self.predictor.predict(self.remaining)
            self.remaining["home_win_prob"] = preds["home_win_prob"].values
            self.remaining["pred_home_points"] = preds["pred_home_points"].values
            self.remaining["pred_away_points"] = preds["pred_away_points"].values

        self.T = sorted(set(self.games["home_team"]) | set(self.games["away_team"]))
        self.n = len(self.T)
        self.tidx = {t: i for i, t in enumerate(self.T)}

        # conferences
        self.conf_of_team = [self.conf_map.get(t, "FBS Independents") for t in self.T]
        self.conf_names = sorted(set(self.conf_of_team))
        self.cidx = {c: i for i, c in enumerate(self.conf_names)}
        self.conf_code = np.array([self.cidx[c] for c in self.conf_of_team])
        conf_meta = self.settings.conferences["conferences"]
        self.conf_has_champ = np.array([
            bool(conf_meta.get(c, {}).get("champ_game")) and
            (self.conf_code == self.cidx[c]).sum() >= 2
            for c in self.conf_names
        ])

        # Elo neutral win-prob matrix  P[i, j] = P(i beats j)
        r = np.array([self.elo.rating(t) for t in self.T])
        self.elo_rating = r
        diff = r[:, None] - r[None, :]
        self.elo_P = 1.0 / (1.0 + 10.0 ** (-diff / 400.0))

        self.adj_eff = r.copy()                       # efficiency proxy (known now)
        self.committee_prior = r.copy()
        # conference strength (fixed): mean member efficiency, per team
        self.conf_strength = np.array([
            r[self.conf_code == self.conf_code[i]].mean() for i in range(self.n)
        ])
        self._z_adj_eff = _z(self.adj_eff)
        self._z_committee = _z(self.committee_prior)
        self._z_conf_strength = _z(self.conf_strength)

        self._static_records()
        self._z_sos = _z(self.sos)

    def _static_records(self) -> None:
        n = self.n
        self.base_wins = np.zeros(n)
        self.base_conf_wins = np.zeros(n)
        self.games_played = np.zeros(n)               # total schedule games (fixed)
        self.conf_games = np.zeros(n)                 # total conf games (fixed)

        # completed games -> fixed winners/losers
        bw, bl = [], []
        for r in self.completed.itertuples():
            h, a = self.tidx[r.home_team], self.tidx[r.away_team]
            home_won = int(r.home_win) == 1
            w, loser = (h, a) if home_won else (a, h)
            bw.append(w)
            bl.append(loser)
            self.base_wins[w] += 1
            self.games_played[h] += 1
            self.games_played[a] += 1
            same = self.conf_code[h] == self.conf_code[a]
            if same:
                self.conf_games[h] += 1
                self.conf_games[a] += 1
                self.base_conf_wins[w] += 1
        self.base_winners = np.array(bw, dtype=int)
        self.base_losers = np.array(bl, dtype=int)

        # remaining games
        rh, ra, rp = [], [], []
        for r in self.remaining.itertuples():
            h, a = self.tidx[r.home_team], self.tidx[r.away_team]
            rh.append(h)
            ra.append(a)
            rp.append(float(r.home_win_prob))
            self.games_played[h] += 1
            self.games_played[a] += 1
            if self.conf_code[h] == self.conf_code[a]:
                self.conf_games[h] += 1
                self.conf_games[a] += 1
        self.rem_h = np.array(rh, dtype=int)
        self.rem_a = np.array(ra, dtype=int)
        self.rem_p = np.array(rp, dtype=float)
        self.rem_same_conf = self.conf_code[self.rem_h] == self.conf_code[self.rem_a] \
            if len(self.rem_h) else np.array([], dtype=bool)

        self.games_played = np.maximum(self.games_played, 1)
        self.conf_games_safe = np.maximum(self.conf_games, 1)

        # strength of schedule is fixed (every opponent is known now):
        # mean opponent adj-efficiency proxy over the full schedule.
        all_h = np.r_[self.base_winners, self.rem_h].astype(int)
        all_a = np.r_[self.base_losers, self.rem_a].astype(int)
        opp_sum = np.zeros(n)
        opp_cnt = np.zeros(n)
        np.add.at(opp_sum, all_h, self.adj_eff[all_a])
        np.add.at(opp_sum, all_a, self.adj_eff[all_h])
        np.add.at(opp_cnt, all_h, 1)
        np.add.at(opp_cnt, all_a, 1)
        self.sos = np.where(opp_cnt > 0, opp_sum / np.maximum(opp_cnt, 1),
                            self.adj_eff.mean())

    # --------------------------------------------------------------- one iter
    def _simulate_once(self, rng: np.random.Generator) -> dict:
        n = self.n
        pcfg = self.pcfg

        home_win = rng.random(len(self.rem_p)) < self.rem_p if len(self.rem_p) else \
            np.array([], dtype=bool)
        if len(self.rem_p):
            rem_winners = np.where(home_win, self.rem_h, self.rem_a)
            rem_losers = np.where(home_win, self.rem_a, self.rem_h)
        else:
            rem_winners = np.array([], dtype=int)
            rem_losers = np.array([], dtype=int)

        wins = self.base_wins.copy()
        np.add.at(wins, rem_winners, 1)
        conf_wins = self.base_conf_wins.copy()
        if len(self.rem_p):
            cc = self.rem_same_conf
            np.add.at(conf_wins, rem_winners[cc], 1)

        win_pct = wins / self.games_played
        conf_win_pct = conf_wins / self.conf_games_safe

        # --- conference championship games ------------------------------
        champ = np.zeros(n, dtype=bool)
        title_game = np.zeros(n, dtype=bool)
        rank_key = conf_win_pct + 1e-3 * win_pct + 1e-6 * self.adj_eff
        for ci, has in enumerate(self.conf_has_champ):
            if not has:
                continue
            members = np.where(self.conf_code == ci)[0]
            if len(members) < 2:
                continue
            order = members[np.argsort(-rank_key[members])]
            a, b = order[0], order[1]
            title_game[a] = title_game[b] = True
            winner = a if rng.random() < self.elo_P[a, b] else b
            champ[winner] = True
            wins[winner] += 1  # title-game win counts

        # --- selection score -------------------------------------------
        prov = win_pct + 1e-4 * self.adj_eff
        prov_rank = np.empty(n, dtype=int)
        prov_rank[np.argsort(-prov)] = np.arange(1, n + 1)
        qwc, blc = pcfg.quality_win_cutoff, pcfg.bad_loss_cutoff

        qwins = np.zeros(n)
        if len(self.base_winners):
            np.add.at(qwins, self.base_winners, (prov_rank[self.base_losers] <= qwc))
        if len(rem_winners):
            np.add.at(qwins, rem_winners, (prov_rank[rem_losers] <= qwc))

        blosses = np.zeros(n)
        if len(self.base_losers):
            np.add.at(blosses, self.base_losers, (prov_rank[self.base_winners] > blc))
        if len(rem_losers):
            np.add.at(blosses, rem_losers, (prov_rank[rem_winners] > blc))

        w = pcfg.weights
        score = (
            w["win_pct"] * win_pct
            + w["adjusted_efficiency"] * self._z_adj_eff
            + w["strength_of_schedule"] * self._z_sos
            + w["quality_wins"] * qwins
            + w["bad_losses"] * blosses
            + w["conference_champion"] * champ.astype(float)
            + w["conference_strength"] * self._z_conf_strength
            + w["committee_prior"] * self._z_committee
        )

        # --- select + seed 12-team field -----------------------------
        order = np.argsort(-score)
        champ_order = [i for i in order if champ[i]][: pcfg.guaranteed_conf_champs]
        selected: list[int] = list(champ_order)
        for i in order:
            if len(selected) >= pcfg.n_teams:
                break
            if i not in selected:
                selected.append(i)
        field = sorted(selected, key=lambda i: -score[i])[: pcfg.n_teams]
        seeds = {s + 1: field[s] for s in range(len(field))}
        byes = set(field[: pcfg.n_byes])

        semifinal, final_teams, champion = self._run_bracket(seeds, rng)

        return {
            "champ": champ, "title_game": title_game,
            "playoff": field, "byes": byes,
            "semifinal": semifinal, "final": final_teams, "champion": champion,
            "wins": wins,
        }

    def _run_bracket(self, seeds: dict[int, int], rng: np.random.Generator):
        P = self.elo_P
        if len(seeds) < 12:
            only = list(seeds.values())
            return set(only[:4]), set(only[:2]), set(only[:1])
        fr = {}
        for hi, lo in [(5, 12), (6, 11), (7, 10), (8, 9)]:
            a, b = seeds[hi], seeds[lo]
            fr[hi] = a if rng.random() < P[a, b] else b
        qfw = []
        for top, slot in [(1, 8), (2, 7), (3, 6), (4, 5)]:
            a, b = seeds[top], fr[slot]
            qfw.append(a if rng.random() < P[a, b] else b)
        sf = []
        for a, b in [(qfw[0], qfw[3]), (qfw[1], qfw[2])]:
            sf.append(a if rng.random() < P[a, b] else b)
        a, b = sf
        champ = a if rng.random() < P[a, b] else b
        semifinal = set(qfw)          # teams reaching the semifinal round
        return semifinal, set(sf), {champ}

    # ---------------------------------------------------------------- driver
    def run(self, n_iterations: int | None = None) -> SeasonSimResult:
        n_iter = n_iterations or int(self.settings.config["simulation"]["n_iterations"])
        rng = np.random.default_rng(self.settings.config["simulation"]["random_seed"])

        C = {k: np.zeros(self.n) for k in
             ("champ", "title_game", "playoff", "byes", "semifinal", "final", "champion")}
        win_sum = np.zeros(self.n)

        for _ in range(n_iter):
            out = self._simulate_once(rng)
            C["champ"] += out["champ"]
            C["title_game"] += out["title_game"]
            win_sum += out["wins"]
            for t in out["playoff"]:
                C["playoff"][t] += 1
            for t in out["byes"]:
                C["byes"][t] += 1
            for t in out["semifinal"]:
                C["semifinal"][t] += 1
            for t in out["final"]:
                C["final"][t] += 1
            for t in out["champion"]:
                C["champion"][t] += 1

        tp = pd.DataFrame({
            "team": self.T,
            "conference": self.conf_of_team,
            "proj_wins": (win_sum / n_iter).round(2),
            "p_conf_title_game": C["title_game"] / n_iter,
            "p_conf_champion": C["champ"] / n_iter,
            "p_playoff": C["playoff"] / n_iter,
            "p_first_round_bye": C["byes"] / n_iter,
            "p_semifinal": C["semifinal"] / n_iter,
            "p_title_game": C["final"] / n_iter,
            "p_national_champion": C["champion"] / n_iter,
        }).sort_values("p_national_champion", ascending=False).reset_index(drop=True)

        return SeasonSimResult(
            team_probabilities=tp,
            game_predictions=self._game_pred_table(),
            generated_at_utc=utc_now().isoformat(),
            n_iterations=n_iter,
            season=self.season,
            meta={"n_remaining_games": int(len(self.remaining)),
                  "n_completed_games": int(len(self.completed)),
                  "next_game": self._next_game_table().to_dict("records")},
        )

    # ---------------------------------------------------------------- tables
    def _game_pred_table(self) -> pd.DataFrame:
        cols = ["game_id", "season", "week", "date", "home_team", "away_team",
                "home_conference", "away_conference", "neutral_site",
                "home_win_prob", "pred_home_points", "pred_away_points"]
        if self.remaining.empty:
            return pd.DataFrame(columns=cols)
        t = self.remaining[cols].copy()
        t["pred_home_points"] = t["pred_home_points"].round(1)
        t["pred_away_points"] = t["pred_away_points"].round(1)
        return t.sort_values(["week", "date"]).reset_index(drop=True)

    def _next_game_table(self) -> pd.DataFrame:
        if self.remaining.empty:
            return pd.DataFrame(columns=["team", "next_opponent", "p_win", "week"])
        rem = self.remaining.sort_values("date")
        rows = []
        for t in self.T:
            g = rem[(rem["home_team"] == t) | (rem["away_team"] == t)].head(1)
            if g.empty:
                continue
            r = g.iloc[0]
            is_home = r["home_team"] == t
            p = r["home_win_prob"] if is_home else 1 - r["home_win_prob"]
            rows.append({
                "team": t,
                "next_opponent": r["away_team"] if is_home else r["home_team"],
                "next_is_home": bool(is_home and not r["neutral_site"]),
                "p_win": float(p),
                "week": int(r["week"]),
            })
        return pd.DataFrame(rows)


def run_season_simulation(
    games: pd.DataFrame,
    teams: pd.DataFrame,
    predictor: GamePredictor,
    elo: EloModel,
    settings: Settings | None = None,
    n_iterations: int | None = None,
) -> SeasonSimResult:
    return SeasonSimulator(games, teams, predictor, elo, settings).run(n_iterations)
