"""Synthetic-but-realistic data generator.

This lets the entire dashboard, modeling pipeline and test-suite run with
**zero API credentials**.  The generator:

* uses the real 2026 FBS conference alignment from ``config/conferences.yaml``;
* gives every team stable latent offensive / defensive quality plus latent
  quality on each advanced metric;
* builds a plausible 12-13 game schedule per team per season (conference
  round-robin sample + non-conference games);
* simulates scores and a full set of advanced box-score metrics that are
  *correlated with, but noisier than*, the latent quality -- exactly the
  situation the real model must cope with;
* marks prior seasons complete and the current season complete only
  through ``demo.as_of_week``.

The output schema is identical to what the CFBD / ESPN adapters emit, so
nothing downstream knows or cares that the data is synthetic.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from src.config import Settings, get_settings

# Per-game advanced metrics.  Each is generated for home and away with a
# ``home_`` / ``away_`` prefix.  ``higher_is_better`` is from the metric
# owner's perspective and is only used for narrative direction defaults.
ADVANCED_METRICS: dict[str, bool] = {
    "off_epa_per_play": True,
    "def_epa_per_play": False,     # EPA allowed -- lower is better
    "success_rate": True,
    "def_success_rate": False,
    "explosiveness": True,
    "def_explosiveness": False,
    "pass_epa_per_play": True,
    "rush_epa_per_play": True,
    "havoc": True,
    "def_havoc": False,            # havoc allowed
    "sack_rate": True,
    "sack_rate_allowed": False,
    "finish_pts_per_opp": True,    # red-zone / scoring-opportunity finishing
    "def_finish_pts_per_opp": False,
    "line_yards": True,            # OL push
    "def_line_yards": False,
    "st_epa": True,                # special teams
    "pace_sec_per_play": True,     # seconds/play -- treated as neutral info
}

def _season_start(year: int) -> datetime:
    """Approx. Thursday before Labor Day (Week 1 kickoff) for any season."""

    d = datetime(year, 9, 1, tzinfo=timezone.utc)
    # step back to the nearest Thursday (weekday 3) on/just before Sep 1
    return d - timedelta(days=(d.weekday() - 3) % 7)


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


# Rough "points above an average FBS team" priors for well-known programs,
# so the SYNTHETIC demo still produces a believable pecking order.  These
# are coarse, hand-set flavour values -- not a projection of any real
# season -- and every team not listed falls back to its conference tier.
PRESEASON_STRENGTH: dict[str, float] = {
    "Georgia": 21, "Ohio State": 20, "Texas": 19, "Oregon": 18, "Alabama": 18,
    "Penn State": 16, "Notre Dame": 16, "Michigan": 14, "Ole Miss": 13,
    "Tennessee": 13, "LSU": 13, "Clemson": 13, "Miami": 12, "Texas A&M": 11,
    "South Carolina": 11, "Oklahoma": 11, "Missouri": 10, "Indiana": 10,
    "Illinois": 9, "Kansas State": 9, "SMU": 9, "Utah": 9, "BYU": 8,
    "Louisville": 8, "Florida": 8, "Iowa State": 8, "Texas Tech": 8,
    "Nebraska": 7, "USC": 7, "Auburn": 7, "Arizona State": 7, "Baylor": 6,
    "Michigan State": 5, "Washington": 6, "TCU": 6, "Florida State": 6,
    "Kansas": 5, "Iowa": 5, "Duke": 4, "Tulane": 4, "Boise State": 5,
    "Memphis": 3, "Navy": 3, "Army": 3, "James Madison": 4, "Toledo": 2,
    "UNLV": 2, "Liberty": 2, "Georgia Tech": 4, "Vanderbilt": 3,
    "Pittsburgh": 3, "Wisconsin": 3, "Minnesota": 4, "Colorado": 3,
}


def build_team_frame(settings: Settings, seed: int = 1729) -> pd.DataFrame:
    """Latent team-quality table (stable across the demo's seasons)."""

    rng = _rng(seed)
    rows = []
    tier_base = {"power": 2.0, "group_of_five": -6.0, "independent": 0.0}
    for conf, meta in settings.conferences["conferences"].items():
        tier = meta.get("tier", "group_of_five")
        for team in meta["teams"]:
            # Latent points-above-average on offense and defense.
            prior = PRESEASON_STRENGTH.get(team, tier_base.get(tier, 0.0))
            base = prior + rng.normal(0, 3.0)
            off = base * 0.55 + rng.normal(0, 3.0)
            deff = base * 0.55 + rng.normal(0, 3.0)
            row = {
                "team": team,
                "conference": conf,
                "tier": tier,
                "fbs": True,
                "latent_offense": round(off, 3),
                "latent_defense": round(deff, 3),
                "latent_pace": round(rng.normal(27.0, 1.8), 2),
            }
            # Latent quality on each advanced metric.  Each metric is
            # tied to overall strength (so rolling averages are
            # predictive) *plus* an independent "style" component the Elo
            # rating cannot see -- which is what lets the matchup model
            # add value over Elo.
            for metric, hib in ADVANCED_METRICS.items():
                tie = (off if "def_" not in metric and metric != "sack_rate_allowed"
                       else -deff)
                row[f"latent_{metric}"] = rng.normal(tie * 0.16, 0.55)
            rows.append(row)
    return pd.DataFrame(rows)


def _round_robin(members: list[str]) -> list[list[tuple[str, str]]]:
    """Circle-method round robin -> list of rounds, each a list of pairs.

    Every team plays exactly once per round and no pairing repeats.
    """

    ms = list(members)
    if len(ms) % 2:
        ms.append("__BYE__")
    n = len(ms)
    rounds: list[list[tuple[str, str]]] = []
    for _ in range(n - 1):
        pairs = []
        for i in range(n // 2):
            a, b = ms[i], ms[n - 1 - i]
            if a != "__BYE__" and b != "__BYE__":
                pairs.append((a, b))
        rounds.append(pairs)
        ms = [ms[0]] + [ms[-1]] + ms[1:-1]  # rotate, keeping ms[0] fixed
    return rounds


def _season_schedule(
    teams: pd.DataFrame, season: int, rng: np.random.Generator
) -> list[dict]:
    """Create a clean schedule: ~9 conference games + ~3 non-conference.

    Guarantees: no duplicate pairings, and no team plays twice in a week.
    """

    n_weeks = 13
    week_slots: list[list[dict]] = [[] for _ in range(n_weeks + 1)]  # 1-indexed
    busy: list[set[str]] = [set() for _ in range(n_weeks + 1)]
    conf_of = dict(zip(teams["team"], teams["conference"]))
    games: list[dict] = []

    def place(home: str, away: str, conf_game: bool, prefer: list[int]) -> bool:
        for wk in (int(w) for w in prefer):
            if home in busy[wk] or away in busy[wk] or len(week_slots[wk]) >= 75:
                continue
            g = dict(home=home, away=away, conf_game=conf_game,
                     home_conf=conf_of[home], away_conf=conf_of[away], week=wk)
            week_slots[wk].append(g)
            busy[wk].update({home, away})
            games.append(g)
            return True
        return False

    # --- conference games ------------------------------------------------
    by_conf = teams.groupby("conference")["team"].apply(list).to_dict()
    for members in by_conf.values():
        members = list(members)
        rng.shuffle(members)
        if len(members) < 2:
            continue
        rounds = _round_robin(members)
        target_rounds = min(len(rounds), 9 if len(members) >= 10 else len(members) - 1)
        for r_idx in range(target_rounds):
            for a, b in rounds[r_idx]:
                if rng.random() < 0.5:
                    a, b = b, a
                order = list(range(2 + r_idx, n_weeks + 1)) + list(range(1, 2 + r_idx))
                place(a, b, True, order)

    # --- non-conference games -----------------------------------------------
    all_teams = list(teams["team"])
    nc_degree = {t: 0 for t in all_teams}
    conf_count = teams.groupby("conference")["team"].count().to_dict()
    max_deg = {t: (11 if conf_count.get(conf_of[t], 0) < 4 else 3) for t in all_teams}
    seen: set[frozenset[str]] = set()
    attempts = 0
    while attempts < len(all_teams) * 80:
        attempts += 1
        a, b = (str(x) for x in rng.choice(all_teams, size=2, replace=False))
        if conf_of[a] == conf_of[b] or frozenset((a, b)) in seen:
            continue
        if nc_degree[a] >= max_deg[a] or nc_degree[b] >= max_deg[b]:
            continue
        order = list(rng.permutation(range(1, n_weeks + 1)))
        if place(a, b, False, order):
            seen.add(frozenset((a, b)))
            nc_degree[a] += 1
            nc_degree[b] += 1

    # --- dates + neutral flag ------------------------------------------------
    start = _season_start(season)
    for g in games:
        wk = g["week"]
        dow = int(rng.choice([1, 2, 2, 2, 3], p=[.08, .12, .55, .12, .13]))
        offset_days = (wk - 1) * 7 + dow
        kickoff = start + timedelta(days=offset_days, hours=int(rng.integers(16, 28)))
        g["season"] = season
        g["date"] = kickoff.astimezone(timezone.utc)
        g["neutral_site"] = bool(rng.random() < 0.03 and not g["conf_game"])
        g["season_type"] = "regular"
    return games


def _simulate_game(
    g: dict, lat: dict[str, dict], rng: np.random.Generator, hfa_pts: float
) -> dict:
    """Fill in scores + advanced metrics for one scheduled game."""

    h, a = g["home"], g["away"]
    lh, la = lat[h], lat[a]
    neutral = g["neutral_site"]
    hfa = 0.0 if neutral else hfa_pts

    # Expected points from latent quality (offense vs opposing defense)
    # plus a secondary style-matchup term: a team that is explosive and
    # efficient through the air gains extra when facing a defense weak in
    # exactly those areas.  Elo cannot represent this; the matchup model
    # can, via its cross-features.
    def style(o: dict, d: dict) -> float:
        # Additive terms (any linear model can learn these) ...
        add = (
            1.4 * (o["latent_explosiveness"] - d["latent_def_explosiveness"])
            + 1.2 * (o["latent_pass_epa_per_play"] + d["latent_def_epa_per_play"])
            + 0.9 * (o["latent_rush_epa_per_play"] - d["latent_def_line_yards"])
        )
        # ... plus genuine interactions: an explosive passing attack is
        # worth far more against a defense that is *also* weak at
        # preventing explosives, and pass-rush only swings games when the
        # opposing pass protection is poor.  Tree models capture these;
        # a logistic model on raw differences cannot.
        interact = (
            1.9 * np.tanh(o["latent_explosiveness"]) * np.tanh(-d["latent_def_explosiveness"])
            + 1.6 * np.tanh(o["latent_sack_rate"]) * np.tanh(d["latent_sack_rate_allowed"])
            + 1.1 * np.tanh(o["latent_line_yards"]) * np.tanh(-d["latent_def_havoc"])
        )
        return add + interact

    exp_home = 27 + lh["latent_offense"] - la["latent_defense"] + hfa + style(lh, la)
    exp_away = 27 + la["latent_offense"] - lh["latent_defense"] + style(la, lh)
    home_pts = max(0, int(round(rng.normal(exp_home, 9.0))))
    away_pts = max(0, int(round(rng.normal(exp_away, 9.0))))
    if home_pts == away_pts:  # college football has no ties
        if rng.random() < 0.5:
            home_pts += 3
        else:
            away_pts += 3

    perf_home = (home_pts - away_pts) / 14.0  # standardized game script
    out = dict(g)
    out["completed"] = True
    out["home_points"] = home_pts
    out["away_points"] = away_pts

    for metric, _hib in ADVANCED_METRICS.items():
        if metric == "pace_sec_per_play":
            out[f"home_{metric}"] = round(lh["latent_pace"] + rng.normal(0, 1.4), 2)
            out[f"away_{metric}"] = round(la["latent_pace"] + rng.normal(0, 1.4), 2)
            continue
        # metric = latent quality + game-script influence + game noise
        h_val = lh[f"latent_{metric}"] + 0.30 * perf_home + rng.normal(0, 0.55)
        a_val = la[f"latent_{metric}"] - 0.30 * perf_home + rng.normal(0, 0.55)
        out[f"home_{metric}"] = round(float(h_val), 4)
        out[f"away_{metric}"] = round(float(a_val), 4)

    # Turnovers: partly skill, mostly luck (regression-to-mean target).
    out["home_turnovers"] = int(rng.poisson(1.4 - 0.15 * perf_home))
    out["away_turnovers"] = int(rng.poisson(1.4 + 0.15 * perf_home))
    out["home_takeaways"] = out["away_turnovers"]
    out["away_takeaways"] = out["home_turnovers"]
    return out


def generate_dataset(
    settings: Settings | None = None,
    seasons: list[int] | None = None,
    as_of: datetime | None = None,
) -> dict[str, pd.DataFrame]:
    """Return ``{"teams": df, "games": df}`` for the demo universe."""

    settings = settings or get_settings()
    cfg = settings.config
    seasons = seasons or sorted(
        set(cfg["season"]["backtest_seasons"] + [cfg["season"]["current_season"]])
    )
    current = cfg["season"]["current_season"]
    as_of_week = int(cfg.get("demo", {}).get("as_of_week", 6))

    teams = build_team_frame(settings)
    lat = teams.set_index("team").to_dict("index")
    hfa_pts = float(settings.config["elo"]["home_advantage"]) / 25.0  # ~2.2 pts

    all_rows: list[dict] = []
    for season in seasons:
        rng = _rng(settings.config["model"]["random_seed"] + season)
        sched = _season_schedule(teams, season, rng)
        for g in sched:
            played = season < current or (season == current and g["week"] <= as_of_week)
            if played:
                all_rows.append(_simulate_game(g, lat, rng, hfa_pts))
            else:
                row = dict(g)
                row["completed"] = False
                row["home_points"] = np.nan
                row["away_points"] = np.nan
                all_rows.append(row)

    games = pd.DataFrame(all_rows)
    games = games.sort_values(["season", "week", "date"]).reset_index(drop=True)
    games["game_id"] = [
        f"{r.season}-{r.week:02d}-{r.home}-vs-{r.away}".replace(" ", "_")
        for r in games.itertuples()
    ]
    # canonical column order
    lead = [
        "game_id", "season", "week", "date", "season_type", "home_team",
        "away_team", "home_conference", "away_conference", "neutral_site",
        "completed", "home_points", "away_points",
    ]
    games = games.rename(
        columns={"home": "home_team", "away": "away_team",
                 "home_conf": "home_conference", "away_conf": "away_conference"}
    )
    ordered = lead + [c for c in games.columns if c not in lead and c != "conf_game"]
    games = games[ordered]
    return {"teams": teams, "games": games}
