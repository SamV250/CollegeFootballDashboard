"""High-level façade the dashboard imports.

Wraps data loading, feature building, model loading, per-game prediction,
team rating tables, and the cached season-simulation artifact.  Pure
Python (no Streamlit) so it can be reused by scripts and tests; the UI
layer adds ``st.cache_*`` on top.
"""

from __future__ import annotations

import functools
import json
import logging
from typing import Any

import numpy as np
import pandas as pd

from src.config import Settings, get_settings
from src.data.loader import freshness, load_games, load_teams
from src.features.build import build_feature_matrix
from src.features.team_stats import (
    latest_team_ratings,
    rolling_team_features,
    team_game_log,
)
from src.models.registry import load_bundle, model_card
from src.simulation.leverage import compute_leverage
from src.simulation.season import SeasonSimulator

log = logging.getLogger(__name__)

ARTIFACT_DIR_NAME = "dashboard"


# --------------------------------------------------------------------------
# core cached loaders
# --------------------------------------------------------------------------
@functools.lru_cache(maxsize=1)
def frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    return load_games(), load_teams()


@functools.lru_cache(maxsize=1)
def bundle() -> dict[str, Any]:
    return load_bundle()


@functools.lru_cache(maxsize=1)
def feature_matrix() -> pd.DataFrame:
    games, teams = frames()
    return build_feature_matrix(games, teams, get_settings(), elo=bundle()["elo"])


def active_season() -> int:
    return get_settings().active_season()


def clear_caches() -> None:
    for fn in (frames, bundle, feature_matrix, load_dashboard_artifacts,
               latest_ratings_snapshot):
        fn.cache_clear()
    from src.models.registry import _load_bundle_cached

    _load_bundle_cached.cache_clear()


# --------------------------------------------------------------------------
# predictions
# --------------------------------------------------------------------------
def predict_games(feat_rows: pd.DataFrame) -> pd.DataFrame:
    pred = bundle()["predictor"].predict(feat_rows)
    return feat_rows.join(pred)


def upcoming_predictions(season: int | None = None) -> pd.DataFrame:
    season = season or active_season()
    feat = feature_matrix()
    upc = feat[(feat["season"] == season) & (~feat["completed"])].copy()
    if upc.empty:
        return upc
    return predict_games(upc)


def completed_predictions(season: int | None = None) -> pd.DataFrame:
    season = season or active_season()
    feat = feature_matrix()
    done = feat[(feat["season"] == season) & feat["completed"] & feat["home_win"].notna()].copy()
    if done.empty:
        return done
    return predict_games(done)


# --------------------------------------------------------------------------
# team rating tables
# --------------------------------------------------------------------------
def _percentile(s: pd.Series) -> pd.Series:
    return s.rank(pct=True) * 100.0


def team_ratings(season: int | None = None) -> pd.DataFrame:
    """One row per FBS team: model rating, Elo, unit ratings, record."""

    season = season or active_season()
    games, teams = frames()
    elo = bundle()["elo"]

    rolled = rolling_team_features(team_game_log(games), get_settings())
    latest = latest_team_ratings(rolled, season)

    rows = []
    cur = games[games["season"] == season]
    for t in get_settings().all_fbs_teams():
        played = cur[((cur["home_team"] == t) | (cur["away_team"] == t)) & cur["completed"]]
        w = losses = 0
        pf = pa = 0.0
        for r in played.itertuples():
            is_home = r.home_team == t
            tf = r.home_points if is_home else r.away_points
            ta = r.away_points if is_home else r.home_points
            pf += tf or 0
            pa += ta or 0
            if (tf or 0) > (ta or 0):
                w += 1
            else:
                losses += 1
        lr = latest[latest["team"] == t]
        off = float(lr["exp_off_epa_per_play"].iloc[0]) if not lr.empty and "exp_off_epa_per_play" in lr else np.nan
        deff = float(lr["exp_def_epa_prevention"].iloc[0]) if not lr.empty and "exp_def_epa_prevention" in lr else np.nan
        st = float(lr["exp_st_epa"].iloc[0]) if not lr.empty and "exp_st_epa" in lr else np.nan
        rows.append({
            "team": t,
            "conference": get_settings().conference_of(t),
            "wins": w, "losses": losses,
            "record": f"{w}-{losses}",
            "elo": round(elo.rating(t), 1),
            "model_rating": round((elo.rating(t) - 1500) / 25.0, 2),
            "points_for_pg": round(pf / max(w + losses, 1), 1),
            "points_against_pg": round(pa / max(w + losses, 1), 1),
            "off_epa": off,
            "def_epa_prevention": deff,
            "st_epa": st,
            "games_played": w + losses,
        })
    df = pd.DataFrame(rows)
    df["offense_pctl"] = _percentile(df["off_epa"])
    df["defense_pctl"] = _percentile(df["def_epa_prevention"])
    df["special_teams_pctl"] = _percentile(df["st_epa"])
    df["overall_pctl"] = _percentile(df["model_rating"])
    df = df.sort_values("model_rating", ascending=False).reset_index(drop=True)
    df["model_rank"] = np.arange(1, len(df) + 1)
    return df


@functools.lru_cache(maxsize=1)
def latest_ratings_snapshot() -> pd.DataFrame:
    games, _ = frames()
    rolled = rolling_team_features(team_game_log(games), get_settings())
    return latest_team_ratings(rolled, active_season())


def simulate_matchup(
    home: str, away: str, neutral: bool = False, adjustments: dict | None = None
) -> dict:
    """Predict a hypothetical matchup and explain it.

    Returns a dict with the prediction, the full explanation, and an echo
    of the user adjustments applied (kept separate from model output).
    """

    from src.explainability.shap_explain import explain_game
    from src.features.build import synthesize_matchup

    b = bundle()
    row = synthesize_matchup(home, away, latest_ratings_snapshot(), b["elo"],
                             neutral=neutral, adjustments=adjustments or {})
    pred = b["predictor"].predict(row).iloc[0]
    exp = explain_game(b["predictor"], row, pred)
    return {"prediction": pred.to_dict(), "explanation": exp,
            "adjustments": adjustments or {}, "neutral": neutral}


def tier_label(pctl: float) -> str:
    if pd.isna(pctl):
        return "No data"
    if pctl >= 85:
        return "Elite"
    if pctl >= 60:
        return "Above Average"
    if pctl >= 35:
        return "Average"
    return "Below Average"


# --------------------------------------------------------------------------
# dashboard artifact (season simulation + leverage), persisted
# --------------------------------------------------------------------------
def _artifact_dir(settings: Settings) -> Any:
    d = settings.processed_dir / ARTIFACT_DIR_NAME
    d.mkdir(parents=True, exist_ok=True)
    return d


def build_dashboard_artifacts(
    settings: Settings | None = None,
    n_iterations: int | None = None,
    with_leverage: bool = True,
) -> dict[str, Any]:
    settings = settings or get_settings()
    games, teams = load_games(settings), load_teams(settings)
    b = load_bundle(settings)
    sim = SeasonSimulator(games, teams, b["predictor"], b["elo"], settings)
    result = sim.run(n_iterations=n_iterations)

    d = _artifact_dir(settings)
    result.team_probabilities.to_parquet(d / "team_probabilities.parquet", index=False)
    result.game_predictions.to_parquet(d / "game_predictions.parquet", index=False)

    leverage = pd.DataFrame()
    if with_leverage:
        try:
            leverage = compute_leverage(sim)
            leverage.to_parquet(d / "leverage.parquet", index=False)
        except Exception as exc:  # noqa: BLE001
            log.warning("leverage computation failed: %s", exc)

    summary = {
        "generated_at_utc": result.generated_at_utc,
        "season": result.season,
        "n_iterations": result.n_iterations,
        "meta": result.meta,
        "model_card": model_card(settings),
        "freshness": freshness(settings),
    }
    (d / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    load_dashboard_artifacts.cache_clear()
    return summary


def run_scenario_simulation(
    forced: dict[str, str] | None = None,
    prob_overrides: dict[str, float] | None = None,
    n_iterations: int = 4000,
) -> dict[str, Any]:
    """Re-run the season simulation with user-forced results / probabilities.

    ``forced``: game_id -> "home" | "away" (that team wins for certain).
    ``prob_overrides``: game_id -> home win probability in [0, 1].
    Returns team_probabilities plus a projected 12-team bracket.
    """

    forced = forced or {}
    prob_overrides = prob_overrides or {}
    games, teams = frames()
    b = bundle()
    sim = SeasonSimulator(games, teams, b["predictor"], b["elo"], get_settings())

    m = sim.remaining["game_id"]
    for gid, side in forced.items():
        sim.remaining.loc[m == gid, "home_win_prob"] = 0.999 if side == "home" else 0.001
    for gid, p in prob_overrides.items():
        sim.remaining.loc[m == gid, "home_win_prob"] = float(np.clip(p, 0.001, 0.999))
    sim._static_records()

    res = sim.run(n_iterations=n_iterations)
    tp = res.team_probabilities
    bracket = _project_bracket(tp)
    return {"team_probabilities": tp, "bracket": bracket,
            "n_iterations": n_iterations, "meta": res.meta}


def _project_bracket(tp: pd.DataFrame) -> pd.DataFrame:
    """A single representative 12-team field from simulation probabilities."""

    from src.simulation.playoff import PlayoffConfig

    cfg = PlayoffConfig.from_settings(get_settings())
    ranked = tp.sort_values("p_playoff", ascending=False).copy()
    champ_pick = (tp.sort_values("p_conf_champion", ascending=False)
                  .groupby("conference").head(1))
    champ_pick = champ_pick[champ_pick["p_conf_champion"] > 0.15]
    guaranteed = list(champ_pick.sort_values("p_playoff", ascending=False)
                      .head(cfg.guaranteed_conf_champs)["team"])
    field = list(guaranteed)
    for t in ranked["team"]:
        if len(field) >= cfg.n_teams:
            break
        if t not in field:
            field.append(t)
    out = tp[tp["team"].isin(field)].copy()
    out = out.sort_values("p_playoff", ascending=False).reset_index(drop=True)
    out["seed"] = np.arange(1, len(out) + 1)
    out["bye"] = out["seed"] <= cfg.n_byes
    out["auto_bid_conf_champ"] = out["team"].isin(guaranteed)
    return out[["seed", "team", "conference", "bye", "auto_bid_conf_champ",
               "p_playoff", "p_national_champion"]]


@functools.lru_cache(maxsize=1)
def load_dashboard_artifacts() -> dict[str, Any]:
    """Read the persisted artifact; build a quick one if missing."""

    settings = get_settings()
    d = _artifact_dir(settings)
    sfile = d / "summary.json"
    if not sfile.exists():
        log.info("no dashboard artifact -- building a quick simulation")
        build_dashboard_artifacts(settings, n_iterations=2000, with_leverage=False)

    summary = json.loads((d / "summary.json").read_text())
    out = {
        "summary": summary,
        "team_probabilities": pd.read_parquet(d / "team_probabilities.parquet"),
        "game_predictions": pd.read_parquet(d / "game_predictions.parquet"),
    }
    lev = d / "leverage.parquet"
    out["leverage"] = pd.read_parquet(lev) if lev.exists() else pd.DataFrame()
    return out


def dashboard_freshness() -> dict[str, Any]:
    return freshness()


def get_model_card() -> dict[str, Any]:
    return model_card()
