"""SHAP-based game explanations, translated into football language.

``explain_game`` returns a structured explanation for a single matchup:

* the five factors most helping the favourite
* the three factors giving the underdog a chance
* the model's confidence level and a plausible score range
* important missing data
* whether the prediction moved a lot in the last week
* a short plain-language summary

Raw SHAP values are converted to directional magnitude words via
:mod:`src.explainability.narrative`; they are never surfaced bare.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.explainability.narrative import (
    confidence_label,
    describe_contribution,
    outcome_range,
    plain_language_summary,
)
from src.models.game_model import GamePredictor


def _missing_data_notes(row: pd.Series, feature_cols: list[str]) -> list[str]:
    notes = []
    if row.get("home_games_played", 99) < 2 or row.get("away_games_played", 99) < 2:
        notes.append(
            "One or both teams have played very few games this season, so the "
            "forecast leans heavily on preseason priors and is less certain."
        )
    return notes


def explain_game(
    predictor: GamePredictor,
    game_row: pd.DataFrame,
    prediction: pd.Series,
    prev_prediction: float | None = None,
) -> dict:
    """``game_row`` is a single-row DataFrame from the feature matrix."""

    feats = predictor.features
    base, sv = predictor.shap_values(game_row)
    sv = np.asarray(sv)
    sv = sv[-1] if sv.ndim == 2 else sv.reshape(-1)   # -> (n_features,)
    if len(sv) != len(feats):                          # last-resort guard
        sv = sv[: len(feats)] if len(sv) > len(feats) else np.pad(sv, (0, len(feats) - len(sv)))
    contribs = pd.DataFrame({"feature": feats, "shap": sv})
    # magnitude is expressed relative to the single largest factor in this
    # game, so the top reason reads as a "major advantage" and the rest
    # scale down from there.
    scale = np.abs(sv).max() or 1.0

    home = game_row.iloc[0]["home_team"]
    away = game_row.iloc[0]["away_team"]
    p_home = float(prediction["home_win_prob"])
    favor_home = p_home >= 0.5

    # positive shap -> favours home; flip sign when the away team is favoured
    contribs["helps_favorite"] = contribs["shap"] if favor_home else -contribs["shap"]
    helps_fav = contribs.sort_values("helps_favorite", ascending=False)
    helps_dog = contribs.sort_values("helps_favorite", ascending=True)

    fav_factors = [
        describe_contribution(r.feature, r.helps_favorite if favor_home else -r.helps_favorite,
                              scale, home, away)
        for r in helps_fav.head(5).itertuples()
        if r.helps_favorite > 1e-6
    ]
    dog_factors = [
        describe_contribution(r.feature, r.helps_favorite if favor_home else -r.helps_favorite,
                              scale, home, away)
        for r in helps_dog.head(3).itertuples()
        if r.helps_favorite < -1e-6
    ]

    pred_margin = float(prediction["pred_margin"])
    lo, hi = outcome_range(pred_margin)

    moved = None
    if prev_prediction is not None:
        delta = abs(p_home - prev_prediction)
        moved = {
            "changed_substantially": bool(delta >= 0.10),
            "delta_win_prob": round(p_home - prev_prediction, 3),
        }

    summary = plain_language_summary(
        home, away, p_home,
        float(prediction["pred_home_points"]),
        float(prediction["pred_away_points"]),
        [f for f in fav_factors if f["favors"] == ("home" if favor_home else "away")],
        [f for f in dog_factors],
    )

    return {
        "home_team": home,
        "away_team": away,
        "favorite": home if favor_home else away,
        "underdog": away if favor_home else home,
        "home_win_prob": p_home,
        "favorite_win_prob": p_home if favor_home else 1 - p_home,
        "upset_prob": (1 - p_home) if favor_home else p_home,
        "pred_home_points": round(float(prediction["pred_home_points"]), 1),
        "pred_away_points": round(float(prediction["pred_away_points"]), 1),
        "pred_margin": round(pred_margin, 1),
        "confidence": confidence_label(p_home),
        "plausible_margin_range": (round(lo, 1), round(hi, 1)),
        "favorite_factors": fav_factors,
        "underdog_factors": dog_factors,
        "missing_data": _missing_data_notes(game_row.iloc[0], feats),
        "prediction_movement": moved,
        "summary": summary,
        "shap_base_value": float(base),
    }
