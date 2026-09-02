"""End-to-end training orchestration with strict chronological splits.

    train        : seasons <= config.model.train_through_season
    calibration  : validation_season, weeks 1-8   (fits the isotonic map)
    validation   : validation_season, weeks 9+    (never seen in fit/calib)
    test         : test_season, completed games   (true out-of-sample)

Elo is fit on the full chronological game list first -- its *pre-game*
ratings are leakage-safe by construction and become a feature.  Baselines
(home win rate, logistic, Elo) are evaluated on exactly the same
validation / test rows as the primary model so the comparison is fair.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from src.config import Settings, get_settings
from src.data.loader import load_games, load_teams
from src.features.build import build_feature_matrix, feature_columns
from src.models.baselines import HomeWinRateBaseline, LogisticBaseline
from src.models.elo import EloModel
from src.models.evaluate import (
    calibration_table,
    location_split,
    performance_by_confidence,
    performance_by_group,
    score_metrics,
    win_metrics,
)
from src.models.game_model import GamePredictor
from src.models.registry import save_bundle

log = logging.getLogger(__name__)


def _split(feat: pd.DataFrame, settings: Settings) -> dict[str, pd.DataFrame]:
    """Strictly chronological splits.

    train        : seasons <= train_through_season (calibration is done
                   with k-fold CV *inside* this split, no leakage)
    validation   : the full validation_season (never seen in fit/calib)
    test         : the test_season, completed games only
    """

    m = settings.config["model"]
    labeled = feat[feat["home_win"].notna()].copy()
    train = labeled[labeled["season"] <= m["train_through_season"]]
    val = labeled[labeled["season"] == m["validation_season"]]
    test = labeled[labeled["season"] == m["test_season"]]
    return {"train": train, "calibration": None, "validation": val, "test": test}


def _score_frame(pred: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "margin_true": truth["point_margin"].values,
        "margin_pred": pred["pred_margin"].values,
        "home_true": truth["home_points"].values,
        "home_pred": pred["pred_home_points"].values,
        "away_true": truth["away_points"].values,
        "away_pred": pred["pred_away_points"].values,
    })


def run_training(settings: Settings | None = None) -> dict[str, Any]:
    settings = settings or get_settings()
    games = load_games(settings)
    teams = load_teams(settings)

    # --- Elo on the full chronological history --------------------------
    elo = EloModel(settings=settings).fit(games, teams)

    # --- feature matrix + splits ---------------------------------------
    feat = build_feature_matrix(games, teams, settings, elo=elo)
    feats = feature_columns()
    parts = _split(feat, settings)
    log.info("split sizes: %s",
             {k: (0 if v is None else len(v)) for k, v in parts.items()})
    if parts["train"].empty or parts["test"].empty:
        raise RuntimeError("Not enough labeled data to train. Check data refresh.")

    # --- primary model ------------------------------------------------
    predictor = GamePredictor(features=feats, settings=settings)
    predictor.fit(parts["train"], calib=parts["calibration"])

    # --- baselines --------------------------------------------------
    hwr = HomeWinRateBaseline().fit(parts["train"]["home_win"])
    logit = LogisticBaseline().fit(parts["train"], parts["train"]["home_win"])

    # --- evaluate on validation and test ----------------------------
    evaluation: dict[str, Any] = {
        "splits": {k: (0 if v is None else int(len(v))) for k, v in parts.items()}
    }
    for split_name in ("validation", "test"):
        part = parts[split_name]
        if part.empty:
            continue
        pred = predictor.predict(part)
        y = part["home_win"].astype(int).values
        market = (part["elo_diff"] > 0).astype(int).values  # Elo as proxy market

        ensemble_p = pred["home_win_prob"].values            # blended (primary)
        gbm_only = predictor.win_probability(part, blend=False)
        base = {
            "ensemble": win_metrics(y, ensemble_p, market_pick=market),
            "gbm": win_metrics(y, gbm_only, market_pick=market),
            "elo": win_metrics(y, part["elo_home_win_prob"].values, market_pick=market),
            "logistic": win_metrics(y, logit.predict_proba(part), market_pick=market),
            "home_rate": win_metrics(y, hwr.predict_proba(len(part))),
        }
        scores = score_metrics(_score_frame(pred, part))
        cal = calibration_table(y, pred["home_win_prob"].values)
        by_conf_frame = part.assign(home_win_prob=pred["home_win_prob"].values)
        evaluation[split_name] = {
            "win_model": base,
            "score_model": scores,
            "calibration_table": cal.to_dict("records"),
            "by_confidence": performance_by_confidence(
                y, pred["home_win_prob"].values).to_dict("records"),
            "by_conference": performance_by_group(
                by_conf_frame, "home_conference").to_dict("records"),
            "by_location": location_split(by_conf_frame).to_dict("records"),
        }

    # --- per-season backtest table --------------------------------
    season_rows = []
    for season, grp in feat[feat["home_win"].notna()].groupby("season"):
        if season <= settings.config["model"]["train_through_season"]:
            continue
        if len(grp) < 40:  # skip a barely-started current season
            continue
        p = predictor.predict(grp)
        season_rows.append({
            "season": int(season),
            **{k: win_metrics(grp["home_win"].astype(int), p["home_win_prob"])[k]
               for k in ("n", "accuracy", "log_loss", "brier", "calibration_error")},
        })
    evaluation["by_season"] = season_rows

    path = save_bundle(predictor, elo, evaluation, settings,
                       extra={"data_rows": int(len(games)),
                              "completed_games": int(games["completed"].sum())})
    log.info("saved model bundle -> %s", path)
    return {"bundle": str(path), "evaluation": evaluation}
