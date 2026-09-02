"""Model evaluation: honest, out-of-sample, baseline-relative.

Nothing here is ever computed on the training split.  All functions take
already-held-out predictions.  The headline rule the UI repeats: *a 70%
prediction is supposed to be wrong about 30% of the time* -- so
calibration error sits next to accuracy everywhere.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)


def expected_calibration_error(
    y_true: np.ndarray, prob: np.ndarray, n_bins: int = 10
) -> float:
    """Weighted mean gap between confidence and accuracy across bins."""

    y_true = np.asarray(y_true, dtype=float)
    prob = np.asarray(prob, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(prob, bins[1:-1])
    ece = 0.0
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        conf = prob[mask].mean()
        acc = y_true[mask].mean()
        ece += (mask.mean()) * abs(conf - acc)
    return float(ece)


def win_metrics(y_true, prob, market_pick: np.ndarray | None = None) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    prob = np.clip(np.asarray(prob, dtype=float), 1e-6, 1 - 1e-6)
    pred = (prob >= 0.5).astype(int)
    out = {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "log_loss": float(log_loss(y_true, prob, labels=[0, 1])),
        "brier": float(brier_score_loss(y_true, prob)),
        "roc_auc": float(roc_auc_score(y_true, prob)) if len(set(y_true)) > 1 else float("nan"),
        "calibration_error": expected_calibration_error(y_true, prob),
        "base_rate": float(y_true.mean()),
    }
    if market_pick is not None:
        mp = np.asarray(market_pick, dtype=int)
        out["accuracy_vs_market_favorite"] = float((pred == mp).mean())
    return out


def score_metrics(df: pd.DataFrame) -> dict:
    """``df`` needs columns: margin_true, margin_pred, home_true, home_pred,
    away_true, away_pred."""

    m_err = (df["margin_pred"] - df["margin_true"]).abs()
    team_true = np.r_[df["home_true"].values, df["away_true"].values]
    team_pred = np.r_[df["home_pred"].values, df["away_pred"].values]
    within = {f"within_{k}": float((m_err <= k).mean()) for k in (3, 7, 10, 14)}
    return {
        "n": int(len(df)),
        "mae_point_margin": float(m_err.mean()),
        "mae_team_score": float(mean_absolute_error(team_true, team_pred)),
        "rmse_point_margin": float(np.sqrt(mean_squared_error(df["margin_true"], df["margin_pred"]))),
        **within,
    }


def calibration_table(y_true, prob, n_bins: int = 10) -> pd.DataFrame:
    y_true = np.asarray(y_true, dtype=float)
    prob = np.asarray(prob, dtype=float)
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(prob, bins[1:-1]), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        rows.append({
            "bin_lo": bins[b],
            "bin_hi": bins[b + 1],
            "mean_predicted": float(prob[mask].mean()),
            "actual_win_rate": float(y_true[mask].mean()),
            "n_games": int(mask.sum()),
        })
    return pd.DataFrame(rows)


def performance_by_confidence(y_true, prob, edges=(0.5, 0.6, 0.7, 0.8, 0.9, 1.01)) -> pd.DataFrame:
    y_true = np.asarray(y_true, dtype=float)
    prob = np.asarray(prob, dtype=float)
    conf = np.where(prob >= 0.5, prob, 1 - prob)
    correct = ((prob >= 0.5).astype(int) == y_true).astype(float)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf >= lo) & (conf < hi)
        if not mask.any():
            continue
        rows.append({
            "confidence_range": f"{int(lo * 100)}-{int(min(hi, 1) * 100)}%",
            "n_games": int(mask.sum()),
            "model_confidence": float(conf[mask].mean()),
            "actual_accuracy": float(correct[mask].mean()),
        })
    return pd.DataFrame(rows)


def performance_by_group(
    frame: pd.DataFrame, group_col: str, prob_col: str = "home_win_prob",
    truth_col: str = "home_win",
) -> pd.DataFrame:
    cols = [group_col, "n", "accuracy", "log_loss", "brier", "calibration_error"]
    rows = []
    for key, grp in frame.groupby(group_col):
        if grp[truth_col].notna().sum() < 5:
            continue
        m = win_metrics(grp[truth_col], grp[prob_col])
        rows.append({group_col: key, **{k: m[k] for k in
                    ("n", "accuracy", "log_loss", "brier", "calibration_error")}})
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)


def location_split(frame: pd.DataFrame) -> pd.DataFrame:
    def bucket(r):
        if r["neutral_site"]:
            return "neutral"
        return "home_favorite" if r["home_win_prob"] >= 0.5 else "road_favorite"

    tmp = frame.copy()
    tmp["location_bucket"] = tmp.apply(bucket, axis=1)
    return performance_by_group(tmp, "location_bucket")
