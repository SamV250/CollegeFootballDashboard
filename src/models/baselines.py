"""Transparent baseline models.

These exist so we can honestly answer "does the gradient-boosted model
actually add value?"  If a one-line heuristic is within a hair of the
big model on log loss, the big model is not earning its complexity.

* :class:`HomeWinRateBaseline` -- predict the historical home win rate
  for every game.  The floor any real model must clear.
* :class:`LogisticBaseline` -- L2 logistic regression on a small, stable
  slice of the feature set (Elo gap, adjusted-rating gap, recent form,
  home indicator).  Interpretable coefficients, naturally calibrated.

The Elo baseline lives in :mod:`src.models.elo`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

LOGISTIC_FEATURES = [
    "elo_diff",
    "adj_rating_diff",
    "recent_form_diff",
    "off_epa_diff",
    "def_epa_prevention_diff",
    "home_indicator",
]


@dataclass
class HomeWinRateBaseline:
    home_win_rate: float = 0.5

    def fit(self, y: pd.Series | np.ndarray) -> "HomeWinRateBaseline":
        self.home_win_rate = float(np.mean(y))
        return self

    def predict_proba(self, n: int) -> np.ndarray:
        return np.full(n, self.home_win_rate)


@dataclass
class LogisticBaseline:
    features: list[str] = field(default_factory=lambda: list(LOGISTIC_FEATURES))
    pipeline: Pipeline | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LogisticBaseline":
        self.pipeline = Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(C=1.0, max_iter=1000)),
        ])
        self.pipeline.fit(X[self.features], y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        assert self.pipeline is not None, "call fit() first"
        return self.pipeline.predict_proba(X[self.features])[:, 1]

    def coefficients(self) -> pd.DataFrame:
        assert self.pipeline is not None
        coefs = self.pipeline.named_steps["clf"].coef_[0]
        return (
            pd.DataFrame({"feature": self.features, "coefficient": coefs})
            .sort_values("coefficient", key=np.abs, ascending=False)
            .reset_index(drop=True)
        )
