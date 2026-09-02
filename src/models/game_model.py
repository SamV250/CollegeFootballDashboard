"""Primary game model: calibrated win probability + score prediction.

Three learners share one feature matrix:

1. **Win-probability classifier** -- gradient-boosted trees
   (LightGBM by default, XGBoost selectable via ``config.model.backend``)
   producing ``P(home team wins)``.  The raw classifier is then wrapped
   in isotonic (or Platt) **calibration** fitted on a held-out, strictly
   later slice of games, because well-calibrated probabilities matter
   more here than raw accuracy.
2. **Margin regressor** -- predicts ``home_points - away_points``.
3. **Total regressor** -- predicts ``home_points + away_points``.

Team scores are recovered as ``(total +/- margin) / 2``.  Modelling
margin + total rather than the two scores directly keeps the home/away
scores coherent and is more stable when pace varies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression

from src.config import Settings, get_settings


def _make_classifier(cfg: dict[str, Any], seed: int):
    backend = cfg["backend"]
    if backend == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=cfg["n_estimators"],
            learning_rate=cfg["learning_rate"],
            max_depth=cfg["max_depth"],
            subsample=cfg["subsample"],
            colsample_bytree=cfg["colsample_bytree"],
            min_child_weight=cfg["min_child_samples"],
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=seed,
            n_jobs=-1,
        )
    from lightgbm import LGBMClassifier

    return LGBMClassifier(
        n_estimators=cfg["n_estimators"],
        learning_rate=cfg["learning_rate"],
        max_depth=cfg["max_depth"],
        num_leaves=2 ** cfg["max_depth"],
        subsample=cfg["subsample"],
        colsample_bytree=cfg["colsample_bytree"],
        min_child_samples=cfg["min_child_samples"],
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )


def _make_regressor(cfg: dict[str, Any], seed: int):
    if cfg["backend"] == "xgboost":
        from xgboost import XGBRegressor

        return XGBRegressor(
            n_estimators=cfg["n_estimators"],
            learning_rate=cfg["learning_rate"],
            max_depth=cfg["max_depth"],
            subsample=cfg["subsample"],
            colsample_bytree=cfg["colsample_bytree"],
            random_state=seed,
            n_jobs=-1,
        )
    from lightgbm import LGBMRegressor

    return LGBMRegressor(
        n_estimators=cfg["n_estimators"],
        learning_rate=cfg["learning_rate"],
        max_depth=cfg["max_depth"],
        num_leaves=2 ** cfg["max_depth"],
        subsample=cfg["subsample"],
        colsample_bytree=cfg["colsample_bytree"],
        min_child_samples=cfg["min_child_samples"],
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )


@dataclass
class GamePredictor:
    features: list[str]
    settings: Settings = field(default_factory=get_settings)
    raw_classifier: Any = None
    calibrator: Any = None            # IsotonicRegression or None
    calibrated_classifier: Any = None  # CalibratedClassifierCV or None
    logit_model: Any = None            # LogisticBaseline, ensemble member
    margin_model: Any = None
    total_model: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)

    # -- training --------------------------------------------------------
    def fit(
        self,
        train: pd.DataFrame,
        calib: pd.DataFrame | None = None,
    ) -> "GamePredictor":
        """Fit the classifier, its calibration map, and the two regressors.

        Calibration strategy (``config.model.calibration``):

        * ``isotonic`` / ``sigmoid`` -- if a dedicated ``calib`` split is
          given, fit that mapping on the held-out (strictly later) games;
        * otherwise fall back to k-fold cross-validated calibration on the
          training set (``CalibratedClassifierCV``), which uses all the
          data and is more stable when the holdout is small.
        """

        cfg = self.settings.config["model"]
        seed = int(cfg["random_seed"])
        Xtr = train[self.features]
        ytr = train["home_win"].astype(int)

        self.raw_classifier = _make_classifier(cfg, seed)
        self.raw_classifier.fit(Xtr, ytr)

        method = cfg.get("calibration", "isotonic")
        n_cv = int(cfg.get("calibration_cv", 4))
        if method == "none":
            pass
        elif calib is not None and len(calib) > 150:
            raw_p = self.raw_classifier.predict_proba(calib[self.features])[:, 1]
            if method == "sigmoid":
                self.calibrated_classifier = CalibratedClassifierCV(
                    self.raw_classifier, method="sigmoid", cv="prefit"
                ).fit(calib[self.features], calib["home_win"].astype(int))
            else:
                self.calibrator = IsotonicRegression(out_of_bounds="clip")
                self.calibrator.fit(raw_p, calib["home_win"].astype(int))
        else:  # cross-validated calibration on the training set
            cv_clf = _make_classifier(cfg, seed)
            self.calibrated_classifier = CalibratedClassifierCV(
                cv_clf, method=("sigmoid" if method == "sigmoid" else "isotonic"),
                cv=n_cv,
            ).fit(Xtr, ytr)

        # Ensemble member: a plain logistic regression on a stable feature
        # subset. On real college-football data the boosted model, Elo and
        # logistic regression are all within noise of each other, and their
        # simple average is a hair better and better calibrated than any
        # one alone -- so the reported win probability is a blend.
        from src.models.baselines import LogisticBaseline

        self.logit_model = LogisticBaseline().fit(train, train["home_win"])

        self.margin_model = _make_regressor(cfg, seed)
        self.margin_model.fit(Xtr, train["point_margin"])
        self.total_model = _make_regressor(cfg, seed)
        self.total_model.fit(Xtr, train["total_points"])

        self.metadata.setdefault("n_train", len(train))
        self.metadata.setdefault("n_calibration", 0 if calib is None else len(calib))
        self.metadata["calibration_method"] = method
        self.metadata["ensemble_weights"] = self._weights()
        return self

    # -- ensemble -------------------------------------------------------
    def _weights(self) -> dict[str, float]:
        w = dict(self.settings.config["model"].get(
            "ensemble", {"gbm": 0.34, "logistic": 0.33, "elo": 0.33}))
        s = sum(w.values()) or 1.0
        return {k: v / s for k, v in w.items()}

    def _gbm_prob(self, X: pd.DataFrame) -> np.ndarray:
        raw = self._raw_prob(X)
        if self.calibrator is not None:
            return np.clip(self.calibrator.predict(raw), 1e-4, 1 - 1e-4)
        if self.calibrated_classifier is not None:
            return self.calibrated_classifier.predict_proba(X[self.features])[:, 1]
        return raw

    @staticmethod
    def _elo_prob(X: pd.DataFrame) -> np.ndarray:
        # elo_diff already includes the home-field bump (see features/build)
        d = X["elo_diff"].to_numpy(dtype=float) if "elo_diff" in X else np.zeros(len(X))
        return 1.0 / (1.0 + 10.0 ** (-d / 400.0))

    # -- inference ------------------------------------------------------
    def _raw_prob(self, X: pd.DataFrame) -> np.ndarray:
        return self.raw_classifier.predict_proba(X[self.features])[:, 1]

    def win_probability(self, X: pd.DataFrame, blend: bool = True) -> np.ndarray:
        """Calibrated home win probability.

        ``blend=True`` (default) returns the ensemble average of the
        calibrated boosted model, the logistic model and the Elo
        probability. ``blend=False`` returns the boosted model alone
        (used for the model-comparison table on the Evaluation page).
        """

        gbm = self._gbm_prob(X)
        if not blend or self.logit_model is None:
            return gbm
        w = self._weights()
        logit = self.logit_model.predict_proba(X)
        elo = self._elo_prob(X)
        p = w["gbm"] * gbm + w["logistic"] * logit + w["elo"] * elo
        return np.clip(p, 1e-4, 1 - 1e-4)

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        p = self.win_probability(X)
        margin = self.margin_model.predict(X[self.features])
        total = self.total_model.predict(X[self.features])
        home_pts = (total + margin) / 2.0
        away_pts = (total - margin) / 2.0
        return pd.DataFrame({
            "home_win_prob": p,
            "pred_margin": margin,
            "pred_total": total,
            "pred_home_points": np.clip(home_pts, 0, None),
            "pred_away_points": np.clip(away_pts, 0, None),
            "pred_winner_is_home": p >= 0.5,
        }, index=X.index)

    # -- explainability support --------------------------------------
    def shap_values(self, X: pd.DataFrame):
        """Return (expected_value, shap_matrix) for the raw classifier."""

        import shap

        explainer = shap.TreeExplainer(self.raw_classifier)
        sv = explainer.shap_values(X[self.features])
        if isinstance(sv, list):  # some backends return [class0, class1]
            sv = sv[1]
        base = explainer.expected_value
        if isinstance(base, (list, np.ndarray)):
            base = float(np.ravel(base)[-1])
        return base, sv
