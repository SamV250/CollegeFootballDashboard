"""Persist trained models together with a full provenance record.

Saved bundle (``models/game_model.joblib`` + ``models/model_card.json``):

* the fitted :class:`~src.models.game_model.GamePredictor`
* the fitted :class:`~src.models.elo.EloModel`
* metadata: model version, training date (UTC), training seasons,
  feature list, evaluation results, data-source + config revision.

Loading is lazy and cached so Streamlit pages share one instance.
"""

from __future__ import annotations

import functools
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib

from src.config import Settings, get_settings, utc_now

BUNDLE_NAME = "game_model.joblib"
CARD_NAME = "model_card.json"


def save_bundle(
    predictor: Any,
    elo: Any,
    evaluation: dict[str, Any],
    settings: Settings | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    settings = settings or get_settings()
    mdir = settings.models_dir
    cfg = settings.config

    card = {
        "model_version": f"{cfg['version']['model_schema']}."
                         f"{cfg['version']['config_revision']}",
        "trained_at_utc": utc_now().isoformat(),
        "backend": cfg["model"]["backend"],
        "training_seasons": sorted(
            s for s in cfg["season"]["backtest_seasons"]
            if s <= cfg["model"]["train_through_season"]
        ),
        "validation_season": cfg["model"]["validation_season"],
        "test_season": cfg["model"]["test_season"],
        "features": list(predictor.features),
        "calibration_method": predictor.metadata.get("calibration_method"),
        "ensemble_weights": predictor.metadata.get("ensemble_weights"),
        "n_train": predictor.metadata.get("n_train"),
        "n_calibration": predictor.metadata.get("n_calibration"),
        "config_revision": cfg["version"]["config_revision"],
        "evaluation": evaluation,
    }
    if extra:
        card.update(extra)

    joblib.dump(
        {"predictor": predictor, "elo": elo, "card": card}, mdir / BUNDLE_NAME
    )
    (mdir / CARD_NAME).write_text(json.dumps(card, indent=2, default=str))
    _load_bundle_cached.cache_clear()
    return mdir / BUNDLE_NAME


@functools.lru_cache(maxsize=1)
def _load_bundle_cached(path_str: str) -> dict[str, Any]:
    return joblib.load(path_str)


def load_bundle(settings: Settings | None = None) -> dict[str, Any]:
    settings = settings or get_settings()
    path = settings.models_dir / BUNDLE_NAME
    if not path.exists():
        raise FileNotFoundError(
            f"No trained model at {path}. Run `python scripts/train_models.py`."
        )
    return _load_bundle_cached(str(path))


def model_card(settings: Settings | None = None) -> dict[str, Any]:
    settings = settings or get_settings()
    path = settings.models_dir / CARD_NAME
    if path.exists():
        return json.loads(path.read_text())
    return load_bundle(settings)["card"]


def model_exists(settings: Settings | None = None) -> bool:
    settings = settings or get_settings()
    return (settings.models_dir / BUNDLE_NAME).exists()
