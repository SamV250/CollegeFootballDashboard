"""Train Elo + baselines + the primary game model and save the bundle.

    python scripts/train_models.py
"""

from __future__ import annotations

import json

import _bootstrap  # noqa: F401

from src.models.train import run_training


def main() -> None:
    result = run_training()
    ev = result["evaluation"]
    print(f"\nBundle: {result['bundle']}\n")
    for split in ("validation", "test"):
        if split not in ev:
            continue
        wm = ev[split]["win_model"]
        sm = ev[split]["score_model"]
        print(f"[{split}]  n={wm['ensemble']['n']}")
        for key, label in (("ensemble", "Ensemble"), ("gbm", "GBM only")):
            d = wm[key]
            print(f"  {label:9s} logloss={d['log_loss']:.4f}  "
                  f"brier={d['brier']:.4f}  acc={d['accuracy']:.3f}  "
                  f"cal_err={d['calibration_error']:.4f}")
        print(f"  Elo      logloss={wm['elo']['log_loss']:.4f}  "
              f"brier={wm['elo']['brier']:.4f}  acc={wm['elo']['accuracy']:.3f}")
        print(f"  Logistic logloss={wm['logistic']['log_loss']:.4f}  "
              f"brier={wm['logistic']['brier']:.4f}  acc={wm['logistic']['accuracy']:.3f}")
        print(f"  HomeRate logloss={wm['home_rate']['log_loss']:.4f}")
        print(f"  Score    MAE(margin)={sm['mae_point_margin']:.2f}  "
              f"MAE(team)={sm['mae_team_score']:.2f}  "
              f"within7={sm['within_7']:.2f}  within14={sm['within_14']:.2f}")
        print()
    print("Per-season backtest:")
    print(json.dumps(ev.get("by_season", []), indent=2))


if __name__ == "__main__":
    main()
