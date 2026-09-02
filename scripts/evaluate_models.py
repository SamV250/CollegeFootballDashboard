"""Print the saved evaluation, or re-run training+evaluation with --retrain.

    python scripts/evaluate_models.py
    python scripts/evaluate_models.py --retrain
"""

from __future__ import annotations

import argparse
import json

import _bootstrap  # noqa: F401

from src.models.registry import model_card, model_exists
from src.models.train import run_training


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--retrain", action="store_true")
    args = ap.parse_args()

    if args.retrain or not model_exists():
        run_training()

    ev = model_card().get("evaluation", {})
    for split in ("validation", "test"):
        if split not in ev:
            continue
        wm = ev[split]["win_model"]
        sm = ev[split]["score_model"]
        print(f"\n=== {split.upper()} (n={wm['ensemble']['n']}) ===")
        for name, key in [("Ensemble (primary)", "ensemble"), ("GBM only", "gbm"),
                          ("Elo", "elo"), ("Logistic", "logistic"),
                          ("Home rate", "home_rate")]:
            m = wm[key]
            print(f"  {name:12s}  logloss={m['log_loss']:.4f}  brier={m['brier']:.4f}  "
                  f"acc={m['accuracy']:.3f}  cal_err={m['calibration_error']:.4f}")
        print(f"  Score model   MAE(margin)={sm['mae_point_margin']:.2f}  "
              f"MAE(team)={sm['mae_team_score']:.2f}  RMSE={sm['rmse_point_margin']:.2f}")
        print(f"                within 3/7/10/14 = "
              f"{sm['within_3']:.2f}/{sm['within_7']:.2f}/"
              f"{sm['within_10']:.2f}/{sm['within_14']:.2f}")
    print("\nPer-season backtest:")
    print(json.dumps(ev.get("by_season", []), indent=2))


if __name__ == "__main__":
    main()
