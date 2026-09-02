"""Run the Monte Carlo season simulation and refresh dashboard artifacts.

    python scripts/run_simulation.py                 # config n_iterations
    python scripts/run_simulation.py --iterations 5000 --no-leverage
"""

from __future__ import annotations

import argparse
import json

import _bootstrap  # noqa: F401

from src.pipeline import build_dashboard_artifacts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--iterations", type=int, default=None)
    ap.add_argument("--no-leverage", action="store_true")
    args = ap.parse_args()

    summary = build_dashboard_artifacts(
        n_iterations=args.iterations, with_leverage=not args.no_leverage
    )
    print(json.dumps({k: summary[k] for k in
                      ("generated_at_utc", "season", "n_iterations", "meta")},
                     indent=2, default=str)[:1500])


if __name__ == "__main__":
    main()
