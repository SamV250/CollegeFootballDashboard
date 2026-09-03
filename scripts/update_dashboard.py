"""One command to bring the whole dashboard up to date.

Idempotent: safe to run on a schedule (cron / GitHub Actions) or by hand.

    python scripts/update_dashboard.py
    python scripts/update_dashboard.py --source cfbd --iterations 10000
    python scripts/update_dashboard.py --skip-train        # data + sim only

Steps: fetch -> validate -> upsert (new/changed only) -> rebuild leakage-safe
features -> update Elo & ratings -> (re)train -> generate predictions ->
Monte Carlo season simulation -> save artifacts -> log.
"""

from __future__ import annotations

import argparse
import json
import logging
import time

import _bootstrap  # noqa: F401

from src.config import get_settings, utc_now
from src.data.loader import refresh
from src.data.store import DataStore
from src.models.registry import model_exists
from src.models.train import run_training
from src.pipeline import build_dashboard_artifacts, clear_caches

log = logging.getLogger("update_dashboard")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default=None)
    ap.add_argument("--iterations", type=int, default=None)
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--no-leverage", action="store_true")
    args = ap.parse_args()

    settings = get_settings()
    store = DataStore(settings)
    t0 = time.time()
    report: dict = {"started_utc": utc_now().isoformat(), "steps": {}}

    # 1-4: fetch / validate / diff / upsert
    try:
        r = refresh(source=args.source)
        report["steps"]["data_refresh"] = r
        log.info("data refresh: %s", r)
    except Exception as exc:  # noqa: BLE001
        report["steps"]["data_refresh"] = {"status": "failed", "error": str(exc)}
        log.warning("data refresh failed, continuing with existing data: %s", exc)

    clear_caches()

    # 5-6: features + Elo + ratings, 7: predictions  -> all inside training
    if not args.skip_train or not model_exists():
        try:
            tr = run_training(settings)
            report["steps"]["training"] = {
                "status": "ok",
                "validation": tr["evaluation"].get("validation", {}).get("win_model", {}).get("ensemble", {}),
                "test": tr["evaluation"].get("test", {}).get("win_model", {}).get("ensemble", {}),
            }
            log.info("training complete")
        except Exception as exc:  # noqa: BLE001
            report["steps"]["training"] = {"status": "failed", "error": str(exc)}
            log.error("training failed: %s", exc)
    else:
        report["steps"]["training"] = {"status": "skipped"}

    clear_caches()

    # 8-9: Monte Carlo season simulation + save artifacts
    try:
        summary = build_dashboard_artifacts(
            settings, n_iterations=args.iterations,
            with_leverage=not args.no_leverage,
        )
        report["steps"]["simulation"] = {
            "status": "ok", "season": summary["season"],
            "n_iterations": summary["n_iterations"],
            "generated_at_utc": summary["generated_at_utc"],
        }
        log.info("simulation complete: %s iterations", summary["n_iterations"])
    except Exception as exc:  # noqa: BLE001
        report["steps"]["simulation"] = {"status": "failed", "error": str(exc)}
        log.error("simulation failed: %s", exc)

    # 10: log
    report["elapsed_seconds"] = round(time.time() - t0, 1)
    report["finished_utc"] = utc_now().isoformat()
    failed = [k for k, v in report["steps"].items()
              if isinstance(v, dict) and v.get("status") == "failed"]
    report["ok"] = not failed
    store.log_refresh(source="update_dashboard",
                      status="ok" if not failed else "failed", detail=report)
    (settings.processed_dir / "last_update_report.json").write_text(
        json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))
    if failed:
        # training/simulation failure -> non-zero exit so CI does not deploy
        # a broken build. A data_refresh failure alone is tolerated (the
        # previous good dataset keeps serving).
        blocking = [k for k in failed if k != "data_refresh"]
        if blocking:
            raise SystemExit(f"update_dashboard: failed steps: {blocking}")


if __name__ == "__main__":
    main()
