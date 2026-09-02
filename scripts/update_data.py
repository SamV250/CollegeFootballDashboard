"""Fetch the latest data and merge it into the local store (idempotent).

    python scripts/update_data.py                 # use config source priority
    python scripts/update_data.py --source demo   # force a source
    python scripts/update_data.py --seasons 2024 2025 2026
"""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from src.data.loader import refresh


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default=None, help="demo | local | cfbd | espn")
    ap.add_argument("--seasons", type=int, nargs="*", default=None)
    args = ap.parse_args()

    result = refresh(source=args.source, seasons=args.seasons)
    print("Refresh complete:")
    for k, v in result.items():
        print(f"  {k:20s}: {v}")


if __name__ == "__main__":
    main()
