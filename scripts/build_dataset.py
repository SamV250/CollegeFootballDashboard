"""Build the processed historical dataset + feature matrix from the store.

Ensures data exists (triggers a demo refresh if the store is empty),
builds the leakage-safe feature matrix and writes it to
``data/processed/features.parquet`` for inspection / reuse.

    python scripts/build_dataset.py
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

from src.config import get_settings
from src.data.loader import load_games, load_teams
from src.features.build import build_feature_matrix, feature_columns


def main() -> None:
    settings = get_settings()
    games = load_games(settings)
    teams = load_teams(settings)
    feat = build_feature_matrix(games, teams, settings)

    out = settings.processed_dir / "features.parquet"
    feat.to_parquet(out, index=False)
    labeled = feat["home_win"].notna().sum()
    print(f"games            : {len(games)}")
    print(f"feature rows     : {len(feat)}")
    print(f"labeled rows     : {labeled}")
    print(f"feature columns  : {len(feature_columns())}")
    print(f"written          : {out}")


if __name__ == "__main__":
    main()
