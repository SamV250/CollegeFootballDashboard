"""Shared fixtures. A small synthetic universe keeps the suite fast."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import get_settings  # noqa: E402
from src.data.demo import generate_dataset  # noqa: E402


@pytest.fixture(scope="session")
def settings():
    return get_settings()


@pytest.fixture(scope="session")
def demo_data(settings):
    """Two prior seasons + a partial current season."""

    seasons = [2024, 2025, 2026]
    return generate_dataset(settings, seasons=seasons)


@pytest.fixture(scope="session")
def games(demo_data) -> pd.DataFrame:
    return demo_data["games"].copy()


@pytest.fixture(scope="session")
def teams(demo_data) -> pd.DataFrame:
    return demo_data["teams"].copy()
