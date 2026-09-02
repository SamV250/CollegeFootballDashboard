"""Streamlit-cached wrappers around :mod:`src.pipeline`.

Every expensive pipeline call is wrapped once here with ``st.cache_data``
/ ``st.cache_resource`` so the pages stay thin and fast.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src import pipeline
from src.config import get_settings

_TTL = 900  # seconds; pages are read-mostly, refresh cadence is coarser


@st.cache_resource(show_spinner=False)
def bundle():
    return pipeline.bundle()


@st.cache_data(ttl=_TTL, show_spinner="Loading games…")
def games_teams() -> tuple[pd.DataFrame, pd.DataFrame]:
    return pipeline.frames()


@st.cache_data(ttl=_TTL, show_spinner="Building features…")
def feature_matrix() -> pd.DataFrame:
    return pipeline.feature_matrix()


@st.cache_data(ttl=_TTL, show_spinner="Scoring upcoming games…")
def upcoming_predictions(season: int | None = None) -> pd.DataFrame:
    return pipeline.upcoming_predictions(season)


@st.cache_data(ttl=_TTL, show_spinner="Scoring completed games…")
def completed_predictions(season: int | None = None) -> pd.DataFrame:
    return pipeline.completed_predictions(season)


@st.cache_data(ttl=_TTL, show_spinner="Rating teams…")
def team_ratings(season: int | None = None) -> pd.DataFrame:
    return pipeline.team_ratings(season)


@st.cache_data(ttl=_TTL, show_spinner="Loading simulation…")
def dashboard_artifacts() -> dict:
    return pipeline.load_dashboard_artifacts()


@st.cache_data(ttl=_TTL, show_spinner=False)
def model_card() -> dict:
    return pipeline.get_model_card()


@st.cache_data(ttl=60, show_spinner=False)
def freshness() -> dict:
    return pipeline.dashboard_freshness()


def active_season() -> int:
    return get_settings().active_season()


def ensure_globals() -> tuple[str, str]:
    """Return (mode, tz_name); safe to call on any page."""

    st.session_state.setdefault("mode", "Executive")
    st.session_state.setdefault("tz", "US/Eastern")
    from src.ui.formatting import TZ_CHOICES

    return st.session_state["mode"], TZ_CHOICES[st.session_state["tz"]]


def clear_all() -> None:
    st.cache_data.clear()
    st.cache_resource.clear()
    pipeline.clear_caches()
