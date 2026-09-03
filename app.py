"""College Football Prediction Dashboard — Executive Overview.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import get_settings
from src.ui import state
from src.ui.components import (
    big_number_card,
    disclaimer_banner,
    freshness_badge,
)
from src.ui.formatting import pct
from src.ui.theme import ACCENT, GOOD, MUTED, PLOTLY_LAYOUT, apply_theme

st.set_page_config(page_title="CFB Prediction Dashboard", page_icon="🏈",
                   layout="wide", initial_sidebar_state="expanded")
apply_theme()


# --------------------------------------------------------------------------
# sidebar — global controls
# --------------------------------------------------------------------------
def sidebar() -> None:
    st.sidebar.title("🏈 CFB Model")
    st.sidebar.caption("Predictions, playoff odds and plain-English explanations "
                       "for the 2026–27 FBS season.")

    st.session_state.setdefault("mode", "Executive")
    st.session_state.setdefault("tz", "US/Eastern")

    st.session_state["mode"] = st.sidebar.radio(
        "Explanation depth", ["Executive", "Analyst"],
        help="Executive: brief plain-language conclusions. "
             "Analyst: metrics, feature values, SHAP detail, diagnostics.",
        horizontal=True,
    )
    st.session_state["tz"] = st.sidebar.selectbox(
        "Time zone", ["US/Eastern", "US/Central", "US/Mountain", "US/Pacific", "UTC"],
    )

    card = state.model_card()
    st.sidebar.divider()
    st.sidebar.markdown("**Model**")
    st.sidebar.caption(
        f"Version `{card.get('model_version', '—')}`  \n"
        f"Trained {str(card.get('trained_at_utc', '—'))[:10]}  \n"
        f"Backend: {card.get('backend', '—')}  \n"
        f"Calibration: {card.get('calibration_method', '—')}"
    )
    if st.sidebar.button("↻ Reload data & caches"):
        state.clear_all()
        st.rerun()
    st.sidebar.caption("Automated refresh runs every 6 h in-season / 15 min on "
                       "game days (see Methodology).")


def tz_name() -> str:
    from src.ui.formatting import TZ_CHOICES
    return TZ_CHOICES[st.session_state.get("tz", "US/Eastern")]


# --------------------------------------------------------------------------
# page
# --------------------------------------------------------------------------
def main() -> None:
    sidebar()
    season = state.active_season()

    st.title(f"Executive Overview — {season}–{str(season + 1)[-2:]} season")
    st.caption("Start with the conclusions. Drill into the evidence on the other pages.")

    fresh = state.freshness()
    freshness_badge(fresh, tz_name())
    if fresh.get("state") in ("delayed", "stale"):
        st.warning(f"Data is **{fresh['state']}** — a scheduled refresh was missed. "
                   f"Figures below may not reflect the latest results.", icon="⚠️")

    art = state.dashboard_artifacts()
    tp: pd.DataFrame = art["team_probabilities"]
    ratings = state.team_ratings(season)
    gp: pd.DataFrame = art["game_predictions"]
    summary = art["summary"]

    demo = summary["freshness"].get("source") == "demo"
    if demo:
        st.info(get_settings().config["demo"]["generated_label"] +
                " — connect a `CFBD_API_KEY` for live data (see Methodology).", icon="🧪")

    # ---- headline answers -------------------------------------------------
    top_champ = tp.iloc[0]
    disclaimer_banner()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        big_number_card("Most likely national champion", top_champ["team"],
                        f"{pct(top_champ['p_national_champion'], 1)} · "
                        f"{top_champ['conference']}", ACCENT)
    with c2:
        top_playoff = tp.sort_values("p_playoff", ascending=False).iloc[0]
        big_number_card("Safest playoff bet", top_playoff["team"],
                        f"{pct(top_playoff['p_playoff'])} to make the field", GOOD)
    with c3:
        n_games = summary["meta"].get("n_remaining_games", len(gp))
        big_number_card("Games left to simulate", f"{n_games:,}",
                        f"{summary['n_iterations']:,} Monte Carlo seasons", MUTED)
    with c4:
        latest = ratings.iloc[0]
        big_number_card("Top model rating", latest["team"],
                        f"{latest['model_rating']:+.1f} pts vs. average", ACCENT)

    # ---- championship + playoff odds -----------------------------------
    left, right = st.columns(2)
    with left:
        st.subheader("National championship probability")
        d = tp.head(12)[["team", "p_national_champion"]].iloc[::-1]
        fig = go.Figure(go.Bar(y=d["team"], x=d["p_national_champion"] * 100,
                               orientation="h", marker_color=ACCENT,
                               text=[f"{v*100:.1f}%" for v in d["p_national_champion"]],
                               textposition="outside"))
        fig.update_layout(height=420, xaxis_title="%", **PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True, key="natl_title_bar")
    with right:
        st.subheader("Playoff probability (top 16)")
        d = tp.sort_values("p_playoff").tail(16)[["team", "p_playoff"]]
        fig = go.Figure(go.Bar(y=d["team"], x=d["p_playoff"] * 100, orientation="h",
                               marker_color=GOOD,
                               text=[f"{v*100:.0f}%" for v in d["p_playoff"]],
                               textposition="outside"))
        fig.update_layout(height=420, xaxis_title="%", **PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True, key="playoff_prob_bar")
    st.caption("Playoff and championship figures are **estimates** from a "
               "transparent, configurable selection model — not a forecast of "
               "the selection committee's choices.")

    # ---- top 25 by model rating ---------------------------------------
    st.subheader("Model top 25")
    t25 = ratings.head(25).merge(
        tp[["team", "p_conf_champion", "p_playoff", "p_national_champion"]],
        on="team", how="left")
    show = t25[["model_rank", "team", "conference", "record", "model_rating",
                "p_conf_champion", "p_playoff", "p_national_champion"]].copy()
    show.columns = ["#", "Team", "Conference", "Record", "Model rating",
                    "Conf title", "Playoff", "Natl title"]
    for c in ["Conf title", "Playoff", "Natl title"]:
        show[c] = (show[c] * 100).round(1).astype(str) + "%"
    st.dataframe(show, use_container_width=True, hide_index=True, height=560)

    # ---- movers, disagreements, big games ----------------------------
    st.subheader("Signals this week")
    m1, m2 = st.columns(2)
    with m1:
        st.markdown("**Where the model disagrees with a rating-only view**")
        ratings2 = ratings.copy()
        ratings2["elo_rank"] = ratings2["elo"].rank(ascending=False)
        ratings2["disagreement"] = ratings2["elo_rank"] - ratings2["model_rank"]
        disagree = pd.concat([
            ratings2.nlargest(4, "disagreement"),
            ratings2.nsmallest(4, "disagreement"),
        ])[["team", "model_rank", "elo_rank", "disagreement"]]
        disagree.columns = ["Team", "Model rank", "Elo rank", "Model − Elo (rank)"]
        st.dataframe(disagree, use_container_width=True, hide_index=True)
        st.caption("Positive = the full model likes the team more than Elo alone.")
    with m2:
        st.markdown("**Highest-leverage upcoming games**")
        lev = art.get("leverage", pd.DataFrame())
        if lev is not None and not lev.empty:
            L = lev.head(8)[["matchup", "week", "leverage", "most_affected_team"]].copy()
            L.columns = ["Matchup", "Wk", "Leverage", "Most affected"]
            st.dataframe(L, use_container_width=True, hide_index=True)
            st.caption("Leverage = total playoff-probability swing across all teams "
                       "between the two possible results.")
        else:
            st.caption("Leverage table not built yet — run "
                       "`python scripts/update_dashboard.py`.")

    st.divider()
    st.caption(
        f"Predictions generated {summary['generated_at_utc'][:19]}Z · "
        f"model version {summary['model_card'].get('model_version','—')} · "
        f"latest completed game "
        f"{str(summary['freshness'].get('latest_completed_game_utc',''))[:10]}. "
        "See **Model Evaluation** for accuracy and calibration, and "
        "**Methodology & Data** for sources and definitions."
    )


if __name__ == "__main__":
    main()
