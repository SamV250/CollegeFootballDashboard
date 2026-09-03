"""Matchup Simulator — any two FBS teams, with what-if adjustments."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src import pipeline
from src.ui import state
from src.ui.components import factor_list, win_prob_bar
from src.ui.formatting import pct, signed
from src.ui.matchup import render_explanation
from src.ui.theme import apply_theme

st.set_page_config(page_title="Matchup Simulator", page_icon="⚔️", layout="wide")
apply_theme()
mode, tz = state.ensure_globals()

st.title("Matchup Simulator")
st.caption("Pick any two FBS teams and stress-test the matchup. Model output and "
           "your manual assumptions are reported separately.")

ratings = state.team_ratings(state.active_season())
team_list = ratings["team"].tolist()

c1, c2, c3 = st.columns([2, 2, 1])
home = c1.selectbox("Team A (home unless neutral)", team_list, index=0)
away = c2.selectbox("Team B (away)", team_list, index=1)
site = c3.radio("Site", ["Team A home", "Neutral"], index=0)
neutral = site == "Neutral"

with st.expander("Scenario adjustments (what-if)", expanded=False):
    st.caption("These are **your** assumptions layered on top of the model — "
               "e.g. a key injury, a tempo change, a turnover script. Leave at "
               "zero for the pure model projection.")
    a1, a2 = st.columns(2)
    with a1:
        st.markdown(f"**{home}**")
        h_rating = st.slider(f"{home} rating adjustment (pts)", -21.0, 21.0, 0.0, 0.5,
                             help="Points added to/removed from the team rating "
                                  "(injuries, personnel, motivation).")
        h_off = st.slider(f"{home} offensive efficiency nudge", -0.30, 0.30, 0.0, 0.02)
        h_def = st.slider(f"{home} defensive efficiency nudge", -0.30, 0.30, 0.0, 0.02)
    with a2:
        st.markdown(f"**{away}**")
        a_rating = st.slider(f"{away} rating adjustment (pts)", -21.0, 21.0, 0.0, 0.5)
        a_off = st.slider(f"{away} offensive efficiency nudge", -0.30, 0.30, 0.0, 0.02)
        a_def = st.slider(f"{away} defensive efficiency nudge", -0.30, 0.30, 0.0, 0.02)
    t1, t2 = st.columns(2)
    to_shift = t1.slider("Turnover-margin scenario (Team A)", -3.0, 3.0, 0.0, 0.5,
                         help="Positive favours Team A. Turnovers are volatile; "
                              "the model already regresses them heavily.")
    pace_shift = t2.slider("Pace shift (sec/play, both teams)", -4.0, 4.0, 0.0, 0.5)

adjustments = {
    "home_rating_delta": h_rating, "away_rating_delta": a_rating,
    "home_off_delta": h_off, "away_off_delta": a_off,
    "home_def_delta": h_def, "away_def_delta": a_def,
    "turnover_margin_shift": to_shift, "pace_shift": pace_shift,
}
any_adj = any(abs(v) > 1e-9 for v in adjustments.values())

if home == away:
    st.warning("Pick two different teams.")
    st.stop()

base = pipeline.simulate_matchup(home, away, neutral, None)
scen = pipeline.simulate_matchup(home, away, neutral, adjustments) if any_adj else base

exp = scen["explanation"]
pred = scen["prediction"]

st.divider()
st.subheader(f"{away} {'vs' if neutral else 'at'} {home}")
win_prob_bar(home, away, pred["home_win_prob"])

m1, m2, m3, m4 = st.columns(4)
m1.metric("Predicted winner", exp["favorite"], f"{pct(exp['favorite_win_prob'])}")
m2.metric("Projected score",
          f"{pred['pred_home_points']:.0f}–{pred['pred_away_points']:.0f}",
          f"{home} – {away}")
m3.metric("Expected margin", signed(pred["pred_margin"]),
          help="Positive = home/Team A by that many points.")
m4.metric("Confidence", exp["confidence"])

lo, hi = exp["plausible_margin_range"]
st.caption(f"Plausible final margin (~80% band): {exp['favorite']} by "
           f"{max(lo, 0):.0f}–{hi:.0f}.")

if any_adj:
    b = base["prediction"]
    st.info(
        f"**Model baseline:** {home} {b['home_win_prob']*100:.0f}% win prob, "
        f"projected {b['pred_home_points']:.0f}–{b['pred_away_points']:.0f}.  \n"
        f"**With your adjustments:** {home} {pred['home_win_prob']*100:.0f}% "
        f"({(pred['home_win_prob']-b['home_win_prob'])*100:+.0f} pts), projected "
        f"{pred['pred_home_points']:.0f}–{pred['pred_away_points']:.0f}.",
        icon="🧪")

lcol, rcol = st.columns(2)
with lcol:
    st.markdown(f"#### Key advantages — {exp['favorite']}")
    factor_list(exp["favorite_factors"])
with rcol:
    st.markdown(f"#### Upset path — {exp['underdog']}")
    factor_list(exp["underdog_factors"])

st.divider()
st.subheader("Side-by-side")
def team_col(t: str) -> pd.Series:
    r = ratings[ratings["team"] == t]
    return r.iloc[0] if not r.empty else pd.Series(dtype=float)
rh, ra = team_col(home), team_col(away)


def _col(r: pd.Series) -> list[str]:
    return [
        f"{r.get('model_rating', float('nan')):+.1f}",
        f"#{int(r.get('model_rank', 0))}",
        f"{r.get('elo', float('nan')):.0f}",
        str(r.get("record", "—")),
        f"{r.get('points_for_pg', float('nan')):.1f}",
        f"{r.get('points_against_pg', float('nan')):.1f}",
        f"{r.get('offense_pctl', float('nan')):.0f}th",
        f"{r.get('defense_pctl', float('nan')):.0f}th",
        f"{r.get('special_teams_pctl', float('nan')):.0f}th",
    ]


comp = pd.DataFrame({
    "Metric": ["Model rating", "Model rank", "Elo", "Record",
               "Points/game for", "Points/game against",
               "Offense pctl", "Defense pctl", "Special teams pctl"],
    home: _col(rh),
    away: _col(ra),
})
st.dataframe(comp, use_container_width=True, hide_index=True)

if mode == "Analyst":
    render_explanation(exp, "Analyst", key_prefix="sim")
