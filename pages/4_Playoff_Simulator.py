"""Playoff Simulator — force results, see the bracket move."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src import pipeline
from src.ui import state
from src.ui.theme import apply_theme

st.set_page_config(page_title="Playoff Simulator", page_icon="🏆", layout="wide")
apply_theme()
mode, tz = state.ensure_globals()
season = state.active_season()

st.title("Playoff Simulator")
st.caption("Pick winners of upcoming games, or override a game's probability, then "
           "re-run the Monte Carlo season. Compare against the untouched baseline.")

art = state.dashboard_artifacts()
base_tp = art["team_probabilities"]
upc = state.upcoming_predictions(season)
if upc.empty:
    st.info("No upcoming games to simulate.")
    st.stop()

upc = upc.sort_values(["week", "date"]).copy()
upc["label"] = [f"W{int(w)}: {a} at {h}  (model: {h} {p*100:.0f}%)"
                for w, h, a, p in zip(upc["week"], upc["home_team"],
                                      upc["away_team"], upc["home_win_prob"])]

st.subheader("1. Set results")
ratings = state.team_ratings(season)
notable = set(ratings.head(40)["team"])
default_games = upc[upc["home_team"].isin(notable) | upc["away_team"].isin(notable)]
pick_from = st.multiselect(
    "Choose games to fix (search by team or week)",
    options=upc["game_id"].tolist(),
    format_func=lambda g: upc.set_index("game_id").loc[g, "label"],
    max_selections=25,
)

forced: dict[str, str] = {}
if pick_from:
    for gid in pick_from:
        r = upc.set_index("game_id").loc[gid]
        choice = st.radio(r["label"], [r["home_team"], r["away_team"], "leave to model"],
                          horizontal=True, key=f"pick_{gid}")
        if choice == r["home_team"]:
            forced[gid] = "home"
        elif choice == r["away_team"]:
            forced[gid] = "away"

n_iter = st.select_slider("Simulation iterations", [2000, 4000, 8000, 12000], value=4000)
go = st.button("▶ Run scenario", type="primary")

if go:
    with st.spinner("Simulating…"):
        scen = pipeline.run_scenario_simulation(forced=forced, n_iterations=n_iter)
    st.session_state["scenario_result"] = scen

scen = st.session_state.get("scenario_result")
if scen is None:
    st.info("Set some results and press **Run scenario**.")
    st.stop()

s_tp = scen["team_probabilities"].set_index("team")
b_tp = base_tp.set_index("team")

st.divider()
st.subheader("2. Projected playoff field (scenario)")
brk = scen["bracket"].copy()
brk["Playoff %"] = (brk["p_playoff"] * 100).round(0).astype(int).astype(str) + "%"
brk["Title %"] = (brk["p_national_champion"] * 100).round(1).astype(str) + "%"
brk["Round 1"] = brk["bye"].map({True: "BYE", False: "hosts first round"})
brk["Bid"] = brk["auto_bid_conf_champ"].map({True: "Conf champ (auto)", False: "At large"})
st.dataframe(brk[["seed", "team", "conference", "Bid", "Round 1", "Playoff %", "Title %"]]
             .rename(columns={"seed": "Seed", "team": "Team", "conference": "Conf"}),
             use_container_width=True, hide_index=True, height=460)

st.subheader("3. Biggest swings vs. baseline")
delta = pd.DataFrame({
    "team": s_tp.index,
    "playoff_base": b_tp.reindex(s_tp.index)["p_playoff"],
    "playoff_scenario": s_tp["p_playoff"],
    "title_base": b_tp.reindex(s_tp.index)["p_national_champion"],
    "title_scenario": s_tp["p_national_champion"],
})
delta["playoff_change"] = delta["playoff_scenario"] - delta["playoff_base"]
delta["title_change"] = delta["title_scenario"] - delta["title_base"]
movers = pd.concat([delta.nlargest(10, "playoff_change"),
                    delta.nsmallest(10, "playoff_change")]).drop_duplicates("team")
disp = movers.copy()
for c in ["playoff_base", "playoff_scenario", "title_base", "title_scenario"]:
    disp[c] = (disp[c] * 100).round(1).astype(str) + "%"
disp["playoff_change"] = (movers["playoff_change"] * 100).round(1).map(lambda v: f"{v:+.1f} pp")
disp["title_change"] = (movers["title_change"] * 100).round(1).map(lambda v: f"{v:+.1f} pp")
st.dataframe(disp[["team", "playoff_base", "playoff_scenario", "playoff_change",
                   "title_change"]].rename(columns={
    "team": "Team", "playoff_base": "Playoff (base)",
    "playoff_scenario": "Playoff (scenario)", "playoff_change": "Δ Playoff",
    "title_change": "Δ Title"}), use_container_width=True, hide_index=True)

st.divider()
st.subheader("Games with the most leverage (baseline)")
lev = art.get("leverage", pd.DataFrame())
if lev is not None and not lev.empty:
    L = lev[["matchup", "week", "leverage", "most_affected_team", "swing_for_team"]].copy()
    L.columns = ["Matchup", "Wk", "Leverage (total swing)", "Most affected", "Their swing"]
    st.dataframe(L, use_container_width=True, hide_index=True)
    st.caption("Leverage = sum over all teams of the absolute change in playoff "
               "probability between the two possible results of that game.")
else:
    st.caption("Run `python scripts/update_dashboard.py` to build the leverage table.")

st.caption("All playoff numbers are estimates from the configurable selection model "
           "in `config/playoff.yaml` — not a prediction of committee behaviour.")
