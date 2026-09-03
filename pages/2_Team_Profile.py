"""Team Profile — one team's rating, schedule, trajectory and outlook."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.pipeline import tier_label
from src.ui import state
from src.ui.components import big_number_card, demo_data_banner, percentile_bar
from src.ui.formatting import fmt_dt, pct
from src.ui.theme import ACCENT, GOOD, MUTED, PLOTLY_LAYOUT, apply_theme

st.set_page_config(page_title="Team Profile", page_icon="📋", layout="wide")
apply_theme()
mode, tz = state.ensure_globals()
season = state.active_season()

ratings = state.team_ratings(season)
art = state.dashboard_artifacts()
tp = art["team_probabilities"].set_index("team")
games, teams = state.games_teams()
feat = state.feature_matrix()

st.title("Team Profile")
demo_data_banner()
team = st.selectbox("Team", ratings["team"].tolist())
row = ratings[ratings["team"] == team].iloc[0]
probs = tp.loc[team] if team in tp.index else None

# ---- headline cards --------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
with c1:
    big_number_card("Record", row["record"], f"{row['conference']}", ACCENT)
with c2:
    big_number_card("Model rating", f"{row['model_rating']:+.1f}",
                    f"#{int(row['model_rank'])} of {len(ratings)} · "
                    f"{tier_label(row['overall_pctl'])}", ACCENT)
with c3:
    big_number_card("Playoff probability",
                    pct(probs["p_playoff"]) if probs is not None else "—",
                    "estimate", GOOD)
with c4:
    big_number_card("National title probability",
                    pct(probs["p_national_champion"], 1) if probs is not None else "—",
                    "estimate", ACCENT)

if probs is not None:
    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Conf. title game", pct(probs["p_conf_title_game"]))
    c6.metric("Conference champion", pct(probs["p_conf_champion"]))
    c7.metric("First-round bye", pct(probs["p_first_round_bye"]))
    c8.metric("Reach title game", pct(probs["p_title_game"]))

# ---- unit ratings vs FBS ------------------------------------------
st.subheader("How this team compares to FBS")
lc, rc = st.columns(2)
with lc:
    percentile_bar("Overall", row["overall_pctl"], tier_label(row["overall_pctl"]))
    percentile_bar("Offense (EPA/play)", row["offense_pctl"], tier_label(row["offense_pctl"]))
    percentile_bar("Defense (EPA/play prevented)", row["defense_pctl"],
                   tier_label(row["defense_pctl"]))
    percentile_bar("Special teams", row["special_teams_pctl"],
                   tier_label(row["special_teams_pctl"]))
with rc:
    st.markdown("<div class='cfb-card'><h4>Snapshot</h4>", unsafe_allow_html=True)
    st.markdown(
        f"- Elo rating: **{row['elo']:.0f}**\n"
        f"- Points/game: **{row['points_for_pg']:.1f}** for, "
        f"**{row['points_against_pg']:.1f}** against\n"
        f"- Games played: **{int(row['games_played'])}**\n"
        f"- Projected final wins: "
        f"**{probs['proj_wins']:.1f}**" if probs is not None else "",
    )
    st.markdown("</div>", unsafe_allow_html=True)

# ---- remaining schedule + game-by-game win prob -------------------
st.subheader("Schedule & game-by-game win probability")
tg = feat[(feat["season"] == season) &
          ((feat["home_team"] == team) | (feat["away_team"] == team))].copy()
predictor = state.bundle()["predictor"]
if not tg.empty:
    pred = predictor.predict(tg)
    tg = tg.join(pred[["home_win_prob", "pred_home_points", "pred_away_points"]])
    tg["is_home"] = tg["home_team"] == team
    tg["opponent"] = np.where(tg["is_home"], tg["away_team"], tg["home_team"])
    tg["team_win_prob"] = np.where(tg["is_home"], tg["home_win_prob"],
                                   1 - tg["home_win_prob"])
    tg["result"] = np.where(
        ~tg["completed"], "—",
        np.where((tg["home_points"] > tg["away_points"]) == tg["is_home"], "W", "L"))
    tg["site"] = np.where(tg["neutral_site"], "N", np.where(tg["is_home"], "H", "A"))

    fig = go.Figure()
    fig.add_bar(x=tg["week"], y=tg["team_win_prob"] * 100,
                marker_color=np.where(tg["completed"], MUTED, ACCENT),
                text=tg["opponent"], textposition="outside")
    fig.add_hline(y=50, line_dash="dot", line_color=MUTED)
    fig.update_layout(title=f"{team} win probability by week (grey = played)",
                      yaxis_title="Win probability %", xaxis_title="Week",
                      height=380, **PLOTLY_LAYOUT)
    fig.update_yaxes(range=[0, 100])
    st.plotly_chart(fig, use_container_width=True, key="tp_winprob_by_week")

    disp = tg[["week", "site", "opponent", "result", "team_win_prob",
               "pred_home_points", "pred_away_points", "date"]].copy()
    disp["Proj score (this team)"] = np.where(
        tg["is_home"], tg["pred_home_points"], tg["pred_away_points"]).round(0)
    disp["Proj score (opp)"] = np.where(
        tg["is_home"], tg["pred_away_points"], tg["pred_home_points"]).round(0)
    disp["Win prob"] = (disp["team_win_prob"] * 100).round(0).astype(int).astype(str) + "%"
    disp["Kickoff"] = disp["date"].apply(lambda d: fmt_dt(d, tz))
    st.dataframe(
        disp[["week", "site", "opponent", "result", "Win prob",
              "Proj score (this team)", "Proj score (opp)", "Kickoff"]].rename(
            columns={"week": "Wk", "site": "Site", "opponent": "Opponent",
                     "result": "Result"}),
        use_container_width=True, hide_index=True)

# ---- strengths / vulnerabilities / key games ---------------------
st.subheader("Strengths, vulnerabilities & swing games")
played = tg[tg["completed"]] if not tg.empty else pd.DataFrame()

sc1, sc2, sc3 = st.columns(3)
with sc1:
    st.markdown("**Strengths**")
    strengths = []
    if row["offense_pctl"] >= 65:
        strengths.append(f"Offense ranks in the {row['offense_pctl']:.0f}th percentile.")
    if row["defense_pctl"] >= 65:
        strengths.append(f"Defense ranks in the {row['defense_pctl']:.0f}th percentile.")
    if row["special_teams_pctl"] >= 70:
        strengths.append("Special teams is a genuine edge.")
    if row["model_rating"] > 5:
        strengths.append("Opponent-adjusted margin is well above average.")
    st.write("\n".join(f"- {s}" for s in strengths) or "- No standout strengths yet.")
with sc2:
    st.markdown("**Vulnerabilities**")
    vulns = []
    if row["offense_pctl"] < 40:
        vulns.append(f"Offense is below average ({row['offense_pctl']:.0f}th pctl).")
    if row["defense_pctl"] < 40:
        vulns.append(f"Defense is below average ({row['defense_pctl']:.0f}th pctl).")
    if row["points_against_pg"] > 28:
        vulns.append(f"Giving up {row['points_against_pg']:.0f} points per game.")
    st.write("\n".join(f"- {v}" for v in vulns) or "- No glaring weaknesses yet.")
with sc3:
    st.markdown("**Best win / worst loss**")
    if not played.empty:
        played = played.assign(
            opp_rating=played["opponent"].map(
                dict(zip(ratings["team"], ratings["model_rating"]))))
        wins = played[played["result"] == "W"]
        losses = played[played["result"] == "L"]
        if not wins.empty:
            bw = wins.loc[wins["opp_rating"].idxmax()]
            st.write(f"- Best win: **{bw['opponent']}** (Wk {int(bw['week'])})")
        if not losses.empty:
            wl = losses.loc[losses["opp_rating"].idxmin()]
            st.write(f"- Worst loss: **{wl['opponent']}** (Wk {int(wl['week'])})")
    if not tg.empty:
        future = tg[~tg["completed"]]
        if not future.empty:
            key = future.loc[(future["team_win_prob"] - 0.5).abs().idxmin()]
            st.write(f"- Most pivotal remaining game: **{key['opponent']}** "
                     f"(Wk {int(key['week'])}, {key['team_win_prob']*100:.0f}% win prob)")

# ---- season trend ------------------------------------------------
if not played.empty and len(played) >= 3:
    st.subheader("Trajectory")
    played = played.sort_values("week")
    rolling = played["team_win_prob"].expanding().mean()
    fig = go.Figure()
    fig.add_scatter(x=played["week"], y=played["team_win_prob"] * 100,
                    mode="markers+lines", name="Model win prob (pre-game)",
                    line=dict(color=ACCENT))
    fig.update_layout(title="Pre-game win probability across the season so far",
                      yaxis_title="%", xaxis_title="Week", height=320, **PLOTLY_LAYOUT)
    st.plotly_chart(fig, use_container_width=True, key="tp_trajectory")

st.caption("Playoff / championship figures are estimates from the simulated "
           "selection model. See Methodology for details.")
