"""Weekly Matchups — every upcoming game with prediction + explanation."""

from __future__ import annotations

import streamlit as st

from src.ui import state
from src.ui.components import disclaimer_banner
from src.ui.formatting import fmt_dt
from src.ui.matchup import build_explanation, render_explanation
from src.ui.theme import apply_theme

st.set_page_config(page_title="Weekly Matchups", page_icon="🗓️", layout="wide")
apply_theme()
mode, tz = state.ensure_globals()

st.title("Weekly Matchups")
st.caption("Predicted winner, score, confidence and the football reasons behind "
           "each pick. Every figure is a probability, not a guarantee.")
disclaimer_banner()

season = state.active_season()
upc = state.upcoming_predictions(season)
if upc.empty:
    st.info("No upcoming games in the current dataset — the regular season may be "
            "complete, or data has not been refreshed.")
    st.stop()

art = state.dashboard_artifacts()
tp = art["team_probabilities"].set_index("team")

upc = upc.copy()
upc["favorite"] = [h if p >= 0.5 else a for h, a, p in
                   zip(upc["home_team"], upc["away_team"], upc["home_win_prob"])]
upc["fav_prob"] = upc["home_win_prob"].where(upc["home_win_prob"] >= 0.5,
                                             1 - upc["home_win_prob"])
upc["upset_prob"] = 1 - upc["fav_prob"]
upc["playoff_impact"] = [
    tp["p_playoff"].get(h, 0) + tp["p_playoff"].get(a, 0)
    for h, a in zip(upc["home_team"], upc["away_team"])
]

# ---- filters -------------------------------------------------------------
f = st.container()
c1, c2, c3, c4 = f.columns(4)
weeks = sorted(upc["week"].unique())
wk = c1.selectbox("Week", ["All"] + list(weeks), index=1 if weeks else 0)
confs = sorted(set(upc["home_conference"]) | set(upc["away_conference"]))
conf = c2.selectbox("Conference", ["All"] + confs)
loc = c3.selectbox("Location", ["All", "Home favorite", "Road favorite", "Neutral"])
min_upset = c4.slider("Min upset probability", 0.0, 0.5, 0.0, 0.05)
c5, c6 = st.columns(2)
min_impact = c5.slider("Min combined playoff impact", 0.0, 2.0, 0.0, 0.1)
ranked_only = c6.checkbox("Ranked teams only (model top 25)")

view = upc.copy()
if wk != "All":
    view = view[view["week"] == wk]
if conf != "All":
    view = view[(view["home_conference"] == conf) | (view["away_conference"] == conf)]
if loc == "Neutral":
    view = view[view["neutral_site"]]
elif loc == "Home favorite":
    view = view[(~view["neutral_site"]) & (view["home_win_prob"] >= 0.5)]
elif loc == "Road favorite":
    view = view[(~view["neutral_site"]) & (view["home_win_prob"] < 0.5)]
view = view[view["upset_prob"] >= min_upset]
view = view[view["playoff_impact"] >= min_impact]
if ranked_only:
    ranked = set(state.team_ratings(season).head(25)["team"])
    view = view[view["home_team"].isin(ranked) | view["away_team"].isin(ranked)]

st.markdown(f"**{len(view)} game(s)** match your filters.")

# ---- summary table -----------------------------------------------------
tbl = view.copy()
tbl["Matchup"] = [f"{a} at {h}" if not nt else f"{a} vs {h}"
                  for h, a, nt in zip(tbl["home_team"], tbl["away_team"], tbl["neutral_site"])]
tbl["Kickoff"] = tbl["date"].apply(lambda d: fmt_dt(d, tz))
tbl["Predicted score"] = [f"{hp:.0f}–{ap:.0f}" for hp, ap in
                          zip(tbl["pred_home_points"], tbl["pred_away_points"])]
tbl["Win prob"] = (tbl["fav_prob"] * 100).round(0).astype(int).astype(str) + "%"
tbl["Upset"] = (tbl["upset_prob"] * 100).round(0).astype(int).astype(str) + "%"
show = tbl[["week", "Matchup", "favorite", "Win prob", "Predicted score", "Upset",
            "playoff_impact", "Kickoff"]].rename(
    columns={"week": "Wk", "favorite": "Pick", "playoff_impact": "Playoff impact"})
show["Playoff impact"] = show["Playoff impact"].round(2)
st.dataframe(show.sort_values(["Wk", "Playoff impact"], ascending=[True, False]),
             use_container_width=True, hide_index=True, height=380)

# ---- per-game explanations -------------------------------------------
st.divider()
st.subheader("Game breakdowns")
predictor = state.bundle()["predictor"]
feat = state.feature_matrix()

order = view.sort_values(["week", "playoff_impact"], ascending=[True, False])
limit = st.slider("How many breakdowns to render", 1, min(30, len(order)),
                  min(8, len(order)))
for gid in order["game_id"].head(limit):
    row = feat[feat["game_id"] == gid]
    if row.empty:
        continue
    g = row.iloc[0]
    with st.expander(f"Week {int(g['week'])} — {g['away_team']} at {g['home_team']}  "
                     f"({fmt_dt(g['date'], tz)})", expanded=(limit <= 3)):
        exp = build_explanation(predictor, row)
        render_explanation(exp, mode)
