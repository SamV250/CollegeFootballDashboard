"""Shared matchup-explanation rendering (used by several pages)."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.explainability.shap_explain import explain_game
from src.ui.components import (
    factor_list,
    headline,
    win_prob_bar,
)
from src.ui.formatting import pct


def build_explanation(predictor, feat_row: pd.DataFrame, prev_prob: float | None = None) -> dict:
    pred = predictor.predict(feat_row).iloc[0]
    return explain_game(predictor, feat_row, pred, prev_prediction=prev_prob)


def render_explanation(exp: dict, mode: str = "Executive", key_prefix: str = "mx") -> None:
    home, away = exp["home_team"], exp["away_team"]
    fav, dog = exp["favorite"], exp["underdog"]
    p_fav = exp["favorite_win_prob"]
    kp = str(key_prefix).replace(" ", "_")

    headline(f"{exp['summary']}")
    win_prob_bar(home, away, exp["home_win_prob"], key=f"{kp}_winprob")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Predicted winner", fav, f"{pct(p_fav)} win prob")
    c2.metric("Projected score",
              f"{exp['pred_home_points']:.0f}–{exp['pred_away_points']:.0f}",
              f"{home} – {away}")
    c3.metric("Confidence", exp["confidence"],
              help="High / Moderate / Low / Toss-up, from how far the win "
                   "probability sits from 50%.")
    c4.metric("Upset probability", pct(exp["upset_prob"]),
              help=f"Chance {dog} wins.")

    lo, hi = exp["plausible_margin_range"]
    st.markdown(
        f"<span class='small-muted'>Plausible final margin (~80% band): "
        f"<b>{fav}</b> by {max(lo,0):.0f} to {hi:.0f} "
        f"— outcomes outside this range happen roughly 1 game in 5.</span>",
        unsafe_allow_html=True,
    )

    st.markdown(f"#### Why the model favors {fav}")
    factor_list(exp["favorite_factors"], "No dominant factors — this looks close.")

    st.markdown(f"#### What gives {dog} a chance")
    factor_list(exp["underdog_factors"], "Few counter-arguments — a one-sided projection.")

    if exp["missing_data"]:
        for note in exp["missing_data"]:
            st.warning(note, icon="⚠️")

    mv = exp.get("prediction_movement")
    if mv and mv.get("changed_substantially"):
        st.info(
            f"This projection moved {mv['delta_win_prob']:+.0%} in win probability "
            f"over the past week — recent results changed the picture.", icon="📈")

    if mode == "Analyst":
        with st.expander("Analyst detail — feature contributions (SHAP)"):
            rows = []
            for f in exp["favorite_factors"] + exp["underdog_factors"]:
                rows.append({
                    "Feature": f["label"],
                    "Direction": f"favors {f['favors']}",
                    "Magnitude": f["magnitude"],
                    "Std. contribution": round(f["z"], 3),
                    "Definition": f["tooltip"],
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True,
                         key=f"{kp}_shap_tbl")
            st.caption(
                f"SHAP base value (model's average log-odds output): "
                f"{exp['shap_base_value']:.3f}. Contributions are standardized by "
                f"the total absolute SHAP mass for the game.")
