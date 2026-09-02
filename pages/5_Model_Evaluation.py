"""Model Evaluation — how it was tested, where it works, where it doesn't."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.ui import state
from src.ui.components import calibration_plot
from src.ui.theme import ACCENT, BAD, GOOD, MUTED, PLOTLY_LAYOUT, apply_theme

st.set_page_config(page_title="Model Evaluation", page_icon="🔬", layout="wide")
apply_theme()
state.ensure_globals()

card = state.model_card()
ev = card.get("evaluation", {})

st.title("Model Evaluation")
st.markdown(
    "Every number here is **out-of-sample**: the model never saw these games "
    "during training or calibration. Training-set accuracy is not reported "
    "because it would be misleading."
)

split_names = [s for s in ("validation", "test") if s in ev]
if not split_names:
    st.warning("No evaluation found. Run `python scripts/train_models.py`.")
    st.stop()

tab_objs = st.tabs([f"{s.capitalize()} "
                    f"({card.get(s + '_season', '') or ev.get('splits', {}).get(s, '')})"
                    for s in split_names])

for tab, split in zip(tab_objs, split_names):
    with tab:
        block = ev[split]
        wm = block["win_model"]
        sm = block["score_model"]

        st.subheader("Win probability model vs. baselines")
        rows = []
        for name, key in [("Ensemble (primary)", "ensemble"),
                          ("— Gradient-boosted alone", "gbm"),
                          ("— Elo", "elo"),
                          ("— Logistic regression", "logistic"),
                          ("Home win rate", "home_rate")]:
            m = wm.get(key, {})
            rows.append({
                "Model": name,
                "Games": m.get("n"),
                "Accuracy": f"{m.get('accuracy', float('nan')):.3f}",
                "Log loss": f"{m.get('log_loss', float('nan')):.4f}",
                "Brier": f"{m.get('brier', float('nan')):.4f}",
                "ROC AUC": f"{m.get('roc_auc', float('nan')):.3f}"
                           if m.get("roc_auc") == m.get("roc_auc") else "—",
                "Calibration error": f"{m.get('calibration_error', float('nan')):.4f}",
                "Acc. vs favorite": f"{m.get('accuracy_vs_market_favorite', float('nan')):.3f}"
                                    if "accuracy_vs_market_favorite" in m else "—",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption("Lower log loss / Brier / calibration error is better. "
                   "'Acc. vs favorite' compares the pick to the Elo favorite "
                   "(used as a market proxy when no betting line is available).")

        c1, c2 = st.columns(2)
        with c1:
            cal = pd.DataFrame(block["calibration_table"])
            st.plotly_chart(calibration_plot(cal, f"Calibration — {split}"),
                            use_container_width=True)
        with c2:
            conf = pd.DataFrame(block["by_confidence"])
            if not conf.empty:
                fig = go.Figure()
                fig.add_bar(x=conf["confidence_range"], y=conf["model_confidence"] * 100,
                            name="Model says", marker_color=MUTED)
                fig.add_bar(x=conf["confidence_range"], y=conf["actual_accuracy"] * 100,
                            name="Actually right", marker_color=ACCENT)
                fig.update_layout(title="Confidence vs. reality", barmode="group",
                                  yaxis_title="%", height=380, **PLOTLY_LAYOUT)
                st.plotly_chart(fig, use_container_width=True)

        st.subheader("Score prediction")
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("MAE — point margin", f"{sm['mae_point_margin']:.1f}")
        s2.metric("MAE — team score", f"{sm['mae_team_score']:.1f}")
        s3.metric("RMSE — margin", f"{sm['rmse_point_margin']:.1f}")
        s4.metric("Within 7 pts", f"{sm['within_7']*100:.0f}%")
        wdf = pd.DataFrame({
            "Band": ["± 3", "± 7", "± 10", "± 14"],
            "Share of games": [sm["within_3"], sm["within_7"], sm["within_10"],
                               sm["within_14"]],
        })
        fig = go.Figure(go.Bar(x=wdf["Band"], y=wdf["Share of games"] * 100,
                               marker_color=GOOD))
        fig.update_layout(title="Margin predicted within N points", yaxis_title="%",
                          height=300, **PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Where it is strong and weak")
        cc1, cc2 = st.columns(2)
        with cc1:
            st.markdown("**By conference**")
            bc = pd.DataFrame(block["by_conference"])
            if not bc.empty:
                st.dataframe(bc.round(4), use_container_width=True, hide_index=True)
        with cc2:
            st.markdown("**By game location**")
            bl = pd.DataFrame(block["by_location"])
            if not bl.empty:
                st.dataframe(bl.round(4), use_container_width=True, hide_index=True)

st.divider()
st.subheader("Per-season backtest")
bs = pd.DataFrame(ev.get("by_season", []))
if not bs.empty:
    fig = go.Figure()
    fig.add_bar(x=bs["season"].astype(str), y=bs["log_loss"], name="Log loss",
                marker_color=ACCENT)
    fig.add_scatter(x=bs["season"].astype(str), y=bs["calibration_error"],
                    name="Calibration error", yaxis="y2", line=dict(color=BAD))
    fig.update_layout(title="Held-out performance by season", height=340,
                      xaxis=dict(type="category"),
                      yaxis=dict(title="Log loss"),
                      yaxis2=dict(title="Calibration error", overlaying="y",
                                  side="right"),
                      **{k: v for k, v in PLOTLY_LAYOUT.items()
                         if k not in ("yaxis", "xaxis")})
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(bs.round(4), use_container_width=True, hide_index=True)

st.divider()
st.markdown(f"""
### How to read this

* **Why probabilities change.** Ratings update after every completed game, so a
  team's win/playoff/title odds move as results come in — especially early in
  the season when a single game is a large share of the sample.
* **Why a 70% pick still loses ~30% of the time.** A calibrated 70% means: across
  all games we call 70%, the favorite should win about 70 of every 100. Thirty
  losses out of a hundred is the model working correctly, not failing.
* **Where the model is least reliable.** Early-season games (thin current-season
  data, leaning on preseason priors), teams with little history, and games with
  major unmodeled news (injuries, weather, suspensions).
* **Backtest scope.** {len(bs)} held-out season(s). Connect multi-season history
  via CFBD for a deeper backtest (see Methodology).
""")
st.caption(f"Model version {card.get('model_version','—')} · trained "
           f"{str(card.get('trained_at_utc',''))[:10]} · backend "
           f"{card.get('backend','—')} · calibration {card.get('calibration_method','—')}.")
