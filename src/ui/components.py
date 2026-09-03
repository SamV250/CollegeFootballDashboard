"""Reusable Streamlit UI components."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.ui.formatting import fmt_dt, humanize_hours
from src.ui.theme import ACCENT, BAD, GOOD, LINE, MUTED, PLOTLY_LAYOUT

DISCLAIMER = (
    "Predictions are probabilistic estimates from a statistical model, not "
    "guarantees. A 70% pick is expected to lose about 30% of the time. "
    "Figures update as games are played."
)


def disclaimer_banner() -> None:
    st.markdown(
        f"<div class='cfb-card' style='border-color:{LINE}'>"
        f"<span class='small-muted'>ℹ️ {DISCLAIMER}</span></div>",
        unsafe_allow_html=True,
    )


def freshness_badge(freshness: dict, tz_name: str = "America/New_York") -> None:
    state = freshness.get("state", "unknown")
    cls = {"current": "freshness-current", "delayed": "freshness-delayed",
           "stale": "freshness-stale"}.get(state, "small-muted")
    label = {"current": "CURRENT", "delayed": "DELAYED", "stale": "STALE"}.get(
        state, "UNKNOWN")
    cols = st.columns([1, 1, 1, 1])
    with cols[0]:
        st.markdown(f"<div class='cfb-card'><h4>Data status</h4>"
                    f"<span class='{cls}'>{label}</span><br>"
                    f"<span class='small-muted'>via {freshness.get('source', '—')} · "
                    f"{humanize_hours(freshness.get('age_hours'))}</span></div>",
                    unsafe_allow_html=True)
    with cols[1]:
        st.markdown(f"<div class='cfb-card'><h4>Data updated</h4>"
                    f"<span>{fmt_dt(freshness.get('last_refresh_utc'), tz_name)}</span></div>",
                    unsafe_allow_html=True)
    with cols[2]:
        st.markdown(f"<div class='cfb-card'><h4>Latest game included</h4>"
                    f"<span>{fmt_dt(freshness.get('latest_completed_game_utc'), tz_name)}</span></div>",
                    unsafe_allow_html=True)
    with cols[3]:
        st.markdown(f"<div class='cfb-card'><h4>Next scheduled refresh</h4>"
                    f"<span>{fmt_dt(freshness.get('next_refresh_utc'), tz_name)}</span></div>",
                    unsafe_allow_html=True)


def big_number_card(title: str, value: str, sub: str = "", accent: str = ACCENT) -> None:
    st.markdown(
        f"<div class='cfb-card'><h4>{title}</h4>"
        f"<div class='bignum' style='color:{accent}'>{value}</div>"
        f"<div class='bignum-sub'>{sub}</div></div>",
        unsafe_allow_html=True,
    )


def headline(text: str) -> None:
    st.markdown(f"<div class='headline'>{text}</div>", unsafe_allow_html=True)


def pill(text: str, kind: str = "neutral") -> str:
    return f"<span class='pill pill-{kind}'>{text}</span>"


def factor_list(factors: Iterable[dict], empty_msg: str = "No strong factors identified.") -> None:
    factors = list(factors)
    if not factors:
        st.markdown(f"<span class='small-muted'>{empty_msg}</span>", unsafe_allow_html=True)
        return
    mag_cls = {"major advantage": "mag-major", "moderate advantage": "mag-moderate",
               "slight advantage": "mag-slight", "essentially even": "mag-even"}
    rows = ""
    for f in factors:
        cls = mag_cls.get(f["magnitude"], "mag-slight")
        rows += (
            f"<div class='factor-row'><span>{f['sentence']}"
            f"<br><span class='small-muted'>{f['label']}</span></span>"
            f"<span class='{cls}'>{f['magnitude'].replace(' advantage','')}</span></div>"
        )
    st.markdown(f"<div class='cfb-card'>{rows}</div>", unsafe_allow_html=True)


def percentile_bar(label: str, value: float, tier: str) -> None:
    v = 0 if value is None or pd.isna(value) else max(0, min(100, value))
    color = GOOD if v >= 60 else (ACCENT if v >= 35 else BAD)
    st.markdown(
        f"<div style='margin:6px 0'>"
        f"<div style='display:flex;justify-content:space-between'>"
        f"<span>{label}</span><span class='small-muted'>{tier} · {v:.0f}th pctl</span></div>"
        f"<div style='background:{LINE};border-radius:6px;height:10px;margin-top:4px'>"
        f"<div style='width:{v}%;background:{color};height:10px;border-radius:6px'></div>"
        f"</div></div>",
        unsafe_allow_html=True,
    )


def win_prob_bar(home: str, away: str, p_home: float, key: str | None = None) -> None:
    fig = go.Figure()
    fig.add_bar(y=["Win probability"], x=[p_home * 100], orientation="h",
                marker_color=ACCENT, name=home,
                text=f"{home} {p_home*100:.0f}%", textposition="inside")
    fig.add_bar(y=["Win probability"], x=[(1 - p_home) * 100], orientation="h",
                marker_color=MUTED, name=away,
                text=f"{away} {(1-p_home)*100:.0f}%", textposition="inside")
    fig.update_layout(barmode="stack", height=90, showlegend=False,
                      xaxis=dict(range=[0, 100], showticklabels=False),
                      **{k: v for k, v in PLOTLY_LAYOUT.items()
                         if k not in ("xaxis", "yaxis")})
    fig.update_yaxes(showticklabels=False)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False},
                    key=key or f"winprob_{home}_{away}".replace(" ", "_"))


def calibration_plot(cal_df: pd.DataFrame, title: str = "Calibration") -> go.Figure:
    fig = go.Figure()
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color=MUTED),
                    name="Perfect calibration")
    if not cal_df.empty:
        fig.add_scatter(
            x=cal_df["mean_predicted"], y=cal_df["actual_win_rate"],
            mode="markers+lines", marker=dict(size=cal_df["n_games"].clip(6, 26),
                                              color=ACCENT),
            name="Model",
            hovertext=[f"n={int(n)}" for n in cal_df["n_games"]],
        )
    fig.update_layout(title=title, height=380,
                      xaxis_title="Predicted home win probability",
                      yaxis_title="Actual home win rate", **PLOTLY_LAYOUT)
    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(range=[0, 1])
    return fig


def bar_chart(df: pd.DataFrame, x: str, y: str, title: str, color: str = ACCENT,
              horizontal: bool = True) -> go.Figure:
    fig = go.Figure()
    if horizontal:
        fig.add_bar(y=df[y], x=df[x], orientation="h", marker_color=color)
        fig.update_layout(yaxis=dict(autorange="reversed"))
    else:
        fig.add_bar(x=df[x], y=df[y], marker_color=color)
    fig.update_layout(title=title, height=max(320, 26 * len(df)), **PLOTLY_LAYOUT)
    return fig
