"""Visual theme: dark football-operations aesthetic.

One place for colours, fonts and the CSS injected into every page.  Keep
this conservative -- large probability numbers, white cards on a charcoal
ground, muted accents, minimal motion, accessible contrast.
"""

from __future__ import annotations

import streamlit as st

# -- palette -----------------------------------------------------------------
NAVY = "#0f1830"
CHARCOAL = "#131a2b"
CARD = "#1c2536"
CARD_LIGHT = "#243049"
INK = "#f4f6fb"
MUTED = "#9aa7bd"
LINE = "#2b3648"
ACCENT = "#4f83cc"          # primary accent (calm blue)
GOOD = "#3fb98c"            # rising / favourable
BAD = "#e2686f"             # falling / risk
WARN = "#d9a441"            # caution / delayed data

CONFERENCE_COLORS = {
    "SEC": "#c8452f", "Big Ten": "#2f5fc8", "Big 12": "#c88a2f",
    "ACC": "#2f9bc8", "Pac-12": "#3f8f5f", "American": "#7d6fc8",
    "Mountain West": "#b5843b", "Sun Belt": "#c85f8a", "MAC": "#5f9f6f",
    "Conference USA": "#6f7f8f", "FBS Independents": "#8a8f9a",
}

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=INK, family="Inter, 'Segoe UI', system-ui, sans-serif", size=13),
    margin=dict(l=40, r=20, t=40, b=40),
    colorway=[ACCENT, GOOD, WARN, BAD, "#8e79c9", "#59b3c4", "#d68f5e"],
    xaxis=dict(gridcolor=LINE, zerolinecolor=LINE),
    yaxis=dict(gridcolor=LINE, zerolinecolor=LINE),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
)

_CSS = f"""
<style>
:root {{
  --navy: {NAVY}; --card: {CARD}; --ink: {INK}; --muted: {MUTED};
  --line: {LINE}; --accent: {ACCENT}; --good: {GOOD}; --bad: {BAD};
}}
.stApp {{ background: {CHARCOAL}; color: {INK}; }}
section[data-testid="stSidebar"] {{ background: {NAVY}; border-right: 1px solid {LINE}; }}
h1, h2, h3, h4 {{ color: {INK}; font-weight: 700; letter-spacing: -0.01em; }}
p, span, li, label, div {{ color: {INK}; }}
.small-muted {{ color: {MUTED}; font-size: 0.85rem; }}

.cfb-card {{
  background: {CARD}; border: 1px solid {LINE}; border-radius: 14px;
  padding: 18px 20px; margin-bottom: 14px;
}}
.cfb-card h4 {{ margin: 0 0 6px 0; font-size: 0.95rem; color: {MUTED};
  text-transform: uppercase; letter-spacing: 0.06em; }}
.bignum {{ font-size: 2.6rem; font-weight: 800; line-height: 1.05; }}
.bignum-sub {{ color: {MUTED}; font-size: 0.9rem; }}
.headline {{ font-size: 1.15rem; font-weight: 650; margin: 2px 0 10px 0; }}
.pill {{ display:inline-block; padding: 2px 10px; border-radius: 999px;
  font-size: 0.78rem; font-weight: 600; border: 1px solid {LINE}; }}
.pill-good {{ background: rgba(63,185,140,0.14); color: {GOOD}; border-color: rgba(63,185,140,0.4); }}
.pill-bad {{ background: rgba(226,104,111,0.14); color: {BAD}; border-color: rgba(226,104,111,0.4); }}
.pill-warn {{ background: rgba(217,164,65,0.14); color: {WARN}; border-color: rgba(217,164,65,0.4); }}
.pill-neutral {{ background: rgba(154,167,189,0.12); color: {MUTED}; }}

.freshness-current {{ color: {GOOD}; font-weight: 700; }}
.freshness-delayed {{ color: {WARN}; font-weight: 700; }}
.freshness-stale {{ color: {BAD}; font-weight: 700; }}

.factor-row {{ display:flex; justify-content: space-between; gap: 12px;
  padding: 7px 0; border-bottom: 1px dashed {LINE}; }}
.factor-row:last-child {{ border-bottom: none; }}
.mag-major {{ color: {GOOD}; font-weight: 700; }}
.mag-moderate {{ color: {ACCENT}; font-weight: 650; }}
.mag-slight {{ color: {MUTED}; font-weight: 600; }}
.mag-even {{ color: {MUTED}; }}

div[data-testid="stMetricValue"] {{ font-size: 2rem; font-weight: 800; }}
.stDataFrame {{ border: 1px solid {LINE}; border-radius: 10px; }}
.block-container {{ padding-top: 2.2rem; max-width: 1300px; }}
</style>
"""


def apply_theme() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


def conf_color(conf: str) -> str:
    return CONFERENCE_COLORS.get(conf, MUTED)
