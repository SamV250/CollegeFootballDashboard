"""Methodology & Data — sources, dictionary, refresh, limitations."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.config import get_settings
from src.features.dictionary import feature_dictionary
from src.ui import state
from src.ui.components import demo_data_banner, freshness_badge
from src.ui.theme import apply_theme

st.set_page_config(page_title="Methodology & Data", page_icon="📚", layout="wide")
apply_theme()
mode, tz = state.ensure_globals()
settings = get_settings()
card = state.model_card()

st.title("Methodology & Data")
demo_data_banner()

st.header("Data status")
freshness_badge(state.freshness(), tz)

st.header("Data sources")
st.markdown(f"""
Sources are tried in priority order; the first that returns valid, non-empty data wins.
Current priority: **{' → '.join(settings.source_priority)}**.

| Source | What it provides | When it is used |
|---|---|---|
| **CollegeFootballData API** | Games, schedules, conferences, advanced team stats | Preferred, when `CFBD_API_KEY` is set |
| **ESPN public endpoints** | Schedules + scores (no advanced metrics) | Fallback when CFBD is unavailable |
| **Local CSV / Parquet** | Whatever you drop in `data/raw/` (canonical schema) | Offline / air-gapped use |
| **Synthetic demo** | A full 5-season simulated universe | So the dashboard runs with no credentials |

A failing source never erases previously validated data — the last good dataset keeps serving
with a *Delayed* or *Stale* badge. Conflicting values across sources are logged, not silently merged.
""")

demo = state.freshness().get("source") == "demo"
if demo:
    st.info(settings.config["demo"]["generated_label"], icon="🧪")

st.header("Refresh process")
st.markdown(f"""
`python scripts/update_dashboard.py` runs the full pipeline idempotently:

1. fetch latest data → 2. validate → 3. diff against the store → 4. upsert only new/changed
rows → 5. rebuild leakage-safe rolling features → 6. update Elo & team ratings →
7. regenerate game predictions → 8. run the Monte Carlo season simulation →
9. save dashboard artifacts → 10. log successes, warnings, failures.

**Cadence:** every {settings.config['data']['refresh_interval_minutes'] // 60} h in-season,
every {settings.config['data']['refresh_interval_gameday_minutes']} min on game days,
results shortly after a game goes final. Times are stored in UTC and shown in your
selected time zone. *Current* = refreshed within the expected interval; *Delayed* = one
refresh missed; *Stale* = no refresh for {settings.config['data']['stale_after_hours']} h.
Scheduling examples: `.github/workflows/refresh.yml` (GitHub Actions) and `scripts/crontab.example`.
""")

st.header("Modeling methodology")
_train_end = (card.get("training_seasons") or ["—"])[-1]
st.markdown(f"""
* **Baselines** — home win rate, an Elo rating model (margin-of-victory adjusted,
  between-season regression), and L2 logistic regression. The primary model must beat these.
* **Primary game model** — gradient-boosted trees (LightGBM/XGBoost) predicting home
  win probability, with **isotonic / Platt calibration** so the probabilities mean what
  they say. Two more boosted models predict point **margin** and **total**, from which
  team scores are recovered.
* **Features** are opponent-adjusted and team-relative (offense vs. opposing defense,
  recent vs. season-long form, Elo gap, rest, special teams, line play, red zone…).
  Noisy quantities (turnover margin) are regressed toward the mean.
* **Leakage controls** — for a game on date *D*, only information available before
  kickoff is used. Rolling stats are `shift(1)` so a game never enters its own features.
  Splits are strictly chronological (train ≤ {_train_end},
  validate {card.get('validation_season', '—')}, test {card.get('test_season', '—')}).
""")

st.header("Simulation methodology")
st.markdown(f"""
Monte Carlo, **{settings.config['simulation']['n_iterations']:,} iterations**. Each iteration
simulates every remaining regular-season game from the calibrated win probability, builds
conference standings, plays the championship games, scores every team with a **transparent
selection model**, selects and seeds a {settings.playoff['format']['n_teams']}-team field
({settings.playoff['format']['highest_ranked_conf_champs']} highest-ranked conference champions
guaranteed, top {settings.playoff['format']['n_first_round_byes']} seeds get byes), and plays
the bracket. Neutral-site probabilities use the Elo model.

The selection model (weights in `config/playoff.yaml`) combines winning percentage,
opponent-adjusted efficiency, strength of schedule, quality wins, bad losses, a
conference-title bonus, conference strength and a committee-style prior. **It is a
transparent proxy, not the real committee** — all playoff/championship numbers are
labelled estimates.
""")

st.header("Data dictionary")
fd = feature_dictionary()
st.dataframe(pd.DataFrame(fd), use_container_width=True, hide_index=True, height=520)
st.caption("Full prose version: `DATA_DICTIONARY.md`.")

st.header("Known limitations")
st.markdown("""
* No injury, weather, suspension or personnel data in the base build — supply manual
  adjustments in the Matchup Simulator.
* Betting lines are not ingested unless a reliable, licensed source is configured.
* The selection model cannot perfectly reproduce committee judgement.
* Live win probability is **not** claimed unless timely play-by-play is connected and validated.
* The demo dataset is synthetic; absolute numbers are illustrative until a live source is connected.
* Backtest depth is limited by available history — connect CFBD for multi-season validation.
""")

st.header("Model card")
st.json(card, expanded=False)
st.caption("Also written to `models/model_card.json` and summarised in `MODEL_CARD.md`.")
