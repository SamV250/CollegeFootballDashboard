# Model Card — College Football Prediction Dashboard

A living summary; the authoritative machine-readable version is
`models/model_card.json`, regenerated on every training run.

## Model details

| | |
|---|---|
| **Version** | `1.2026.09.01` (schema 1) |
| **Primary win-prob model** | **Equal-weight ensemble** of (a) an isotonic-calibrated gradient-boosted classifier, (b) an L2 logistic regression on a stable feature subset, and (c) the Elo win probability. On real data no single method reliably beats the others; the blend is a touch better and better calibrated. Weights in `config.model.ensemble`. |
| **Other components** | Elo rating model (also a standalone baseline); two gradient-boosted regressors for point margin and total; Monte Carlo season/playoff simulator; SHAP-based explanation layer (explains the boosted component) |
| **Backend** | LightGBM (XGBoost selectable via `config.model.backend`) |
| **Features** | 30 opponent-adjusted, team-relative matchup features (see `DATA_DICTIONARY.md`); of the advanced metrics, CFBD currently supplies PPA/EPA, success rate, explosiveness and line yards on offense and defense |
| **Calibration** | Isotonic regression, 4-fold cross-validated on the training split |
| **Training data** | CollegeFootballData, FBS-vs-FBS games, seasons **2017–2022** (3,868 games); validated on **2023** (792); tested on **2024** (798); backtested through 2025 |
| **Randomness** | Seeded throughout (`model.random_seed=1729`, `simulation.random_seed=20262027`) |
| **Last trained** | recorded in `models/model_card.json → trained_at_utc` |

## Intended use

* **Primary:** decision-support and briefing for coaches, athletic-department
  staff, analysts, media and fans — game previews, playoff-odds context,
  what-if exploration.
* **Appropriate:** comparing the model's read to public polls; identifying
  high-leverage games; framing uncertainty for stakeholders.
* **Not intended for:** wagering; any use that treats a probability as a
  certainty; personnel or eligibility decisions about individual athletes;
  contexts where a wrong probabilistic call causes material harm.

## Out-of-sample performance (real CollegeFootballData)

> Figures below are from the training run described above (trained 2017–2022,
> tested on 2024). `models/model_card.json` and the in-app **Model Evaluation**
> page always carry the current numbers.

### Win probability — held-out 2024 season (n = 798)

| Model | Log loss | Brier | Accuracy | Calibration error |
|---|---|---|---|---|
| **Ensemble (primary)** | 0.567 | 0.194 | 0.693 | 0.046 |
| — Gradient-boosted alone | 0.590 | 0.197 | 0.687 | 0.060 |
| — Logistic regression | 0.565 | 0.194 | 0.695 | 0.033 |
| — Elo | 0.577 | 0.198 | 0.684 | 0.046 |
| Home-team win rate | 0.679 | 0.243 | 0.584 | 0.003 |

### Win probability — 2023 validation (n = 792)

| Model | Log loss | Brier | Accuracy | Calibration error |
|---|---|---|---|---|
| **Ensemble (primary)** | **0.548** | **0.184** | 0.731 | **0.024** |
| — Gradient-boosted alone | 0.570 | 0.186 | 0.735 | 0.033 |
| — Logistic regression | 0.548 | 0.184 | 0.729 | 0.031 |
| — Elo | 0.553 | 0.187 | 0.716 | 0.032 |
| Home-team win rate | 0.680 | 0.244 | 0.580 | 0.001 |

**Honest read:** on real college-football data the boosted model, Elo and
logistic regression are all within ~0.02 log loss of each other — once you have
a good opponent-adjusted rating, method choice barely matters. The gradient
booster on its own does **not** beat the linear baselines here. Their equal-weight
average (the shipped model) matches the best single model on accuracy and is the
best-calibrated of the lot, which is why it is the primary. Every baseline is
shown alongside it on the Evaluation page so this stays visible.

Accuracy against the Elo favorite (a market proxy — no betting line is ingested):
ensemble agrees with the favorite ~93% of the time on both splits.

### Score prediction (held-out)

| Split | MAE margin | MAE team score | RMSE margin | Within 7 pts | Within 14 pts |
|---|---|---|---|---|---|
| 2023 | 13.0 | 9.6 | 16.5 | 34% | 62% |
| 2024 | 13.3 | 9.6 | 16.7 | 33% | 60% |

### Per-season backtest (ensemble)

| Season | n | Accuracy | Log loss | Calibration error |
|---|---|---|---|---|
| 2023 | 792 | 0.731 | 0.548 | 0.024 |
| 2024 | 798 | 0.693 | 0.567 | 0.046 |
| 2025 | 808 | 0.718 | 0.538 | 0.029 |

The current (2026) season is excluded from the backtest table until enough games
are complete to be meaningful.

## Evaluation methodology

* **Chronological splits only** — train on earlier seasons, test on later ones.
  Games are never randomly shuffled across time.
* **Leakage guards** — a game's own result and box score cannot enter its own
  feature row; rolling stats are `shift(1)`-ed; verified by
  `tests/test_leakage.py`.
* **Baseline comparison is mandatory** — Elo, logistic regression and the naive
  home-win-rate model are evaluated on identical rows.
* **Calibration is reported next to accuracy** everywhere.

## Limitations

* No injury, weather, suspension, personnel or travel-fatigue data in the base
  build. Use the Matchup Simulator's manual adjustments.
* Betting markets are not ingested unless a licensed source is configured; the
  "accuracy vs. favorite" metric uses the Elo favorite as a proxy.
* The **playoff selection model is a transparent proxy**, not the real
  committee. All playoff/championship numbers are estimates.
* **Live win probability is not claimed** — it requires timely play-by-play and
  separate validation.
* Backtest depth is limited by available history. Connect multi-season CFBD data
  for a deeper backtest.
* The demo dataset is synthetic; absolute values are illustrative until a live
  source is connected.

## Ethical considerations

* **Uncertainty is surfaced, not hidden.** Probabilities are calibrated, ranges
  are shown, and the "a 70% pick loses ~30% of the time" message is repeated.
* **Facts, estimates and assumptions are kept distinct** in the UI (observed
  results vs. model output vs. user "what-if" inputs).
* **No individual-athlete modeling.** The system rates teams, not players, and
  is not designed to inform decisions about specific athletes.
* **Not a betting product.** Gambling is out of scope and discouraged in-app.
* **Bias / fairness.** Ratings can lag for programs with volatile rosters or
  thin histories; preseason priors are conservative and explicitly flagged, and
  Group-of-Five access is modeled via the configurable guaranteed-champion rule.
* **Transparency.** Every prediction is explained; every selection weight is in
  a config file; the model card and evaluation are shipped with the product.

## Maintenance

Retrained automatically by `scripts/update_dashboard.py` after new results are
ingested. Each run rewrites `models/game_model.joblib` and `model_card.json` and
reruns the season simulation.
