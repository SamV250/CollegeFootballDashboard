# College Football Prediction Dashboard — 2026–27 FBS season

An executive-friendly machine-learning dashboard that predicts every FBS game,
estimates each team's playoff and championship odds, and **explains every
prediction in plain football language**. It runs out of the box with **no API
keys** on a bundled synthetic dataset, and switches to live data when a
CollegeFootballData key is provided.

> ⚠️ Predictions are probabilistic estimates, not guarantees. A 70% pick is
> supposed to lose about 30% of the time. Playoff/championship numbers are
> estimates from a transparent, configurable selection model — not a forecast of
> the selection committee.

---

## What's in the box

| Page | Answers |
|---|---|
| **Executive Overview** | Who's most likely to win it all? Who's rising/falling? Which games matter most? Where does the model disagree with consensus? |
| **Weekly Matchups** | Predicted winner, score, confidence, upset odds and key factors for every upcoming game, with rich filters. |
| **Team Profile** | Rating, Elo, unit percentiles vs. FBS, remaining schedule with game-by-game win probability, strengths/vulnerabilities, best win / worst loss. |
| **Matchup Simulator** | Any two FBS teams, neutral/home, plus what-if adjustments (injuries, tempo, turnovers, efficiency). Model output and your assumptions are shown separately. |
| **Playoff Simulator** | Force game results, re-run 10k Monte Carlo seasons, see the projected bracket and every team's probability swing. Includes a leverage table. |
| **Model Evaluation** | Out-of-sample accuracy, log loss, Brier, calibration, score-error bands, per-conference / per-location / per-season breakdowns, and a baseline comparison. |
| **Methodology & Data** | Sources, refresh process, data dictionary, simulation methodology, known limitations, full model card. |

---

## Quick start (demo mode, no credentials)

Requires **Python 3.11+**. On macOS also `brew install libomp` (LightGBM/XGBoost
need the OpenMP runtime).

```bash
cd cfb-dashboard
python -m venv .venv && source .venv/bin/activate      # or use an existing env
pip install -r requirements.txt

# Build everything from the synthetic dataset (≈ 1–2 minutes):
python scripts/update_data.py --source demo   # 1. generate + store demo data
python scripts/train_models.py                # 2. train Elo + baselines + ensemble
python scripts/run_simulation.py --iterations 10000   # 3. season Monte Carlo

# Launch the dashboard:
streamlit run app.py
```

Then open http://localhost:8501.

If you skip steps 1–3, the app will build a demo dataset and a quick model/
simulation on first load automatically — the explicit commands just make the
first page load instant.

`make` shortcuts are available: `make setup`, `make update`, `make app`,
`make test`, `make lint`.

---

## Live data mode

1. Get a free key at <https://collegefootballdata.com/key>.
2. `cp .env.example .env` and set `CFBD_API_KEY=...`.
3. Run the full pipeline:

```bash
python scripts/update_dashboard.py --iterations 10000
```

The data layer tries sources in the order set by
`config/config.yaml → data.source_priority` (default: `cfbd → espn → local →
demo`) and uses the first that returns valid, non-empty data. A failing source
never erases previously validated data.

**CFBD scope:** the adapter pulls **FBS-vs-FBS games only** (matching how
efficiency systems like SP+/FEI are built) plus per-game advanced stats where
available (PPA/EPA, success rate, explosiveness, line yards, on offense and
defense). A few "cupcake" wins over FCS opponents are therefore not reflected in
records. Historical seasons pulled are set by `config → season.backtest_seasons`.

---

## Keeping it current (automation)

`python scripts/update_dashboard.py` is **idempotent** and does the whole
pipeline: fetch → validate → diff → upsert only new/changed rows → rebuild
leakage-safe features → update Elo & ratings → retrain → predict → Monte Carlo
simulation → save artifacts → log.

* **GitHub Actions** — `.github/workflows/refresh.yml` (every 6 h; uncomment the
  `*/15` schedule for game days). Set `CFBD_API_KEY` as a repo secret.
* **Cron** — `scripts/crontab.example`.
* **Manual** — the sidebar's "↻ Reload data & caches" button, or run the script.

Freshness is shown on every page: **Current** (refreshed within the expected
interval), **Delayed** (one refresh missed), **Stale** (> 24 h). Cached/delayed
data is never labelled "live".

---

## Deploy (Hugging Face Spaces)

A `Dockerfile` (host-agnostic, listens on `$PORT`) and a deploy workflow
(`.github/workflows/deploy-hf-space.yml`) are included. The pattern: a scheduled
Action refreshes data + model + simulation and force-pushes the tree (code +
baked-in artifacts) to a Hugging Face **Docker** Space, which rebuilds and
serves. The Space holds no API key.

Set two GitHub secrets (`HF_TOKEN`, `CFBD_API_KEY`), point the workflow's
`HF_USERNAME` / `HF_SPACE` at your Space, and run it. Full walkthrough — plus
notes for Streamlit Community Cloud, Railway/Render/Fly, and plain Docker — in
**[docs/DEPLOY.md](docs/DEPLOY.md)**.

```bash
docker build -t cfb-dashboard . && docker run -p 7860:7860 cfb-dashboard
```

---

## Project layout

```
app.py                     Streamlit entry (Executive Overview)
pages/                     the other six dashboard pages
config/
  config.yaml              seasons, data, Elo, model, features, simulation
  conferences.yaml         2026 FBS alignment (offline fallback)
  playoff.yaml             playoff format + committee-style selection weights
src/
  config.py               single settings loader
  data/                   source adapters, idempotent store, loader, demo generator
  features/               leakage-safe rolling stats, matchup feature builder, dictionary
  models/                 Elo, baselines, calibrated GBM, the 3-way ensemble, training, evaluation, registry
  simulation/             Monte Carlo season engine, playoff selection, leverage
  explainability/         SHAP → football-language narratives
  ui/                     theme, components, cached state, matchup renderer
scripts/                  update_data, build_dataset, train_models, evaluate_models,
                          run_simulation, update_dashboard
tests/                     leakage, data integrity, priors, models, simulation, explainability
docs/                      walkthrough + screenshots
```

## Reproducibility

* All randomness is seeded (`config.yaml → model.random_seed`,
  `simulation.random_seed`, and the demo generator).
* Trained models are saved to `models/game_model.joblib` with a companion
  `models/model_card.json` recording version, training date, training seasons,
  feature list, evaluation results and data-source/config revision.
* Streamlit caches expensive operations; the sidebar button clears them.

## Tests & linting

```bash
python -m pytest      # 31 tests: leakage guards, data integrity, priors, models, sim, explanations
ruff check .
```

The leakage suite specifically verifies that a game's own box score cannot
change its own feature row, that future games cannot change past feature rows,
that rolling stats are `shift`-ed, that neutral sites carry no home-field term,
that team↔conference mappings are consistent, that missing data does not crash
the pipeline, and that new / low-history teams get sane preseason priors.

## Documentation

* **[METHODOLOGY.md](METHODOLOGY.md)** — written for football executives.
* **[DATA_DICTIONARY.md](DATA_DICTIONARY.md)** — every field, technical + football.
* **[MODEL_CARD.md](MODEL_CARD.md)** — intended use, validation, limitations, ethics.
* **[docs/walkthrough.md](docs/walkthrough.md)** — page-by-page tour with screenshots.
* **[docs/LIMITATIONS.md](docs/LIMITATIONS.md)** — known gaps and recommended next steps.

## License

MIT.
