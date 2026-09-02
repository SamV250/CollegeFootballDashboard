# Dashboard walkthrough

A page-by-page tour. To capture your own screenshots, run the app
(`streamlit run app.py`) and use your browser or OS screenshot tool; drop images
in `docs/screenshots/` and they will render below.

The app has a **global sidebar** on every page:

* **Explanation depth** — *Executive* (brief plain-language conclusions) or
  *Analyst* (adds feature values, SHAP detail, diagnostics).
* **Time zone** — all timestamps are stored in UTC and displayed here.
* **Model panel** — version, training date, backend, calibration method.
* **↻ Reload data & caches** — manual refresh / cache clear.

---

## 1. Executive Overview  (`app.py`)

*The landing page. Conclusions first, evidence one click away.*

* **Freshness strip** — data status (Current / Delayed / Stale), when data was
  last updated, the latest completed game included, and the next scheduled
  refresh.
* **Four headline cards** — most likely national champion, safest playoff bet,
  games left to simulate (with iteration count), and the top team by model
  rating.
* **National championship probability** and **Playoff probability (top 16)** bar
  charts.
* **Model top 25** — rank, record, model rating, and conference-title / playoff /
  national-title probabilities.
* **Signals this week** — teams the model rates very differently from a
  ratings-only (Elo) view, and the highest-leverage upcoming games.
* Footer: prediction-generation timestamp, model version, latest completed game.

Every playoff/championship figure is explicitly labelled an **estimate**.

![Executive Overview](screenshots/01_executive_overview.png)

---

## 2. Weekly Matchups  (`pages/1_Weekly_Matchups.py`)

*Every upcoming game, predicted and explained.*

* **Filters** — week, conference, location (home/road favorite/neutral), minimum
  upset probability, minimum combined playoff impact, ranked-teams-only.
* **Summary table** — pick, win probability, predicted score, upset %, playoff
  impact, kickoff (in your time zone).
* **Game breakdowns** — expandable per game:
  * plain-language summary and a win-probability bar;
  * predicted winner / score / confidence / upset probability;
  * plausible final-margin band;
  * **"Why the model favors [team]"** — up to five factors in football language
    with magnitude labels (major / moderate / slight advantage / essentially
    even);
  * **"What gives [underdog] a chance"** — up to three counter-factors;
  * missing-data warnings; a note if the projection moved a lot in the past week;
  * in Analyst mode, a SHAP contribution table with definitions.

![Weekly Matchups](screenshots/02_weekly_matchups.png)

---

## 3. Team Profile  (`pages/2_Team_Profile.py`)

*Everything about one program.*

* Headline cards: record, model rating (with rank and Elite/Above-Average/…
  tier), playoff probability, national-title probability; plus conference-title
  game, conference champion, first-round bye, reach-title-game.
* **Percentile bars vs. all of FBS** — overall, offense, defense, special teams,
  each with a football tier label.
* **Schedule & game-by-game win probability** — a bar per game (grey = already
  played) with the opponent labelled, plus a table with site, result, win
  probability and projected score.
* **Strengths, vulnerabilities & swing games** — auto-generated bullets, best
  win, worst loss, and the most pivotal remaining game.
* **Trajectory** — pre-game win probability across the season so far.

![Team Profile](screenshots/03_team_profile.png)

---

## 4. Matchup Simulator  (`pages/3_Matchup_Simulator.py`)

*Any two FBS teams, plus what-if.*

* Pick Team A / Team B and the site (Team A home or neutral).
* **Scenario adjustments** (collapsible): per-team rating adjustment (points),
  offensive and defensive efficiency nudges, a turnover-margin scenario, and a
  pace shift. All optional; zero = pure model.
* Output: win-probability bar, predicted winner / score / expected margin /
  confidence, plausible margin band, key advantages for the favorite, the
  underdog's upset path, and a **side-by-side comparison table**.
* When you apply adjustments, the page shows the **model baseline and your
  adjusted scenario side by side** — model output and user assumptions never
  blur.

![Matchup Simulator](screenshots/04_matchup_simulator.png)

---

## 5. Playoff Simulator  (`pages/4_Playoff_Simulator.py`)

*Force results, watch the bracket move.*

* **Set results** — pick any upcoming games and set each to Team A, Team B, or
  "leave to model."
* Choose iteration count and **Run scenario** (re-runs the Monte Carlo season).
* **Projected playoff field** — seed, team, conference, bid type (auto conf
  champ vs at-large), bye vs hosts-first-round, and each seed's playoff/title %.
* **Biggest swings vs. baseline** — the teams whose playoff or title
  probability moved most because of your picks.
* **Games with the most leverage** — the upcoming results that would swing the
  field the most, from the baseline simulation.

![Playoff Simulator](screenshots/05_playoff_simulator.png)

---

## 6. Model Evaluation  (`pages/5_Model_Evaluation.py`)

*How good is it, honestly?*

* Tabs for each held-out split (validation / test).
* **Win model vs. baselines** table — accuracy, log loss, Brier, ROC AUC,
  calibration error, and accuracy vs. the (Elo-proxy) favorite, for the
  gradient-boosted model, Elo, logistic regression and home-win-rate.
* **Calibration plot** (predicted vs. actual) and **confidence-vs-reality** bars.
* **Score prediction** — MAE (margin and team score), RMSE, and the share of
  games whose margin was predicted within 3 / 7 / 10 / 14 points.
* **By conference** and **by game location** breakdowns.
* **Per-season backtest** chart and table.
* A "How to read this" section explaining why probabilities move and why a 70%
  pick still loses ~30% of the time.

![Model Evaluation](screenshots/06_model_evaluation.png)

---

## 7. Methodology & Data  (`pages/6_Methodology_and_Data.py`)

*Sources, process, definitions, limitations, model card.*

* Live data-status strip.
* Source priority and a table of what each source provides and when it is used.
* The 10-step refresh process and the cadence (6 h in-season / 15 min game days
  / after finals), with pointers to the GitHub Actions and cron examples.
* Modeling and simulation methodology in brief.
* The full **data dictionary** (same source as `DATA_DICTIONARY.md`).
* Known limitations and the complete **model card** JSON.

![Methodology & Data](screenshots/07_methodology.png)
