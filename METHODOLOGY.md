# Methodology — for football executives

This document explains **how the dashboard reaches its conclusions**, in plain
language. No statistics background is assumed. Precise field definitions are in
[DATA_DICTIONARY.md](DATA_DICTIONARY.md); model provenance is in
[MODEL_CARD.md](MODEL_CARD.md).

---

## 1. The one-paragraph version

We rate every FBS team from its results and how it produced them (efficiency on
early downs, explosive plays, protecting the quarterback, finishing drives,
special teams, and so on), **adjusted for the quality of its opponents**. For
each game we compare the two teams' ratings, add home field and rest, and
produce a **win probability** and a **projected score**. We then play the rest of
the season **10,000 times** on a computer to estimate each team's chances of
winning its division/conference, making the 12-team Playoff, and winning the
national title. Every prediction comes with a plain-English explanation and an
honest statement of uncertainty.

---

## 2. Where the numbers come from (data)

* **Preferred:** the CollegeFootballData API (schedules, results, conference
  alignment, advanced team statistics).
* **Fallback:** public ESPN endpoints (schedules and scores only).
* **Offline:** local CSV/Parquet files you provide.
* **Demo:** a realistic *synthetic* five-season universe so the product runs with
  no credentials. Demo numbers are illustrative, not real.

Sources are tried in priority order and the first valid one wins. If a source
fails, the last good dataset keeps serving and the page shows a **Delayed** or
**Stale** badge. We never blend conflicting values silently — we log the
discrepancy and keep the higher-priority source.

**The golden rule:** for a game that kicks off at a given time, the model may
only use information that existed *before* kickoff. A team's season statistics
are always "as of the morning of the game" — the game being predicted is never
included in its own inputs. This is enforced in code and covered by tests.

---

## 3. How teams are rated

Two complementary ratings:

1. **Elo** — the classic chess-style rating. A team gains points for winning
   (more for beating a strong team, more for a convincing margin) and loses
   points for losing. Ratings carry over between seasons but are pulled part-way
   back toward average, because rosters turn over.
2. **Opponent-adjusted efficiency** — how much better than average a team has
   been on a per-play basis, on offense and defense, once you account for who it
   played. This separates "good team" from "easy schedule."

The gradient-boosted component of the model also looks at **matchup-specific**
edges: your explosive passing offense matters more against a defense that gives
up big plays; your pass rush matters more against a team that can't protect. A
simple ratings gap can't see those; the tree model can — and it is the part we
use to generate the written explanations.

**Noise control.** Some statistics are mostly luck over a few games —
turnover margin is the worst offender. The model deliberately discounts these
and pulls them toward the average, so one lucky or unlucky afternoon doesn't
dominate a forecast.

---

## 4. Predicting a single game

The model produces three things:

* **Win probability** for the home team (and therefore the away team).
* **Projected margin** and **projected total**, which combine into a
  **projected final score**.
* An **explanation**: the five factors most helping the favorite, the three
  giving the underdog a chance, a confidence level, and a plausible score range.

**Calibration matters more than "accuracy."** We tune the model so that when it
says 70%, the favorite really does win about 70 times in 100. A model that is
right 75% of the time but whose "90%" games only win 70% of the time is worse
for decision-making than a slightly less accurate model whose numbers you can
trust. The Model Evaluation page shows a calibration chart for exactly this
reason.

---

## 5. Simulating the season and the Playoff

We run the remaining schedule **10,000 times**. In each simulated season:

1. every unplayed game is decided by a coin weighted to its win probability;
2. conference standings are computed and championship games are played;
3. every team gets a **selection score** built from: record, opponent-adjusted
   efficiency, strength of schedule, quality wins, bad losses, a
   conference-title bonus, conference strength, and a committee-style prior;
4. the 12-team field is chosen — the five highest-ranked conference champions
   are guaranteed a spot — and seeded, with first-round byes to the top four;
5. the bracket is played out.

Counting across all 10,000 runs gives each team's probability of winning its
conference, making the Playoff, earning a bye, reaching the semifinal and final,
and winning the title.

**Important honesty note.** The real selection committee's judgement can't be
reproduced exactly. Our selection score is a *transparent, tunable proxy* — all
of its weights live in `config/playoff.yaml` and can be changed without touching
code. Every Playoff and championship number in the dashboard is labelled an
**estimate** for this reason.

---

## 6. How we know it works (and where it doesn't)

* We **never** judge the model on games it trained on. Results are reported on
  seasons held out of training entirely, split **by time** (train on earlier
  seasons, test on later ones) — never by randomly shuffling games.
* We always compare against **simple baselines** (home-team win rate, Elo alone,
  a plain logistic regression). On real college-football data these baselines
  are genuinely hard to beat — Elo, logistic regression and the gradient-boosted
  model all land within about two log-loss points of each other. That is *why*
  the shipped model is an **average of all three**: it never trails the best one
  and is the best-calibrated of the group. If you only trust a single method,
  the Evaluation page shows each on its own so you can see for yourself.
* The dashboard's **Model Evaluation** page shows accuracy, log loss, Brier
  score, calibration error, score-error bands, and breakdowns by conference,
  location and season, with every model side by side.

**The model is least reliable for:** early-season games (little current data, so
it leans on preseason priors), teams with little history, and games swung by
things it can't see (injuries, weather, suspensions). Use the Matchup Simulator's
manual adjustments for those.

---

## 7. Reading probabilities like a professional

* A **70% favorite that loses is not a model failure** — it's the 30% showing up.
  Judge the model over dozens of games, not one.
* Probabilities **move** as games are played, especially early, when one result
  is a big share of the evidence.
* "The model sees a closer game than the ranking suggests" is a *lean*, not a
  lock. The dashboard always shows the range of plausible outcomes next to the
  point prediction.
