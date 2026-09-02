# Known limitations & recommended next improvements

## Current limitations

### Data
1. **No injury / availability data.** A star QB being out is invisible to the
   base model. Mitigation: manual rating adjustments in the Matchup Simulator.
2. **No weather data.** Wind and precipitation meaningfully affect passing and
   kicking. Not modeled.
3. **No betting lines.** Without a licensed odds feed, "accuracy vs. the closing
   favorite" uses the Elo favorite as a proxy rather than a real market line.
4. **No travel/altitude/short-week fatigue** beyond a simple rest-days
   difference. No stadium coordinates in the base build.
5. **Advanced metrics depend on the source.** ESPN fallback provides scores
   only; the model then runs on results-derived features and is correspondingly
   blunter.
6. **Backtest depth.** The demo ships three synthetic training seasons. Real
   multi-season validation requires connecting CFBD history.

### Modeling
7. **Opponent adjustment is single-pass**, not a full simultaneous solve. Good
   enough for ranking; a ridge/SRS-style iterative solve would be tighter.
8. **Preseason priors are coarse** — previous-season final rating regressed to
   the mean, plus a tier constant for new teams. No returning-production or
   recruiting/transfer-portal signal yet.
9. **Score model is margin + total**, assuming roughly symmetric noise. A
   bivariate or quantile model would give better score *distributions*.
10. **The playoff selection model is a proxy.** It cannot capture committee
    narrative, eye-test or politics. Weights are configurable but still a guess.
11. **Bracket re-seeding** is by selection score only; the real "top-4 seeds to
    the four highest-ranked conference champions" wrinkle is simplified.
12. **Live win probability is not implemented** — pregame/season only.

### Product
13. **No authentication / admin area** — the "manual refresh" is a cache-clear
    button, not a gated control.
14. **Team logos** are not bundled (licensing). The UI uses names and conference
    accent colors.
15. **Single-node, file-based.** No database, no multi-user state, no horizontal
    scaling. Fine for a briefing tool; not a public high-traffic service as-is.

## Done since first draft

- ~~Connect live CFBD data end-to-end~~ — done (2017–2026, FBS-vs-FBS, advanced
  stats where available); ensemble win model; real backtest on the Evaluation page.
- ~~Dockerfile + hosting path~~ — done (`Dockerfile`, `.github/workflows/deploy-hf-space.yml`,
  `docs/DEPLOY.md`; Hugging Face Docker Space, Option A).

## Recommended next improvements (roughly in priority order)

1. **Incremental CFBD fetch** — each refresh currently re-pulls every configured
   season (~3 min, ~140 advanced-stat calls). After the initial backfill, only
   fetch the current season's unplayed/just-played weeks.
2. **Injury / availability ingestion** (CFBD or a licensed feed) with automatic
   rating deltas, replacing the manual slider for known absences.
3. **Returning production & transfer-portal priors** for a much better Week 0–4
   model; blend down as the season progresses (hooks already exist).
4. **Betting-line ingestion** (where licensed) for a genuine market benchmark
   and a "model vs. market" edge view.
5. **Iterative opponent-adjusted efficiency** (SRS / ridge) replacing the
   single-pass adjustment.
6. **Quantile or distributional score model** so the "plausible score range" is
   learned rather than a fixed band.
7. **Weather features** for outdoor venues (wind especially).
8. **Richer playoff logic**: exact seeding rules, conference tiebreakers from
   `config`, and a committee-ranking calibration against historical top-25s.
9. **Live-game mode** behind a play-by-play provider, with its own validation
   and a clear "live" vs "pregame" split — only enabled once validated.
10. **Persistence + scheduler** (Postgres + a real cron/cloud scheduler) and a
    small admin page with an audited manual-refresh control.
11. **Team logos & branding** once licensing is sorted.
12. **Model monitoring**: track calibration drift week-to-week and alert when it
    degrades.
