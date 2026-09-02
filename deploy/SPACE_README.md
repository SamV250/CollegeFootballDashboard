---
title: College Football Dashboard
emoji: 🏈
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: ML predictions and playoff odds for the 2026 FBS season
---

# College Football Prediction Dashboard

Predicts every FBS game (calibrated win probability + score), estimates each
team's conference / playoff / national-title odds via a 10,000-iteration Monte
Carlo simulation, and explains every prediction in plain football language.

This Space is **read-only**: it serves data and model artifacts that are
refreshed on a schedule by a GitHub Action and pushed here (see
[`docs/DEPLOY.md`](https://github.com/SamV250/CollegeFootballDashboard/blob/main/docs/DEPLOY.md)).
The freshness badge on every page reports how current the data is.

Source, methodology and model card:
**https://github.com/SamV250/CollegeFootballDashboard**
