# Data Dictionary

Every field the dashboard uses, in both technical and football language. The
model-feature rows are generated from the same source that drives the in-app
**Methodology & Data → Data dictionary** table (`src/features/dictionary.py`),
so this file and the app never drift apart.

Conventions:

* All **diff** features are **home minus away**, so a **positive value always
  favors the home team**.
* Defensive stats are stored as **"prevention"** (higher = better defense) so
  every feature reads "higher = better for that team."
* **Cross** features pit one team's offense against the other's defense.
* `game_id`, `season`, `week`, `date`, teams, conferences, `neutral_site`,
  `completed` and scores form the **canonical game schema** that every data
  source adapter must emit.

## Storage

| Store | Location | Notes |
|---|---|---|
| Games + teams | `data/processed/games.parquet`, `teams.parquet` | idempotent upsert; keyed on `game_id` |
| Refresh log | `data/processed/refresh_log.json` | timestamp, source, status, row counts per attempt |
| Feature matrix | `data/processed/features.parquet` | rebuilt by `scripts/build_dataset.py` |
| Dashboard artifacts | `data/processed/dashboard/` | `team_probabilities`, `game_predictions`, `leverage`, `summary.json` |
| Model bundle | `models/game_model.joblib` + `models/model_card.json` | predictor + Elo + provenance |

## Fields

| Field | Kind | Technical definition | In football terms |
|---|---|---|---|
| `game_id` | raw | Stable unique game identifier. | One row per game. |
| `season` | raw | Season year (Aug–Jul). | Which year's season the game belongs to. |
| `week` | raw | Scheduling week number. | Week of the season. |
| `date` | raw | Kickoff timestamp, UTC. | When the game starts. |
| `home_team / away_team` | raw | Canonical team names. | Who is playing. |
| `home_conference / away_conference` | raw | Conference affiliation for the season. | What league each team is in. |
| `neutral_site` | raw | Boolean; True if neither team is hosting. | Whether the game is at a neutral venue (no home edge). |
| `completed` | raw | Boolean; game is final. | Has the game been played. |
| `home_points / away_points` | raw | Final score by team (null until final). | The score. |
| `off_epa_per_play` | raw | Expected points added per offensive play. | How much each snap improves a team's scoring position. |
| `def_epa_per_play` | raw | EPA allowed per play (lower is better). | How much a defense limits the opponent each snap. |
| `success_rate` | raw | Share of plays gaining ≥50%/70%/100% of yards to go on 1st/2nd/3rd+ down. | How often a team stays on schedule. |
| `explosiveness` | raw | Average EPA/yardage value of successful plays. | The big-play dimension of an offense. |
| `havoc` | raw | Share of plays with a TFL, forced fumble, INT or pass breakup. | How often a defense blows up a play. |
| `sack_rate` | raw | Sacks per pass attempt (offense: allowed; defense: generated). | Pass protection / pass rush. |
| `finish_pts_per_opp` | raw | Points per trip inside the opponent 40 / scoring opportunity. | Red-zone / scoring-chance finishing. |
| `line_yards` | raw | Opportunity-adjusted rushing yards credited to the line. | How well the offensive line is winning up front. |
| `st_epa` | raw | Special-teams EPA (kicking, punting, returns). | The hidden-yardage / special teams edge. |
| `pace_sec_per_play` | raw | Seconds of game clock per offensive play. | How fast a team plays (context, not quality). |
| `turnovers / takeaways` | raw | Giveaways and takeaways per game. | Ball security and ball-hawking (very noisy). |
| `elo_diff` | model feature | Overall team rating (Elo). Elo rating gap, including home-field adjustment. ~25 points ≈ 1 point on the scoreboard. | Team A carries the stronger overall rating built from results and margins |
| `adj_rating_diff` | model feature | Opponent-adjusted scoring margin. Season scoring margin per game adjusted for the quality of opponents faced. | Team A has outscored opponents by more once schedule strength is accounted for |
| `recent_form_diff` | model feature | Recent form. Exponentially weighted scoring margin — recent games count more than early ones. | Team A has been playing better over its last few games |
| `season_margin_diff` | model feature | Season-long scoring margin. Average points scored minus points allowed, all season. | Team A has the better points margin across the whole season |
| `win_pct_diff` | model feature | Winning percentage. Games won divided by games played entering this week. | Team A has simply won a higher share of its games |
| `sos_diff` | model feature | Strength of schedule faced. Average opponent rating for games played so far. | Team A has already beaten a tougher slate, which the model rewards |
| `turnover_margin_diff` | model feature | Turnover margin (regressed). Takeaways minus giveaways. Turnovers are noisy, so the model discounts this heavily. | Team A has protected the ball and taken it away more often |
| `off_epa_diff` | model feature | Offensive efficiency (EPA/play). Expected points added per offensive play — how much each snap improves scoring position. | Team A's offense has been more efficient on a per-play basis |
| `def_epa_prevention_diff` | model feature | Defensive efficiency (EPA/play allowed). Expected points added allowed per play — lower is better; shown so higher = better defense. | Team A's defense has given up less on a per-play basis |
| `success_rate_diff` | model feature | Offensive success rate. Share of plays that gain enough yardage to stay ahead of the chains. | Team A stays on schedule more often (more 'successful' plays) |
| `def_success_prevention_diff` | model feature | Defensive success rate allowed. Opponent success rate allowed; shown so higher = better defense. | Team A's defense forces more off-schedule downs |
| `explosiveness_diff` | model feature | Explosive-play offense. Average yardage value of successful plays — the big-play dimension. | Team A generates more chunk plays that flip field position |
| `def_explosiveness_prevention_diff` | model feature | Explosive-play defense. Explosiveness allowed; shown so higher = better defense. | Team A gives up fewer chunk plays |
| `pass_epa_diff` | model feature | Passing-game efficiency. Expected points added per dropback. | Team A has been more productive throwing the ball |
| `rush_epa_diff` | model feature | Rushing-game efficiency. Expected points added per rush. | Team A has been more productive running the ball |
| `havoc_diff` | model feature | Defensive havoc. Share of plays with a tackle for loss, pass broken up, or forced fumble. | Team A's defense disrupts more plays (TFLs, PBUs, forced fumbles) |
| `pass_protection_diff` | model feature | Pass protection. Sack rate allowed; shown so higher = better protection. | Team A keeps its quarterback cleaner |
| `pass_rush_diff` | model feature | Pass rush. Sacks generated per opponent dropback. | Team A gets after the quarterback more effectively |
| `red_zone_diff` | model feature | Red-zone finishing. Points per trip inside the opponent 40 / scoring opportunity. | Team A turns more scoring chances into touchdowns |
| `red_zone_defense_diff` | model feature | Red-zone defense. Points allowed per opponent scoring opportunity; shown so higher = better. | Team A holds opponents to fewer points per scoring chance |
| `line_yards_diff` | model feature | Offensive line push (run game). Line yards — the portion of rushing yards credited to blocking. | Team A's offensive line is winning more of the line-of-scrimmage |
| `special_teams_diff` | model feature | Special teams. Special-teams expected points added (kicking, punting, returns). | Team A has the edge in the kicking and return game |
| `home_off_vs_away_def` | model feature | Home offense vs. visiting defense. Cross-matchup: home offensive efficiency against away defensive efficiency. | Team A's offense matches up well with Team B's defense |
| `away_off_vs_home_def` | model feature | Visiting offense vs. home defense. Cross-matchup: away offensive efficiency against home defensive efficiency. | Team A's defense matches up well with Team B's offense |
| `oline_matchup` | model feature | Line-of-scrimmage battle. Combined run-blocking vs. run-defense edge. | Team A projects to control the line of scrimmage |
| `home_indicator` | model feature | Home-field advantage. Whether the home team is actually hosting (0 for neutral-site games). | Team A is playing at home |
| `is_neutral` | model feature | Neutral site. Flag for games played at a neutral venue. | the neutral site removes Team B's road disadvantage |
| `rest_days_diff` | model feature | Rest advantage. Home team's rest days minus away team's rest days. | Team A has had more days off since its last game |
| `home_games_played` | model feature | Home team sample size. Completed games for the home team — low values lean on preseason priors. | the model has more data on Team A this season |
| `away_games_played` | model feature | Away team sample size. Completed games for the away team — low values lean on preseason priors. | the model has limited data on Team B this season |

## Advanced-metric availability

The pipeline runs **without** any of the advanced metrics above. When a data
source does not provide them, the affected rolling features fall back to values
derived from final scores (scoring margin, opponent-adjusted margin, recent
form, win %), and the dashboard still produces predictions, explanations and
simulations. Missing-data notes appear on the affected game breakdowns.
