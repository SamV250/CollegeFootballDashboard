"""Translate model features into plain football language.

Every model feature has an entry here with:

* ``label``      -- short human name
* ``favors_home_when_positive`` -- direction convention (all our diff
  features favour the home team when positive)
* ``home_phrase`` / ``away_phrase`` -- how to describe the factor when it
  helps the home or the away team, written for a coach or a broadcaster
* ``tooltip``    -- one-sentence definition of the underlying stat

Magnitude buckets convert a standardized contribution into words:
"major advantage", "moderate advantage", "slight advantage",
"essentially even".  Raw SHAP numbers are never shown on their own.
"""

from __future__ import annotations

from dataclasses import dataclass

MAGNITUDE_BUCKETS = [
    (0.60, "major advantage"),
    (0.30, "moderate advantage"),
    (0.12, "slight advantage"),
    (0.00, "essentially even"),
]


@dataclass(frozen=True)
class FeatureNarrative:
    label: str
    home_phrase: str
    away_phrase: str
    tooltip: str


# key -> narrative.  ``{home}`` / ``{away}`` are filled with team names.
FEATURE_NARRATIVES: dict[str, FeatureNarrative] = {
    "elo_diff": FeatureNarrative(
        "Overall team rating (Elo)",
        "{home} carries the stronger overall rating built from results and margins",
        "{away} carries the stronger overall rating built from results and margins",
        "Elo rating gap, including home-field adjustment. ~25 points ≈ 1 point on the scoreboard.",
    ),
    "adj_rating_diff": FeatureNarrative(
        "Opponent-adjusted scoring margin",
        "{home} has outscored opponents by more once schedule strength is accounted for",
        "{away} has outscored opponents by more once schedule strength is accounted for",
        "Season scoring margin per game adjusted for the quality of opponents faced.",
    ),
    "recent_form_diff": FeatureNarrative(
        "Recent form",
        "{home} has been playing better over its last few games",
        "{away} has been playing better over its last few games",
        "Exponentially weighted scoring margin — recent games count more than early ones.",
    ),
    "season_margin_diff": FeatureNarrative(
        "Season-long scoring margin",
        "{home} has the better points margin across the whole season",
        "{away} has the better points margin across the whole season",
        "Average points scored minus points allowed, all season.",
    ),
    "win_pct_diff": FeatureNarrative(
        "Winning percentage",
        "{home} has simply won a higher share of its games",
        "{away} has simply won a higher share of its games",
        "Games won divided by games played entering this week.",
    ),
    "sos_diff": FeatureNarrative(
        "Strength of schedule faced",
        "{home} has already beaten a tougher slate, which the model rewards",
        "{away} has already beaten a tougher slate, which the model rewards",
        "Average opponent rating for games played so far.",
    ),
    "turnover_margin_diff": FeatureNarrative(
        "Turnover margin (regressed)",
        "{home} has protected the ball and taken it away more often",
        "{away} has protected the ball and taken it away more often",
        "Takeaways minus giveaways. Turnovers are noisy, so the model discounts this heavily.",
    ),
    "off_epa_diff": FeatureNarrative(
        "Offensive efficiency (EPA/play)",
        "{home}'s offense has been more efficient on a per-play basis",
        "{away}'s offense has been more efficient on a per-play basis",
        "Expected points added per offensive play — how much each snap improves scoring position.",
    ),
    "def_epa_prevention_diff": FeatureNarrative(
        "Defensive efficiency (EPA/play allowed)",
        "{home}'s defense has given up less on a per-play basis",
        "{away}'s defense has given up less on a per-play basis",
        "Expected points added allowed per play — lower is better; shown so higher = better defense.",
    ),
    "success_rate_diff": FeatureNarrative(
        "Offensive success rate",
        "{home} stays on schedule more often (more 'successful' plays)",
        "{away} stays on schedule more often (more 'successful' plays)",
        "Share of plays that gain enough yardage to stay ahead of the chains.",
    ),
    "def_success_prevention_diff": FeatureNarrative(
        "Defensive success rate allowed",
        "{home}'s defense forces more off-schedule downs",
        "{away}'s defense forces more off-schedule downs",
        "Opponent success rate allowed; shown so higher = better defense.",
    ),
    "explosiveness_diff": FeatureNarrative(
        "Explosive-play offense",
        "{home} generates more chunk plays that flip field position",
        "{away} generates more chunk plays that flip field position",
        "Average yardage value of successful plays — the big-play dimension.",
    ),
    "def_explosiveness_prevention_diff": FeatureNarrative(
        "Explosive-play defense",
        "{home} gives up fewer chunk plays",
        "{away} gives up fewer chunk plays",
        "Explosiveness allowed; shown so higher = better defense.",
    ),
    "pass_epa_diff": FeatureNarrative(
        "Passing-game efficiency",
        "{home} has been more productive throwing the ball",
        "{away} has been more productive throwing the ball",
        "Expected points added per dropback.",
    ),
    "rush_epa_diff": FeatureNarrative(
        "Rushing-game efficiency",
        "{home} has been more productive running the ball",
        "{away} has been more productive running the ball",
        "Expected points added per rush.",
    ),
    "havoc_diff": FeatureNarrative(
        "Defensive havoc",
        "{home}'s defense disrupts more plays (TFLs, PBUs, forced fumbles)",
        "{away}'s defense disrupts more plays (TFLs, PBUs, forced fumbles)",
        "Share of plays with a tackle for loss, pass broken up, or forced fumble.",
    ),
    "pass_protection_diff": FeatureNarrative(
        "Pass protection",
        "{home} keeps its quarterback cleaner",
        "{away} keeps its quarterback cleaner",
        "Sack rate allowed; shown so higher = better protection.",
    ),
    "pass_rush_diff": FeatureNarrative(
        "Pass rush",
        "{home} gets after the quarterback more effectively",
        "{away} gets after the quarterback more effectively",
        "Sacks generated per opponent dropback.",
    ),
    "red_zone_diff": FeatureNarrative(
        "Red-zone finishing",
        "{home} turns more scoring chances into touchdowns",
        "{away} turns more scoring chances into touchdowns",
        "Points per trip inside the opponent 40 / scoring opportunity.",
    ),
    "red_zone_defense_diff": FeatureNarrative(
        "Red-zone defense",
        "{home} holds opponents to fewer points per scoring chance",
        "{away} holds opponents to fewer points per scoring chance",
        "Points allowed per opponent scoring opportunity; shown so higher = better.",
    ),
    "line_yards_diff": FeatureNarrative(
        "Offensive line push (run game)",
        "{home}'s offensive line is winning more of the line-of-scrimmage",
        "{away}'s offensive line is winning more of the line-of-scrimmage",
        "Line yards — the portion of rushing yards credited to blocking.",
    ),
    "special_teams_diff": FeatureNarrative(
        "Special teams",
        "{home} has the edge in the kicking and return game",
        "{away} has the edge in the kicking and return game",
        "Special-teams expected points added (kicking, punting, returns).",
    ),
    "home_off_vs_away_def": FeatureNarrative(
        "Home offense vs. visiting defense",
        "{home}'s offense matches up well with {away}'s defense",
        "{away}'s defense matches up well with {home}'s offense",
        "Cross-matchup: home offensive efficiency against away defensive efficiency.",
    ),
    "away_off_vs_home_def": FeatureNarrative(
        "Visiting offense vs. home defense",
        "{home}'s defense matches up well with {away}'s offense",
        "{away}'s offense matches up well with {home}'s defense",
        "Cross-matchup: away offensive efficiency against home defensive efficiency.",
    ),
    "oline_matchup": FeatureNarrative(
        "Line-of-scrimmage battle",
        "{home} projects to control the line of scrimmage",
        "{away} projects to control the line of scrimmage",
        "Combined run-blocking vs. run-defense edge.",
    ),
    "home_indicator": FeatureNarrative(
        "Home-field advantage",
        "{home} is playing at home",
        "this game is on {away}'s side of a home-and-home or at {away}'s place",
        "Whether the home team is actually hosting (0 for neutral-site games).",
    ),
    "is_neutral": FeatureNarrative(
        "Neutral site",
        "the neutral site removes {away}'s road disadvantage",
        "the neutral site removes {home}'s usual home edge",
        "Flag for games played at a neutral venue.",
    ),
    "rest_days_diff": FeatureNarrative(
        "Rest advantage",
        "{home} has had more days off since its last game",
        "{away} has had more days off since its last game",
        "Home team's rest days minus away team's rest days.",
    ),
    "home_games_played": FeatureNarrative(
        "Home team sample size",
        "the model has more data on {home} this season",
        "the model has limited data on {home} this season",
        "Completed games for the home team — low values lean on preseason priors.",
    ),
    "away_games_played": FeatureNarrative(
        "Away team sample size",
        "the model has limited data on {away} this season",
        "the model has more data on {away} this season",
        "Completed games for the away team — low values lean on preseason priors.",
    ),
}


def magnitude_label(z: float) -> str:
    a = abs(z)
    for thresh, label in MAGNITUDE_BUCKETS:
        if a >= thresh:
            return label
    return "essentially even"


def confidence_label(win_prob: float) -> str:
    edge = abs(win_prob - 0.5)
    if edge >= 0.35:
        return "High"
    if edge >= 0.20:
        return "Moderate"
    if edge >= 0.08:
        return "Low"
    return "Toss-up"


def describe_contribution(
    feature: str, contribution: float, scale: float, home: str, away: str
) -> dict:
    """Turn one SHAP contribution into a football sentence + magnitude."""

    nar = FEATURE_NARRATIVES.get(feature)
    if nar is None:
        return {
            "feature": feature,
            "label": feature,
            "favors": "home" if contribution > 0 else "away",
            "magnitude": "slight advantage",
            "sentence": f"{feature} contributed to the prediction.",
            "tooltip": "",
            "z": contribution / scale if scale else 0.0,
        }
    favors_home = contribution > 0
    z = contribution / scale if scale else 0.0
    phrase = nar.home_phrase if favors_home else nar.away_phrase
    sentence = phrase.format(home=home, away=away)
    return {
        "feature": feature,
        "label": nar.label,
        "favors": "home" if favors_home else "away",
        "magnitude": magnitude_label(z),
        "sentence": sentence,
        "tooltip": nar.tooltip,
        "z": float(z),
    }


def plain_language_summary(
    home: str, away: str, win_prob_home: float,
    pred_home_pts: float, pred_away_pts: float,
    top_home_factors: list[dict], top_away_factors: list[dict],
) -> str:
    fav, dog = (home, away) if win_prob_home >= 0.5 else (away, home)
    p = win_prob_home if win_prob_home >= 0.5 else 1 - win_prob_home
    hp, ap = (pred_home_pts, pred_away_pts)
    fav_pts, dog_pts = (hp, ap) if fav == home else (ap, hp)
    lead = (f"The model favours **{fav}** with a **{p*100:.0f}%** win probability "
            f"and a projected **{fav_pts:.0f}–{dog_pts:.0f}** final.")
    if top_home_factors:
        drivers = "; ".join(f["sentence"] for f in top_home_factors[:2])
        lead += f" Chief reasons: {drivers}."
    if top_away_factors:
        keep = top_away_factors[0]["sentence"]
        lead += f" {dog}'s best path: {keep}."
    lead += (f" A {p*100:.0f}% favourite still loses roughly "
             f"{(1-p)*100:.0f} times out of 100 — treat this as a lean, not a lock.")
    return lead


def outcome_range(pred_margin: float, margin_mae: float = 11.0) -> tuple[float, float]:
    """A rough ~80% plausible margin band around the point prediction."""

    half = 1.28 * margin_mae
    return (pred_margin - half, pred_margin + half)
