"""Machine-readable data dictionary.

Combines the model-feature narratives with the raw data fields so the
dashboard and ``DATA_DICTIONARY.md`` can be generated from one source.
"""

from __future__ import annotations

from src.explainability.narrative import FEATURE_NARRATIVES

RAW_FIELDS: list[dict[str, str]] = [
    {"field": "game_id", "type": "raw", "technical": "Stable unique game identifier.",
     "football": "One row per game."},
    {"field": "season", "type": "raw", "technical": "Season year (Aug–Jul).",
     "football": "Which year's season the game belongs to."},
    {"field": "week", "type": "raw", "technical": "Scheduling week number.",
     "football": "Week of the season."},
    {"field": "date", "type": "raw", "technical": "Kickoff timestamp, UTC.",
     "football": "When the game starts."},
    {"field": "home_team / away_team", "type": "raw",
     "technical": "Canonical team names.", "football": "Who is playing."},
    {"field": "home_conference / away_conference", "type": "raw",
     "technical": "Conference affiliation for the season.",
     "football": "What league each team is in."},
    {"field": "neutral_site", "type": "raw",
     "technical": "Boolean; True if neither team is hosting.",
     "football": "Whether the game is at a neutral venue (no home edge)."},
    {"field": "completed", "type": "raw", "technical": "Boolean; game is final.",
     "football": "Has the game been played."},
    {"field": "home_points / away_points", "type": "raw",
     "technical": "Final score by team (null until final).",
     "football": "The score."},
    {"field": "off_epa_per_play", "type": "raw",
     "technical": "Expected points added per offensive play.",
     "football": "How much each snap improves a team's scoring position."},
    {"field": "def_epa_per_play", "type": "raw",
     "technical": "EPA allowed per play (lower is better).",
     "football": "How much a defense limits the opponent each snap."},
    {"field": "success_rate", "type": "raw",
     "technical": "Share of plays gaining ≥50%/70%/100% of yards to go on "
                  "1st/2nd/3rd+ down.", "football": "How often a team stays on schedule."},
    {"field": "explosiveness", "type": "raw",
     "technical": "Average EPA/yardage value of successful plays.",
     "football": "The big-play dimension of an offense."},
    {"field": "havoc", "type": "raw",
     "technical": "Share of plays with a TFL, forced fumble, INT or pass breakup.",
     "football": "How often a defense blows up a play."},
    {"field": "sack_rate", "type": "raw",
     "technical": "Sacks per pass attempt (offense: allowed; defense: generated).",
     "football": "Pass protection / pass rush."},
    {"field": "finish_pts_per_opp", "type": "raw",
     "technical": "Points per trip inside the opponent 40 / scoring opportunity.",
     "football": "Red-zone / scoring-chance finishing."},
    {"field": "line_yards", "type": "raw",
     "technical": "Opportunity-adjusted rushing yards credited to the line.",
     "football": "How well the offensive line is winning up front."},
    {"field": "st_epa", "type": "raw",
     "technical": "Special-teams EPA (kicking, punting, returns).",
     "football": "The hidden-yardage / special teams edge."},
    {"field": "pace_sec_per_play", "type": "raw",
     "technical": "Seconds of game clock per offensive play.",
     "football": "How fast a team plays (context, not quality)."},
    {"field": "turnovers / takeaways", "type": "raw",
     "technical": "Giveaways and takeaways per game.",
     "football": "Ball security and ball-hawking (very noisy)."},
]


def feature_dictionary() -> list[dict[str, str]]:
    rows = list(RAW_FIELDS)
    for key, nar in FEATURE_NARRATIVES.items():
        rows.append({
            "field": key,
            "type": "model feature",
            "technical": f"{nar.label}. {nar.tooltip}",
            "football": nar.home_phrase.replace("{home}", "Team A")
                                       .replace("{away}", "Team B"),
        })
    return rows
