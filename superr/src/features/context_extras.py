"""Optional context features tested as ablations only: team strength and venue.

Both are built strictly from matches played *before* the match they describe.
They are not part of the pressure definition unless the ablation shows they earn
their place (see ``outputs/phase2b_pressure_validation.md``).
"""

from __future__ import annotations

from typing import Any

import pandas as pd

ELO_START = 1500.0


def pre_match_elo(match_table: pd.DataFrame, k_factor: float) -> pd.DataFrame:
    """Elo rating of both sides before each match (home minus away), from earlier results only."""
    ratings: dict[int, float] = {}
    rows = []
    for m in match_table.sort_values(["match_date", "match_id"]).itertuples():
        home = ratings.get(m.home_team_id, ELO_START)
        away = ratings.get(m.away_team_id, ELO_START)
        rows.append({"match_id": m.match_id, "elo_home": home, "elo_away": away})
        if m.has_winner and not m.level_after_regulation:
            expected = 1.0 / (1.0 + 10 ** ((away - home) / 400.0))
            actual = 1.0 if m.win_team_id == m.home_team_id else 0.0
            ratings[m.home_team_id] = home + k_factor * (actual - expected)
            ratings[m.away_team_id] = away - k_factor * (actual - expected)
    return pd.DataFrame(rows)


def pre_match_venue_factor(match_table: pd.DataFrame, shrink: float, full_balls: int, per_over: int) -> pd.DataFrame:
    """Runs a completed first innings at this venue has run above the environment, shrunk to zero.

    Uses earlier matches at the same venue only; ``shrink`` is the number of pseudo-matches at zero.
    """
    totals: dict[int, list[float]] = {}
    rows = []
    for m in match_table.sort_values(["match_date", "match_id"]).itertuples():
        seen = totals.get(m.venue_id, [])
        rows.append({"match_id": m.match_id, "venue_factor": sum(seen) / (len(seen) + shrink)})
        completed = m.scheduled1 == full_balls and (m.legal1 == full_balls or m.wickets1 == 10)
        if completed:
            totals.setdefault(m.venue_id, []).append(m.total1 - m.env_rpo * full_balls / per_over)
    return pd.DataFrame(rows)


def add_context_extras(balls: pd.DataFrame, match_table: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Attach ``elo_diff`` (batting side minus bowling side) and ``venue_factor`` to the ball table."""
    cfg = config["ablation"]
    sem = config["ball_semantics"]
    elo = pre_match_elo(match_table, cfg["elo_k"])
    venue = pre_match_venue_factor(
        match_table, cfg["venue_shrink"], sem["scheduled_overs"] * sem["balls_per_over"], sem["balls_per_over"]
    )
    out = balls.merge(elo, on="match_id", how="left").merge(venue, on="match_id", how="left")
    home_batting = out["batting_team_id"] == out["home_team_id"]
    diff = out["elo_home"] - out["elo_away"]
    out["elo_diff"] = diff.where(home_batting, -diff)
    return out.drop(columns=["elo_home", "elo_away"])
