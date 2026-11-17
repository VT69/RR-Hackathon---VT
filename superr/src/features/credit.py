"""Who is credited with what on each delivery, and the situation features of Phase 3.

Crediting rules (DECISIONS.md D22):
  * Batter runs   - runs off the bat on balls faced (never wides, byes or leg-byes).
  * Batter wicket - the dismissed batter is charged, whether striker or non-striker
                    (a non-striker run-out costs a wicket but adds no ball faced).
                    A tactical retirement ("Retire Out") is charged to nobody.
  * Bowler runs   - bat runs + wides + no-ball penalty (``bowler_runs``).
  * Bowler wicket - Caught, Bowled, LBW, Stumped, Hit Wicket (``is_bowler_wicket``).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

INNINGS_KEY: list[str] = ["match_id", "innings"]


def add_credit_columns(balls: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Add ``bat_runs``, ``striker_out`` and ``non_striker_out`` (0/1 floats where relevant)."""
    uncharged = config["value"]["uncharged_dismissals"]
    out = balls.copy()
    charged = out["is_wicket"] & ~out["dismissal_type"].isin(uncharged)
    out["striker_out"] = (charged & (out["dismissed_player_id"] == out["striker_id"])).astype(float)
    out["non_striker_out"] = (charged & (out["dismissed_player_id"] == out["non_striker_id"])).astype(float)
    out["bat_runs"] = np.where(out["is_ball_faced"], out["batter_runs"], 0).astype(float)
    out["bowler_wicket"] = out["is_bowler_wicket"].astype(float)
    return out


def add_situation_features(balls: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Add the three situation features that the Phase 2 state does not carry.

    * ``striker_balls_faced`` - balls the striker has already faced in this innings.
    * ``free_hit``            - the last delivery that was not a wide was a no-ball.
    * ``in_powerplay``        - fielding restrictions apply (scaled for shortened innings).
    """
    out = balls.sort_values([*INNINGS_KEY, "innings_ball_num"], kind="stable").copy()
    faced = out["is_ball_faced"].astype(int)
    by_batter = out.groupby([*INNINGS_KEY, "striker_id"], sort=False)
    out["striker_balls_faced"] = by_batter["is_ball_faced"].cumsum() - faced

    # A free hit carries over a wide, so look back to the last delivery that was not a wide.
    last_real = out["is_noball"].astype(float).where(~out["is_wide"])
    carried = last_real.groupby([out[k] for k in INNINGS_KEY], sort=False).ffill()
    previous = carried.groupby([out[k] for k in INNINGS_KEY], sort=False).shift(1)
    out["free_hit"] = previous.fillna(0.0).astype(int)

    powerplay_balls = np.round(config["value"]["powerplay_share"] * out["scheduled_balls"])
    out["in_powerplay"] = (out["legal_before"] < powerplay_balls).astype(int)
    return out.loc[balls.index]
