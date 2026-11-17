"""Transparent rule-based pressure index (companion to leverage).

It answers "how pressured does this look on the scoreboard?" with three visible
ingredients, each scaled 0..1 and combined with fixed weights from ``config.yaml``:

  * rate    - chase: required rate relative to the scoring environment;
              first innings: how far the scoring rate is behind the environment.
  * wickets - wickets lost beyond what is typical at this stage of an innings.
  * time    - how far through the innings we are.

A chase that is effectively lost or won is set to zero. The weights are judgement
calls, which is exactly why this is the cross-check and not the definition.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def typical_wickets(balls: pd.DataFrame) -> pd.Series:
    """Average wickets already lost at each point of an innings (by innings and balls bowled)."""
    return balls.groupby(["innings", "legal_before"])["wickets_before"].transform("mean")


def rule_pressure_index(balls: pd.DataFrame, config: dict[str, Any]) -> np.ndarray:
    """Rule-based pressure in [0, 1] for each ball of the state table."""
    cfg = config["rule_index"]
    per_over = config["ball_semantics"]["balls_per_over"]
    first = balls["innings"].eq(1).to_numpy()

    time = (balls["legal_before"] / balls["scheduled_balls"]).clip(0, 1).to_numpy()
    excess = (balls["wickets_before"] - typical_wickets(balls)).to_numpy()
    wickets = np.clip(excess / cfg["wickets_span"], 0, 1)

    chase_rate = (balls["rrr_rel"].to_numpy() - cfg["chase_rate_low"]) / (cfg["chase_rate_high"] - cfg["chase_rate_low"])
    bowled = balls["legal_before"].to_numpy()
    run_rate_rel = np.where(
        bowled > 0, balls["score_before"].to_numpy() * per_over / np.maximum(bowled, 1) / balls["env_rpo"].to_numpy(), 1.0
    )
    first_rate = (1.0 - run_rate_rel) / cfg["first_rate_span"]
    rate = np.clip(np.where(first, first_rate, chase_rate), 0, 1)

    lost = balls["rrr_rel"].to_numpy() > cfg["chase_dead_rate"]
    won = (balls["runs_required"].to_numpy() <= balls["balls_remaining"].to_numpy() * cfg["chase_won_runs_per_ball"]) & (
        balls["wickets_in_hand"].to_numpy() >= 6
    )
    live = np.where(first, 1.0, np.where(lost | won, 0.0, 1.0))

    w = cfg["weights"]
    return live * (w["rate"] * rate + w["wickets"] * wickets + w["time"] * time)
