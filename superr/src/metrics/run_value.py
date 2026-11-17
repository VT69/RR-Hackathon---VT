"""Run value: one number per delivery per player, in runs, against the league in the same situation.

    batting  v = (runs off the bat - expected) - c * (dismissed - expected chance)
    bowling  v = (expected runs charged - runs charged) + c * (wicket - expected chance)

``c`` is the wicket's cost in runs for that state (``wicket_cost``). Positive is good
for the player. Leverage is NOT in these values: it is applied later, once, when
pressure balls are selected and weighted.

Win probability added (WPA) is the companion metric. It already scales with leverage
and must never be weighted by LI.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.credit import INNINGS_KEY

VALUE_COLUMNS: list[str] = [
    "v_bat_runs", "v_bat_wicket", "v_bat", "v_ns", "v_bowl_runs", "v_bowl_wicket", "v_bowl",
]


def add_run_value(balls: pd.DataFrame, expected: pd.DataFrame, cost: pd.Series) -> pd.DataFrame:
    """Add the run-value columns. ``v_ns`` is the non-striker's share (run-outs at the other end)."""
    out = balls.copy()
    out[list(expected.columns)] = expected
    out["c"] = cost
    out["v_bat_runs"] = out["bat_runs"] - out["xr_bat"]
    out["v_bat_wicket"] = -out["c"] * (out["striker_out"] - out["xw_bat"])
    out["v_bat"] = out["v_bat_runs"] + out["v_bat_wicket"]
    out["v_ns"] = -out["c"] * (out["non_striker_out"] - out["xw_ns"])
    out["v_bowl_runs"] = out["xr_bowl"] - out["bowler_runs"]
    out["v_bowl_wicket"] = out["c"] * (out["bowler_wicket"] - out["xw_bowl"])
    out["v_bowl"] = out["v_bowl_runs"] + out["v_bowl_wicket"]
    return out


def add_wpa(balls: pd.DataFrame) -> pd.DataFrame:
    """Add ``wpa``: the batting side's win probability after the ball minus before it.

    Within an innings the "after" value is the next scored ball's WP. At a natural end
    of the first innings it is one minus the chasing side's opening WP; at a natural
    end of the chase it is the result (a tie counts 0.5). Innings cut short, and the
    innings break of a rain-affected match, are left blank: that jump is the weather's.
    """
    out = balls.sort_values([*INNINGS_KEY, "innings_ball_num"], kind="stable").copy()
    scored = out[out["state_ok"]]
    grp = scored.groupby(INNINGS_KEY, sort=False)["wp"]
    nxt = grp.shift(-1)
    last = nxt.isna()

    runs_left = scored["runs_required"] - scored["total_runs"]
    balls_left = scored["balls_remaining"] - scored["is_legal"].astype(int)
    wickets_left = scored["wickets_in_hand"] - scored["is_wicket"].astype(int)
    natural = (balls_left <= 0) | (wickets_left <= 0) | ((scored["innings"] == 2) & (runs_left <= 0))

    opening = scored[scored["innings"] == 2].groupby("match_id")["wp"].first()
    after_first = 1.0 - scored["match_id"].map(opening)
    result = scored["batting_team_won"].where(~scored["level_after_regulation"], 0.5)
    end_value = np.where(scored["innings"] == 1, after_first.where(~scored["rain_affected"]), result)
    nxt = nxt.where(~last, pd.Series(end_value, index=scored.index).where(natural))

    out["wpa"] = np.nan
    out.loc[scored.index, "wpa"] = nxt - scored["wp"]
    return out.loc[balls.index]
