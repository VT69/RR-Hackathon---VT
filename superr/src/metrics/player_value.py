"""Player-level totals of run value, and the split into pressure and ordinary balls.

Everything is built from one long table with a row per (player, delivery): the value
credited to the player on that delivery and whether it counts towards their balls.
Rates are value per 100 balls for batters and bowlers alike.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

CONTEXT: list[str] = ["match_id", "match_date", "innings", "year", "comp_type", "phase", "involves_associate", "li", "team"]
PER_100 = 100.0


def _team_names(balls: pd.DataFrame, side: str) -> np.ndarray:
    return np.where(balls[f"{side}_team_id"] == balls["home_team_id"], balls["home_team"], balls["away_team"])


def player_balls(balls: pd.DataFrame, role: str, flag: str) -> pd.DataFrame:
    """One row per (player, delivery) with the value credited and the ball count it adds.

    Batters get two kinds of row: as striker (runs and their own dismissal; a ball faced
    unless it was a wide) and as non-striker (run-out risk only; no ball faced).
    """
    scored = balls[balls["state_ok"]]
    if role == "bowler":
        base = scored.assign(team=_team_names(scored, "bowling"))
        parts = [pd.DataFrame({
            "player_id": base["bowler_id"], "name": base["bowler_name"], "value": base["v_bowl"],
            "value_runs": base["v_bowl_runs"], "value_wickets": base["v_bowl_wicket"], "count": 1.0,
            "runs": base["bowler_runs"], "wickets": base["bowler_wicket"], "legal": base["is_legal"].astype(float),
            "wpa": -base["wpa"], "pressure": base[flag], **{c: base[c] for c in CONTEXT},
        })]
    else:
        base = scored.assign(team=_team_names(scored, "batting"))
        striker = pd.DataFrame({
            "player_id": base["striker_id"], "name": base["striker_name"], "value": base["v_bat"],
            "value_runs": base["v_bat_runs"], "value_wickets": base["v_bat_wicket"],
            "count": base["is_ball_faced"].astype(float), "runs": base["bat_runs"], "wickets": base["striker_out"],
            "legal": base["is_legal"].astype(float), "wpa": base["wpa"], "pressure": base[flag], **{c: base[c] for c in CONTEXT},
        })
        other = pd.DataFrame({
            "player_id": base["non_striker_id"], "name": base["non_striker_name"], "value": base["v_ns"],
            "value_runs": 0.0, "value_wickets": base["v_ns"], "count": 0.0, "runs": 0.0,
            "wickets": base["non_striker_out"], "legal": 0.0, "wpa": np.nan, "pressure": base[flag],
            **{c: base[c] for c in CONTEXT},
        })
        parts = [striker, other]
    return pd.concat(parts, ignore_index=True).assign(role=role)


def _names_and_teams(rows: pd.DataFrame) -> pd.DataFrame:
    names = rows.groupby("player_id")["name"].agg(lambda s: s.mode().iloc[0])
    teams = rows.groupby("player_id")["team"].agg(lambda s: ", ".join(sorted(s.unique())))
    return pd.DataFrame({"name": names, "teams": teams})


def overall_table(rows: pd.DataFrame) -> pd.DataFrame:
    """Per player: traditional numbers and overall run value (all balls, no pressure split)."""
    role = rows["role"].iloc[0]
    g = rows.groupby("player_id")
    out = _names_and_teams(rows)
    out["matches"] = g["match_id"].nunique()
    out["balls"] = g["count"].sum()
    out["runs"] = g["runs"].sum()
    out["wickets"] = g["wickets"].sum()
    out["value_total"] = g["value"].sum()
    out["value_from_runs"] = g["value_runs"].sum()
    out["value_from_wickets"] = g["value_wickets"].sum()
    out["value_per_100"] = out["value_total"] / out["balls"] * PER_100
    out["wpa_total"] = g["wpa"].sum()
    if role == "batter":
        out["strike_rate"] = out["runs"] / out["balls"] * PER_100
        out["average"] = out["runs"] / out["wickets"].where(out["wickets"] > 0)
    else:
        out["economy"] = out["runs"] / g["legal"].sum() * 6
        out["balls_per_wicket"] = g["legal"].sum() / out["wickets"].where(out["wickets"] > 0)
    return out.reset_index().assign(role=role)


def _rate_and_se(rows: pd.DataFrame) -> pd.DataFrame:
    """Value per 100 balls per player with a ball-level standard error."""
    g = rows.groupby("player_id")
    balls = g["count"].sum()
    rate = g["value"].sum() / balls
    resid = rows["value"] - rows["player_id"].map(rate) * rows["count"]
    se = np.sqrt((resid**2).groupby(rows["player_id"]).sum()) / balls
    return pd.DataFrame({"balls": balls, "rate": rate * PER_100, "se": se * PER_100})


def pressure_split(rows: pd.DataFrame, cells: list[str], config: dict[str, Any]) -> pd.DataFrame:
    """Pressure performance, usual level, context-matched usual level and uplift, per player.

    The matched usual level averages the player's ordinary-ball rate in each cell
    (``cells``, e.g. bowling phase) using the cell shares of their pressure balls. A cell
    with too few ordinary balls uses the player's overall ordinary rate. Leverage enters
    only through the ``pressure`` flag; no LI weights are applied here (Phase 5).
    """
    min_cell = config["value"]["baseline_min_cell_balls"]
    hot, cold = rows[rows["pressure"]], rows[~rows["pressure"]]
    p, u = _rate_and_se(hot), _rate_and_se(cold)
    out = pd.DataFrame({"pressure_balls": p["balls"], "ordinary_balls": u["balls"], "pressure_value": p["rate"],
                        "pressure_se": p["se"], "usual_value": u["rate"], "usual_se": u["se"]})
    key = ["player_id", *cells]
    cold_cells = cold.groupby(key).agg(value=("value", "sum"), balls=("count", "sum"))
    hot_cells = hot.groupby(key)["count"].sum().rename("hot_balls").reset_index()
    merged = hot_cells.merge(cold_cells.reset_index(), on=key, how="left")
    enough = merged["balls"] >= min_cell
    fallback = merged["player_id"].map(u["rate"])
    cell_rate = (merged["value"] / merged["balls"] * PER_100).where(enough, fallback)
    merged["weighted"] = cell_rate * merged["hot_balls"]
    merged["covered"] = merged["hot_balls"].where(enough, 0.0)
    by_player = merged.groupby("player_id")[["weighted", "covered", "hot_balls"]].sum()
    out["usual_value_matched"] = by_player["weighted"] / by_player["hot_balls"]
    out["matched_coverage"] = by_player["covered"] / by_player["hot_balls"]
    out["uplift"] = out["pressure_value"] - out["usual_value_matched"]
    out["uplift_se"] = np.sqrt(out["pressure_se"] ** 2 + out["usual_se"] ** 2)
    return out.join(_names_and_teams(rows)).reset_index().assign(role=rows["role"].iloc[0])
