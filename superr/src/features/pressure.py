"""Leverage Index and high-pressure flags built from raw leverage."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def bowling_phase(overs: pd.Series, config: dict[str, Any]) -> pd.Series:
    """Label each ball powerplay / middle / death from its over number."""
    phases = config["pressure"]["phases"]
    out = pd.Series("death", index=overs.index)
    for name, (lo, hi) in phases.items():
        out[(overs >= lo) & (overs <= hi)] = name
    return out


def add_pressure_columns(balls: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Add LI, percentiles and both versions of the high-pressure flag.

    * ``li``                 - leverage / mean leverage over all scored balls (common scale).
    * ``li_pct``             - percentile of ``li`` among all scored balls.
    * ``li_pct_innings``     - percentile within the ball's own innings (1st or 2nd).
    * ``hp_common``          - top share of balls on the common scale.
    * ``hp_within_innings``  - top share of balls within each innings.
    """
    share = config["pressure"]["high_pressure_share"]
    out = balls.copy()
    scored = out["state_ok"] & out["leverage"].notna()
    out["li"] = out["leverage"] / out.loc[scored, "leverage"].mean()
    out["li_pct"] = out.loc[scored, "li"].rank(pct=True)
    out["li_pct_innings"] = out.loc[scored].groupby("innings")["li"].rank(pct=True)
    out["hp_common"] = (out["li_pct"] > 1 - share).fillna(False)
    out["hp_within_innings"] = (out["li_pct_innings"] > 1 - share).fillna(False)
    out["phase"] = bowling_phase(out["overid"], config)
    return out


def share_table(balls: pd.DataFrame, by: str, flags: tuple[str, ...]) -> pd.DataFrame:
    """For each level of ``by``: balls, and under each flag the count, exposure rate and share of the pressure set."""
    scored = balls[balls["state_ok"]]
    table = scored.groupby(by).size().rename("balls").to_frame()
    for flag in flags:
        hits = scored.groupby(by)[flag].sum()
        table[f"{flag}_balls"] = hits.astype(int)
        table[f"{flag}_rate"] = hits / table["balls"]
        table[f"{flag}_share"] = hits / hits.sum()
    return table.reset_index()


def outcome_summary(balls: pd.DataFrame, group: list[str]) -> pd.DataFrame:
    """Scoring and dismissal profile of groups of balls (used to prove pressure differs from ordinary)."""
    g = balls.groupby(group)
    legal = g["is_legal"].sum()
    faced = g["is_ball_faced"].sum()
    out = pd.DataFrame(
        {
            "balls": g.size(),
            "run_rate": g["total_runs"].sum() / legal * 6,
            "strike_rate": g["batter_runs"].sum() / faced * 100,
            "dismissal_pct": g["is_wicket"].sum() / legal * 100,
            "dot_pct": g.apply(lambda d: ((d["total_runs"] == 0) & d["is_legal"]).sum() / d["is_legal"].sum() * 100),
            "boundary_pct": (g["is_four"].sum() + g["is_six"].sum()) / legal * 100,
            "runs_variance": g["total_runs"].var(),
            "mean_abs_wp_swing_pct": g["abs_wp_change"].mean() * 100,
        }
    )
    return out.reset_index()


def add_realised_swing(balls: pd.DataFrame) -> pd.DataFrame:
    """Actual WP change on each ball (next ball's WP minus this one's, innings-end resolved by result).

    Used only for validation: leverage is the *expected* swing, this is the realised one.
    """
    out = balls.copy()
    nxt = out.groupby(["match_id", "innings"])["wp"].shift(-1)
    out["abs_wp_change"] = np.abs(nxt - out["wp"])
    return out
