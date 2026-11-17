"""Phase 4 evidence: does the opposition adjustment close the same-player gaps, and who moves?"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.metrics.player_value import overall_table, player_balls

PER_100 = 100.0
ROLE_COLUMNS: dict[str, dict[str, str]] = {
    "batter": {"player": "striker_id", "opp_tier": "bowling_tier", "raw": "v_bat", "adj": "v_bat_adj"},
    "bowler": {"player": "bowler_id", "opp_tier": "batting_tier", "raw": "v_bowl", "adj": "v_bowl_adj"},
}


def _role_rows(balls: pd.DataFrame, role: str) -> pd.DataFrame:
    scored = balls[balls["state_ok"]]
    return scored[scored["is_ball_faced"]] if role == "batter" else scored


def same_player_gaps(balls: pd.DataFrame, comparisons: list[list[str]], min_balls: int) -> pd.DataFrame:
    """For players with enough balls against both tiers: mean value against tier B minus against tier A.

    Shown for raw run value and for opposition-adjusted value. If the adjustment works,
    the adjusted gap is near zero: a player's skill does not depend on who they face.
    """
    out = []
    for role, cols in ROLE_COLUMNS.items():
        rows = _role_rows(balls, role)
        for a, b in comparisons:
            part = rows[rows[cols["opp_tier"]].isin([a, b])]
            g = part.groupby([cols["player"], cols["opp_tier"]]).agg(
                n=(cols["raw"], "size"), raw=(cols["raw"], "mean"), adj=(cols["adj"], "mean")).unstack(cols["opp_tier"])
            if not {("n", a), ("n", b)} <= set(g.columns):
                continue
            ok = (g[("n", a)] >= min_balls) & (g[("n", b)] >= min_balls)
            row: dict[str, Any] = {"role": role, "against": b, "minus_against": a, "players": int(ok.sum())}
            for kind in ("raw", "adj"):
                diff = (g.loc[ok, (kind, b)] - g.loc[ok, (kind, a)]) * PER_100
                row[f"gap_{kind}"] = diff.mean()
                row[f"se_{kind}"] = diff.std() / np.sqrt(ok.sum())
            row["share_closed"] = 1 - row["gap_adj"] / row["gap_raw"]
            out.append(row)
    return pd.DataFrame(out)


def overall_before_after(balls: pd.DataFrame, flag: str, min_balls: int) -> dict[str, pd.DataFrame]:
    """Overall value per 100 balls per player, raw and adjusted, with ranks and opposition mix."""
    out = {}
    for role, cols in ROLE_COLUMNS.items():
        raw = overall_table(player_balls(balls, role, flag))
        adjusted = overall_table(player_balls(balls.assign(**{cols["raw"]: balls[cols["adj"]]}), role, flag))
        table = raw[["player_id", "name", "teams", "balls", "value_per_100"]].rename(columns={"value_per_100": "value_raw"})
        table = table.merge(adjusted[["player_id", "value_per_100"]].rename(columns={"value_per_100": "value_adjusted"}), on="player_id")
        table["adjustment"] = table["value_adjusted"] - table["value_raw"]
        mix = _role_rows(balls, role).groupby(cols["player"])[cols["opp_tier"]].value_counts(normalize=True).unstack().fillna(0)
        table = table.join(mix.add_prefix("share_v_"), on="player_id")
        pool = table["balls"] >= min_balls
        table.loc[pool, "rank_raw"] = table.loc[pool, "value_raw"].rank(ascending=False)
        table.loc[pool, "rank_adjusted"] = table.loc[pool, "value_adjusted"].rank(ascending=False)
        out[role] = table.sort_values("value_adjusted", ascending=False)
    return out


def adjusted_means(balls: pd.DataFrame) -> pd.DataFrame:
    """Mean value per 100 balls by competition and opposition tier, raw and adjusted."""
    out = []
    for role, cols in ROLE_COLUMNS.items():
        rows = _role_rows(balls, role)
        for label, column in (("competition", "comp_type"), ("opposition tier", cols["opp_tier"]),
                              ("own group", "batter_group" if role == "batter" else "bowler_group")):
            g = rows.groupby(column)
            table = pd.DataFrame({"balls": g.size(), "raw": g[cols["raw"]].mean() * PER_100, "adjusted": g[cols["adj"]].mean() * PER_100})
            out.append(table.reset_index().rename(columns={column: "level"}).assign(role=role, slice=label))
    return pd.concat(out, ignore_index=True)[["role", "slice", "level", "balls", "raw", "adjusted"]]


def by_batting_position(balls: pd.DataFrame) -> pd.DataFrame:
    """Mean batting value per 100 balls faced by position, raw and adjusted."""
    rows = _role_rows(balls, "batter")
    rows = rows[rows["batting_order"] <= 11]
    g = rows.groupby("batting_order")
    return pd.DataFrame({"balls": g.size(), "raw": g["v_bat"].mean() * PER_100, "adjusted": g["v_bat_adj"].mean() * PER_100}).reset_index()


def eligibility(balls: pd.DataFrame, flag: str, players: dict[str, pd.DataFrame], config: dict[str, Any]) -> pd.DataFrame:
    """Per player and role: pressure balls, matches containing them, and the eligibility status."""
    cfg = config["eligibility"]
    out = []
    for role, cols in ROLE_COLUMNS.items():
        rows = _role_rows(balls, role)
        hot = rows[rows[flag]]
        g = pd.DataFrame({
            "balls": rows.groupby(cols["player"]).size(),
            "pressure_balls": hot.groupby(cols["player"]).size(),
            "pressure_matches": hot.groupby(cols["player"])["match_id"].nunique(),
            "matches": rows.groupby(cols["player"])["match_id"].nunique(),
            "share_ipl": rows["comp_type"].eq("IPL").groupby(rows[cols["player"]]).mean(),
        }).fillna({"pressure_balls": 0, "pressure_matches": 0})
        enough_balls = g["pressure_balls"] >= cfg["min_pressure_balls"]
        enough_matches = g["pressure_matches"] >= cfg["min_pressure_matches"]
        g["status"] = np.where(enough_balls & enough_matches, "eligible", "insufficient evidence")
        g["reason"] = np.select([enough_balls & enough_matches, ~enough_balls], ["", "too few pressure balls"],
                                default="pressure balls in too few matches")
        names = players[role].set_index("player_id")[["name", "teams"]]
        out.append(g.join(names).reset_index().rename(columns={cols["player"]: "player_id", "index": "player_id"}).assign(role=role))
    table = pd.concat(out, ignore_index=True)
    return table[["role", "player_id", "name", "teams", "balls", "matches", "pressure_balls", "pressure_matches",
                  "share_ipl", "status", "reason"]].sort_values(["role", "pressure_balls"], ascending=[True, False])
