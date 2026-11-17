"""Phase 5 report tables: draft top 10, the next five, overall v pressure, and the tier-split check."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.metrics.player_value import overall_table, player_balls

PER_100 = 100.0


def player_context(rows: pd.DataFrame) -> pd.DataFrame:
    """Per player: matches, and where their pressure balls came from (death overs, chases)."""
    counted = rows[rows["count"] > 0]
    hot = counted[counted["pressure"]]
    return pd.DataFrame({
        "matches": counted.groupby("player_id")["match_id"].nunique(),
        "pressure_matches": hot.groupby("player_id")["match_id"].nunique(),
        "share_pressure_in_death": hot["phase"].eq("death").groupby(hot["player_id"]).mean(),
        "share_pressure_in_chase": hot["innings"].eq(2).groupby(hot["player_id"]).mean(),
        "share_balls_in_chase": counted["innings"].eq(2).groupby(counted["player_id"]).mean(),
        "share_balls_ipl": counted["comp_type"].eq("IPL").groupby(counted["player_id"]).mean(),
    })


def assemble(table: pd.DataFrame, rows: pd.DataFrame, overall: pd.DataFrame) -> pd.DataFrame:
    """Rating table with names, teams, match counts and the overall-value rank among the same eligible players."""
    out = table.merge(overall[["player_id", "name", "teams", "balls", "value_per_100"]], on="player_id", how="left")
    out = out.rename(columns={"value_per_100": "overall_value", "balls": "balls_all"})
    out = out.join(player_context(rows), on="player_id")
    out["overall_rank"] = out["overall_value"].rank(ascending=False, method="first").astype(int)
    out["rank_change"] = out["overall_rank"] - out["rank"]
    return out.sort_values("rank")


def overall_v_pressure(rated: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """Everyone in either top ``top_n``: overall rank, pressure rank and the numbers that explain the move."""
    keep = rated[(rated["rank"] <= top_n) | (rated["overall_rank"] <= top_n)].copy()
    keep["group"] = np.select([(keep["rank"] <= top_n) & (keep["overall_rank"] <= top_n), keep["rank"] <= top_n],
                              ["in both", "rises into pressure top 10"], default="drops out of pressure top 10")
    cols = ["group", "player_id", "name", "overall_rank", "rank", "overall_value", "usual_value_raw", "pressure_value_raw",
            "pressure_value_shrunk", "uplift_raw", "consistency_shrunk", "pressure_balls", "share_pressure_in_death",
            "share_pressure_in_chase", "p_top_n"]
    return keep[cols].sort_values(["group", "rank"])


def split_check(with_split: pd.DataFrame, without_split: pd.DataFrame, flag: str, n_show: int, min_balls: int,
                named: list[int]) -> pd.DataFrame:
    """T20I-only players: overall and pressure value with and without the IPL split of the tiers.

    Pressure value here is the plain (uniform) mean adjusted value on pressure balls, so the
    check isolates the opposition model from the Phase 5 rating choices.
    """
    out = []
    for role, raw, adj, group in (("batter", "v_bat", "v_bat_adj", "batter_group"),
                                  ("bowler", "v_bowl", "v_bowl_adj", "bowler_group")):
        tables = {}
        for label, balls in (("split", with_split), ("no_split", without_split)):
            rows = player_balls(balls.assign(**{raw: balls[adj]}), role, flag)
            overall = overall_table(rows).set_index("player_id")
            hot = rows[rows["pressure"]]
            pressure = hot.groupby("player_id")["value"].sum() / hot.groupby("player_id")["count"].sum() * PER_100
            pool = overall["balls"] >= min_balls
            tables[label] = pd.DataFrame({
                "overall": overall["value_per_100"], "pressure": pressure,
                "overall_rank": overall.loc[pool, "value_per_100"].rank(ascending=False),
            })
            if label == "split":
                base = overall[["name", "balls"]]
        player_col = "striker_id" if role == "batter" else "bowler_id"
        groups = with_split.drop_duplicates(player_col).set_index(player_col)[group]
        t20i_only = groups[groups.str.startswith("T20I") & ~groups.str.endswith("(also IPL)")]
        table = base.join(groups.rename("group")).join(tables["split"].add_suffix("_split")).join(tables["no_split"].add_suffix("_no_split"))
        table = table[table.index.isin(t20i_only.index)]
        named_ok = set(named) & set(table.index[table["balls"] >= min_balls])
        chosen = set(table.nlargest(n_show, "balls").index) | named_ok
        table = table.loc[sorted(chosen, key=lambda p: -table.loc[p, "balls"])]
        table["overall_change"] = table["overall_split"] - table["overall_no_split"]
        table["pressure_change"] = table["pressure_split"] - table["pressure_no_split"]
        out.append(table.reset_index().rename(columns={"index": "player_id"}).assign(role=role))
    return pd.concat(out, ignore_index=True)


def summary_rows(rated: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """Draft top n and the next few, with the columns the checkpoint asks for."""
    cols = ["rank", "player_id", "name", "teams", "pressure_balls", "pressure_matches", "matches", "pressure_value_raw",
            "pressure_value_shrunk", "pressure_value_p05", "pressure_value_p95", "consistency_raw", "consistency_shrunk",
            "uplift_shrunk", "composite", "rank_p05", "rank_p95", "p_top_n", "p_top_n_equal_balls", "overall_rank",
            "share_balls_ipl"]
    top, nxt = cfg["top_n"], cfg["top_n"] + cfg["show_next"]
    head = rated[rated["rank"] <= nxt][cols].copy()
    tenth = float(rated.loc[rated["rank"] == top, "composite"].iloc[0])
    head["gap_to_rank_10"] = head["composite"] - tenth
    return {"top": head[head["rank"] <= top], "next": head[head["rank"] > top]}
