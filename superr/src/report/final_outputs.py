"""Phase 7: the submission files.

* ``outputs/final_rankings.csv``          - top 10 batters and top 10 bowlers (the answer).
* ``outputs/final_rankings_top15.csv``    - ranks 1-15 per role, with the evidence columns.
* ``outputs/final_rankings_eligible.csv`` - every eligible player with their final rank.

Names come from the competition-team-player mapping file (the registry, DECISIONS.md D2);
every row carries ``player_id`` and team(s) so shared names cannot be confused.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.robustness.sensitivity import _reason, pressure_record

logger = logging.getLogger(__name__)
PER_100 = 100.0


def registry_names(mapping: pd.DataFrame) -> pd.Series:
    """player_id -> registry name (the most frequent spelling if a player has more than one)."""
    return mapping.groupby("player_id")["player_name"].agg(lambda s: s.mode().iloc[0])


PRESSURE_SPECIFIC = "Pressure-specific"
ALL_ROUND = "All-round"


def profile_label(b_rank: int, b_rank_gap: int) -> str:
    """Descriptive label (D38): does option B (usual level + uplift) agree with the pressure-only rating?

    "Pressure-specific" = option B rank worse than ``b_rank_gap``: the player delivered under pressure well
    beyond their usual level. "All-round" = option B agrees: elite under pressure and overall. The caveat that
    pressure-vs-usual gaps do not reproduce across matches is stated once in the report, not per player.
    """
    return PRESSURE_SPECIFIC if b_rank > b_rank_gap else ALL_ROUND


def record_note(pressure_balls: int, short_record: int) -> str:
    """Short-record flag: fewer pressure balls than ``short_record`` means a wide interval."""
    if pressure_balls < short_record:
        return f"Short record: fewer than {short_record} pressure balls, so the interval is wide."
    return ""


def final_tables(finals: dict[str, pd.DataFrame], classes: dict[str, pd.DataFrame], balls: pd.DataFrame,
                 names: pd.Series, config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """Assemble the submission tables from the guarded final order of each role."""
    top_n, zone = config["rating"]["top_n"], config["robustness"]["tie_break_zone"]
    flag = config["pressure"]["high_pressure_flag"]
    rows = []
    for role, final in finals.items():
        scored = balls[balls["state_ok"] & balls[flag]]
        pool = scored[scored["is_ball_faced"]] if role == "batter" else scored
        league_b = (pool["is_four"] | pool["is_six"]).mean() * PER_100
        f = final.join(classes[role][["share_in_top_n", "variants_eligible"]], on="player_id")
        for r in f.itertuples():
            record = pressure_record(balls, int(r.player_id), role, flag) if r.final_rank <= zone else {}
            rows.append({
                "role": role, "rank": int(r.final_rank), "player_id": int(r.player_id),
                "player_name": names.get(int(r.player_id), r.name), "name_in_ball_data": r.name, "teams": r.teams,
                "stability": r.stability, "share_of_variants_in_top_10": r.share_in_top_n,
                "rating": r.composite, "set_by_tie_break": int(r.final_rank) != int(r.rank),
                "p_top_10": r.p_top_n, "pressure_value_shrunk": r.pressure_value_shrunk,
                "pressure_value_p05": r.pressure_value_p05, "pressure_value_p95": r.pressure_value_p95,
                "pressure_value_raw": r.pressure_value_raw, "pressure_balls": int(r.pressure_balls),
                "pressure_matches": r.pressure_matches, "matches": r.matches,
                "consistency_pts_above_base": r.consistency_shrunk * PER_100, "uplift_raw": r.uplift_raw,
                "rank_option_a": int(r.rank), "rank_option_b": int(r.rank_hierarchical), "overall_rank": int(r.overall_rank),
                "share_balls_ipl": r.share_balls_ipl,
                "label": profile_label(int(r.rank_hierarchical), config["report"]["framing_min_b_rank"]),
                "note": record_note(int(r.pressure_balls), config["report"]["short_record_pressure_balls"]),
                "evidence": _reason(role, record, league_b) if record else "",
            })
    table = pd.DataFrame(rows)
    return {
        "top10": table[table["rank"] <= top_n],
        "top15": table[table["rank"] <= zone],
        "eligible": table,
    }


def write(tables: dict[str, pd.DataFrame], out_root: Path) -> None:
    """Write the three submission files."""
    tables["top10"].to_csv(out_root / "final_rankings.csv", index=False)
    tables["top15"].to_csv(out_root / "final_rankings_top15.csv", index=False)
    tables["eligible"].to_csv(out_root / "final_rankings_eligible.csv", index=False)
    mismatch = tables["top15"][tables["top15"]["player_name"] != tables["top15"]["name_in_ball_data"]]
    if len(mismatch):
        logger.warning("Registry and ball-data names differ for: %s", mismatch[["player_id", "player_name"]].to_dict("records"))
    logger.info("Final rankings written to %s", out_root / "final_rankings.csv")
