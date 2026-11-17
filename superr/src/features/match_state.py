"""Ball-level match state: what the situation was *before* each delivery.

Everything downstream (win probability, leverage, expected runs) reads this table.
Resources follow decision D5: balls remaining and target always use the revised or
inferred overs and the revised target, never the nominal 20 overs.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from src.data.audit import innings_summary
from src.data.cleaning import add_ball_flags, classify_competitions
from src.data.loading import RawTables

logger = logging.getLogger(__name__)


def build_match_table(raw: RawTables, config: dict[str, Any]) -> pd.DataFrame:
    """One row per in-scope match: scheduled balls, target, flags and scoring environment."""
    sem = config["ball_semantics"]
    full_balls = sem["scheduled_overs"] * sem["balls_per_over"]
    matches = classify_competitions(raw.matches, raw.competitions, config)
    flagged = add_ball_flags(raw.balls, config)
    m = innings_summary(flagged, matches)
    m = m[m["comp_type"].isin(config["scope"]["comp_types"]) & m["total1"].notna()].copy()
    revised_cols = ["first_inn_revised_overs", "second_inn_revised_overs", "revised_target"]
    m[revised_cols] = m[revised_cols].astype(float)

    # Shortened with no revised-overs field: the first innings stops on an over boundary
    # with wickets in hand and a second innings follows.
    inferred = (
        m["first_inn_revised_overs"].isna()
        & m["total2"].notna()
        & (m["legal1"] < full_balls)
        & (m["wickets1"] < 10)
        & (m["legal1"] % sem["balls_per_over"] == 0)
    )
    m["scheduled_inferred"] = inferred
    m["scheduled1"] = np.where(
        m["first_inn_revised_overs"].notna(),
        m["first_inn_revised_overs"],
        np.where(inferred, m["legal1"], full_balls),
    ).astype(int)
    m["scheduled2"] = (
        m["second_inn_revised_overs"]
        .fillna(m["first_inn_revised_overs"])
        .fillna(m["scheduled1"].where(inferred))
        .fillna(full_balls)
        .astype(int)
    )
    m["target"] = m["revised_target"].fillna(m["total1"] + 1)

    level = m["total2"].notna() & (m["total1"] == m["total2"])
    said_tied = m["result"].str.contains(config["match_state"]["result_tied_token"], case=False)
    dls_label = m["result"].str.contains(r"dls|d/l", case=False, regex=True)
    # A "revised target" that the chasing side batted on past is a par score worked out
    # after play stopped, not a target the batters were chasing.
    posthoc = m["match_id"].isin(_chases_continued_past_target(flagged, m, sem["balls_per_over"]))
    m["target_unknown"] = (m["revised_target"].isna() & (dls_label | (said_tied & ~level))) | posthoc
    m["rain_affected"] = m["dls_or_revised"] | inferred
    m["level_after_regulation"] = level
    m["has_winner"] = m["win_team_id"].notna()
    # WP models learn only from clean, decided, full-information matches.
    m["wp_train_ok"] = m["has_winner"] & ~level & ~m["rain_affected"] & m["total2"].notna()

    full = set(config["competitions"]["full_members"])
    is_t20i = m["comp_type"].eq("T20I")
    m["involves_associate"] = (is_t20i & ~(m["home_team"].isin(full) & m["away_team"].isin(full))).astype(int)
    imp = config["impact_player"]
    m["impact_rule"] = (m["comp_type"].eq(imp["comp_type"]) & (m["year"] >= imp["first_year"])).astype(int)
    m["is_ipl"] = m["comp_type"].eq("IPL").astype(int)
    m["env_rpo"] = _season_run_rate(m, sem["balls_per_over"])
    keep = [
        "match_id", "comp_id", "comp_name", "comp_type", "year", "match_date", "stage", "knock_out",
        "venue_id", "home_team_id", "away_team_id", "home_team", "away_team", "win_team_id", "result", "total1", "wickets1", "legal1",
        "total2", "wickets2", "legal2", "scheduled1", "scheduled2", "target", "scheduled_inferred",
        "target_unknown", "rain_affected", "level_after_regulation", "has_winner", "wp_train_ok",
        "involves_associate", "impact_rule", "is_ipl", "env_rpo",
    ]
    logger.info("Match table: %d in-scope matches, %d usable for WP training", len(m), int(m["wp_train_ok"].sum()))
    return m[keep].reset_index(drop=True)


def _chases_continued_past_target(flagged: pd.DataFrame, m: pd.DataFrame, min_balls: int) -> list[int]:
    """Match ids whose second innings has at least ``min_balls`` deliveries after the target was passed."""
    chase = flagged[flagged["innings"] == 2].merge(m[["match_id", "target"]], on="match_id")
    before = chase.groupby("match_id")["total_runs"].cumsum() - chase["total_runs"]
    past = chase[before >= chase["target"]].groupby("match_id").size()
    return past[past >= min_balls].index.tolist()


def _season_run_rate(m: pd.DataFrame, balls_per_over: int) -> pd.Series:
    """Runs per over for the match's competition type and season (the scoring environment).

    Deliberately *not* leave-one-out. Excluding each match makes the value unique to
    the match and a mirror of that match's own total (high-scoring match -> lower
    value), which lets flexible models identify matches: an identity leak. With the
    plain season average a match contributes about 1/75th of its own environment.
    """
    runs = m["total1"].fillna(0) + m["total2"].fillna(0)
    balls = m["legal1"].fillna(0) + m["legal2"].fillna(0)
    grp = [m["comp_type"], m["year"]]
    return runs.groupby(grp).transform("sum") / balls.groupby(grp).transform("sum") * balls_per_over


def outcome_class_index(balls: pd.DataFrame, config: dict[str, Any]) -> np.ndarray:
    """Map each delivery to its outcome class (index into ``config['outcome_classes']``).

    Classes are defined by what matters to the match state: team runs, whether a ball
    was used up, and whether a wicket fell. Rare combinations fold into the nearest class.
    """
    names = [c[0] for c in config["outcome_classes"]]
    runs = balls["total_runs"].to_numpy()
    legal = balls["is_legal"].to_numpy()
    wicket = balls["is_wicket"].to_numpy()
    legal_runs = np.select([runs <= 3, runs <= 5], [runs, 4], default=6)
    illegal_runs = np.select([runs <= 3, runs <= 5], [np.maximum(runs, 1), 5], default=7)
    label = np.where(
        legal & wicket,
        np.where(runs == 0, "LW0", "LW1"),
        np.where(legal, np.char.add("L", legal_runs.astype(str)), np.char.add("X", illegal_runs.astype(str))),
    )
    lookup = {name: i for i, name in enumerate(names)}
    return np.array([lookup[x] for x in label], dtype=np.int16)


def build_ball_states(raw: RawTables, match_table: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """One row per in-scope regular-innings delivery with the state before it was bowled."""
    flagged = add_ball_flags(raw.balls, config)
    b = flagged[~flagged["is_super_over"] & flagged["match_id"].isin(match_table["match_id"])].copy()
    b = b.sort_values(["match_id", "innings", "innings_ball_num"], kind="stable")
    b = b.merge(match_table, on="match_id", how="left", validate="many_to_one")

    grp = b.groupby(["match_id", "innings"], sort=False)
    b["legal_before"] = grp["is_legal"].cumsum() - b["is_legal"]
    b["score_before"] = grp["total_runs"].cumsum() - b["total_runs"]
    b["wickets_before"] = grp["is_wicket"].cumsum() - b["is_wicket"]
    b["wickets_in_hand"] = 10 - b["wickets_before"]

    first = b["innings"].eq(1)
    b["scheduled_balls"] = np.where(first, b["scheduled1"], b["scheduled2"])
    b["balls_remaining"] = b["scheduled_balls"] - b["legal_before"]
    b["runs_required"] = np.where(first, np.nan, b["target"] - b["score_before"])

    per_over = config["ball_semantics"]["balls_per_over"]
    rrr = b["runs_required"] * per_over / b["balls_remaining"].clip(lower=1)
    b["rrr_rel"] = rrr / b["env_rpo"]
    b["runs_vs_env"] = b["score_before"] - b["env_rpo"] * b["legal_before"] / per_over

    won = (b["win_team_id"] == b["batting_team_id"]).astype(float)
    b["batting_team_won"] = won.where(b["has_winner"] & ~b["level_after_regulation"])

    live = (b["balls_remaining"] >= 1) & (b["wickets_in_hand"] >= 1)
    chase_ok = ~b["target_unknown"] & (b["runs_required"] >= 1)
    b["state_ok"] = live & (first | chase_ok)
    b["wp_train_row"] = b["state_ok"] & b["wp_train_ok"]
    b["outcome_class"] = outcome_class_index(b, config)
    logger.info(
        "Ball states: %d rows, %d with a usable state, %d usable for WP training",
        len(b), int(b["state_ok"].sum()), int(b["wp_train_row"].sum()),
    )
    return b.reset_index(drop=True)
