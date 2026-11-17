"""Unit tests for the cricket rules that are easy to get silently wrong."""

from __future__ import annotations

import pandas as pd

from src.config import load_config
from src.data.cleaning import add_ball_flags

CONFIG = load_config()


def _ball(**overrides: object) -> dict[str, object]:
    base = {
        "innings": 1,
        "runs": 0,
        "extras_type": None,
        "extras1_runs": 0,
        "extras_2_runs": 0,
        "is_wicket": False,
        "dismissal_type": None,
    }
    base.update(overrides)
    return base


def _flags(**overrides: object) -> pd.Series:
    return add_ball_flags(pd.DataFrame([_ball(**overrides)]), CONFIG).iloc[0]


def test_ordinary_ball_is_legal_and_faced() -> None:
    row = _flags(runs=4)
    assert row.is_legal and row.is_ball_faced
    assert (row.total_runs, row.batter_runs, row.bowler_runs) == (4, 4, 4)


def test_wide_is_not_faced_not_legal_and_charged_to_bowler() -> None:
    row = _flags(extras_type="wide", extras1_runs=5)  # wide that ran away for four
    assert not row.is_legal and not row.is_ball_faced
    assert (row.total_runs, row.batter_runs, row.bowler_runs) == (5, 0, 5)


def test_noball_is_faced_but_not_legal() -> None:
    for label in ("Overstep", "other"):
        row = _flags(extras_type=label, runs=6, extras1_runs=1)
        assert row.is_ball_faced and not row.is_legal
        assert (row.total_runs, row.batter_runs, row.bowler_runs) == (7, 6, 7)


def test_byes_off_a_noball_are_not_charged_to_bowler() -> None:
    row = _flags(extras_type="Overstep", extras1_runs=1, extras_2_runs=4)
    assert (row.total_runs, row.bowler_runs) == (5, 1)


def test_byes_and_legbyes_are_not_charged_to_bowler() -> None:
    for label in ("byes", "legbyes"):
        row = _flags(extras_type=label, extras1_runs=4)
        assert row.is_legal and row.is_ball_faced
        assert (row.total_runs, row.batter_runs, row.bowler_runs) == (4, 0, 0)


def test_penalty_runs_are_not_charged_to_bowler() -> None:
    row = _flags(extras_type="Penalty", extras1_runs=5)
    assert (row.total_runs, row.bowler_runs) == (5, 0)


def test_bowler_wicket_types() -> None:
    for kind in ("Caught", "Bowled", "LBW", "Stump", "Hit Wicket"):
        assert _flags(is_wicket=True, dismissal_type=kind).is_bowler_wicket
    for kind in ("RunOut", "Retire Out", "Obstructing the field", "Obstracting the field"):
        assert not _flags(is_wicket=True, dismissal_type=kind).is_bowler_wicket


def test_stumping_off_a_wide_is_a_bowler_wicket_without_a_ball_faced() -> None:
    row = _flags(extras_type="wide", extras1_runs=1, is_wicket=True, dismissal_type="Stump")
    assert row.is_bowler_wicket and not row.is_ball_faced and not row.is_legal


def test_super_over_flag() -> None:
    assert not _flags(innings=2).is_super_over
    assert _flags(innings=3).is_super_over and _flags(innings=4).is_super_over
