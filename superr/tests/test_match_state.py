"""Match state must describe the situation before each ball, using revised resources."""

from __future__ import annotations

import pandas as pd

from src.config import load_config
from src.data.loading import RawTables
from src.features.match_state import build_ball_states, build_match_table

CONFIG = load_config()


def _ball(innings: int, n: int, runs: int = 0, extras_type: str | None = None, extras1: int = 0,
          wicket: bool = False, dismissal: str | None = None) -> dict[str, object]:
    batting, bowling = (1, 2) if innings == 1 else (2, 1)
    return {
        "match_id": 1, "innings": innings, "innings_ball_num": n, "overid": 1, "over_ball_num": n,
        "batting_team_id": batting, "bowling_team_id": bowling, "striker_id": 10, "non_striker_id": 11,
        "bowler_id": 20, "runs": runs, "extras_type": extras_type, "extras1_runs": extras1,
        "extras_2_runs": 0, "is_wicket": wicket, "dismissal_type": dismissal,
        "dismissed_player_id": 10 if wicket else None, "is_four": runs == 4, "is_six": runs == 6,
    }


def _raw(balls: list[dict[str, object]], **match_overrides: object) -> RawTables:
    match = {
        "match_id": 1, "comp_id": 1, "match_date": "2024-04-01", "stage": "League", "knock_out": 0,
        "venue_id": 1, "home_team_id": 1, "away_team_id": 2, "home_team": "A", "away_team": "B", "win_team_id": 1, "result": "A Won by 2 runs",
        "first_inn_revised_overs": None, "second_inn_revised_overs": None, "revised_target": None,
    }
    match.update(match_overrides)
    comps = pd.DataFrame([{"comp_id": 1, "comp_name": "IPL 2024", "match_format": "T20", "season": "2024"}])
    empty = pd.DataFrame()
    return RawTables(pd.DataFrame(balls), pd.DataFrame([match]), comps, empty, empty, empty)


def _states(balls: list[dict[str, object]], **match_overrides: object) -> pd.DataFrame:
    raw = _raw(balls, **match_overrides)
    return build_ball_states(raw, build_match_table(raw, CONFIG), CONFIG)


FIRST = [
    _ball(1, 1, runs=4),
    _ball(1, 2, extras_type="wide", extras1=1),
    _ball(1, 3, extras_type="Overstep", runs=1, extras1=1),
    _ball(1, 4, wicket=True, dismissal="Bowled"),
    _ball(1, 5, runs=2),
]
SECOND = [_ball(2, 1, runs=1), _ball(2, 2, runs=6)]


def test_state_is_measured_before_the_ball() -> None:
    s = _states(FIRST + SECOND)
    first = s[s["innings"] == 1]
    assert first["score_before"].tolist() == [0, 4, 5, 7, 7]
    assert first["wickets_in_hand"].tolist() == [10, 10, 10, 10, 9]


def test_wides_and_noballs_do_not_use_up_balls() -> None:
    first = _states(FIRST + SECOND).query("innings == 1")
    assert first["balls_remaining"].tolist() == [120, 119, 119, 119, 118]


def test_chase_uses_first_innings_total_plus_one() -> None:
    chase = _states(FIRST + SECOND).query("innings == 2")
    assert chase["runs_required"].tolist() == [10, 9]  # first innings made 9


def test_revised_target_and_overs_override_the_nominal_ones() -> None:
    chase = _states(
        FIRST + SECOND, second_inn_revised_overs=30, revised_target=7, result="A Won by 2 runs (DLS method)"
    ).query("innings == 2")
    assert chase["runs_required"].tolist() == [7, 6]
    assert chase["balls_remaining"].tolist() == [30, 29]
    assert not chase["wp_train_row"].any()  # rain matches are scored but never used for training


def test_label_is_from_the_batting_sides_point_of_view() -> None:
    s = _states(FIRST + SECOND)
    assert s.loc[s["innings"] == 1, "batting_team_won"].eq(1).all()
    assert s.loc[s["innings"] == 2, "batting_team_won"].eq(0).all()
