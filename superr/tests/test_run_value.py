"""Crediting rules, situation features, the wicket exchange rate, run value and WPA."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import load_config
from src.features.credit import add_credit_columns, add_situation_features
from src.metrics.player_value import player_balls, pressure_split
from src.metrics.run_value import add_run_value, add_wpa
from src.metrics.wicket_cost import cost_parts, raw_cost, wicket_cost

CONFIG = load_config()
STRIKER, PARTNER, BOWLER = 1, 2, 9


def _delivery(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "match_id": 1, "innings": 1, "innings_ball_num": 1, "striker_id": STRIKER, "non_striker_id": PARTNER,
        "bowler_id": BOWLER, "is_wicket": False, "dismissal_type": None, "dismissed_player_id": np.nan,
        "is_wide": False, "is_noball": False, "is_ball_faced": True, "is_legal": True, "batter_runs": 0,
        "bowler_runs": 0, "is_bowler_wicket": False, "scheduled_balls": 120, "legal_before": 0,
    }
    base.update(overrides)
    return base


def _credited(rows: list[dict[str, object]]) -> pd.DataFrame:
    return add_credit_columns(pd.DataFrame(rows), CONFIG)


def test_a_wide_gives_the_batter_no_runs_and_no_ball_faced() -> None:
    out = _credited([_delivery(is_wide=True, is_ball_faced=False, is_legal=False, batter_runs=0, bowler_runs=1)])
    assert out.loc[0, "bat_runs"] == 0 and out.loc[0, "bowler_runs"] == 1


def test_striker_dismissal_is_charged_to_the_striker_and_credited_to_the_bowler() -> None:
    out = _credited([_delivery(is_wicket=True, dismissal_type="Caught", dismissed_player_id=STRIKER, is_bowler_wicket=True)])
    assert out.loc[0, "striker_out"] == 1 and out.loc[0, "non_striker_out"] == 0 and out.loc[0, "bowler_wicket"] == 1


def test_non_striker_run_out_is_charged_to_the_dismissed_batter_not_the_bowler() -> None:
    out = _credited([_delivery(is_wicket=True, dismissal_type="RunOut", dismissed_player_id=PARTNER)])
    assert out.loc[0, "striker_out"] == 0 and out.loc[0, "non_striker_out"] == 1 and out.loc[0, "bowler_wicket"] == 0


def test_retired_out_is_charged_to_nobody() -> None:
    out = _credited([_delivery(is_wicket=True, dismissal_type="Retire Out", dismissed_player_id=STRIKER)])
    assert out.loc[0, "striker_out"] == 0 and out.loc[0, "non_striker_out"] == 0


def test_balls_faced_so_far_skips_wides_and_is_per_batter() -> None:
    rows = [
        _delivery(innings_ball_num=1),
        _delivery(innings_ball_num=2, is_wide=True, is_ball_faced=False, is_legal=False),
        _delivery(innings_ball_num=3),
        _delivery(innings_ball_num=4, striker_id=PARTNER, non_striker_id=STRIKER),
        _delivery(innings_ball_num=5),
    ]
    out = add_situation_features(pd.DataFrame(rows), CONFIG)
    assert out["striker_balls_faced"].tolist() == [0, 1, 1, 0, 2]


def test_free_hit_follows_a_no_ball_and_survives_a_wide() -> None:
    rows = [
        _delivery(innings_ball_num=1, is_noball=True, is_legal=False),
        _delivery(innings_ball_num=2, is_wide=True, is_ball_faced=False, is_legal=False),
        _delivery(innings_ball_num=3),
        _delivery(innings_ball_num=4),
    ]
    out = add_situation_features(pd.DataFrame(rows), CONFIG)
    assert out["free_hit"].tolist() == [0, 1, 1, 0]


def test_powerplay_is_six_overs_of_twenty_and_scales_for_a_short_innings() -> None:
    rows = [
        _delivery(innings_ball_num=1, legal_before=35), _delivery(innings_ball_num=2, legal_before=36),
        _delivery(innings_ball_num=3, legal_before=17, scheduled_balls=60, match_id=2),
        _delivery(innings_ball_num=4, legal_before=18, scheduled_balls=60, match_id=2),
    ]
    out = add_situation_features(pd.DataFrame(rows), CONFIG)
    assert out["in_powerplay"].tolist() == [1, 0, 1, 0]


class _LinearChase:
    """Toy chase model: every run needed costs ``per_run`` and every wicket in hand is worth ``per_wicket``."""

    def __init__(self, per_run: float, per_wicket: float) -> None:
        self.per_run, self.per_wicket = per_run, per_wicket

    def predict(self, innings: int, states: pd.DataFrame) -> np.ndarray:
        return 0.5 - self.per_run * states["x"].to_numpy() + self.per_wicket * states["wickets_in_hand"].to_numpy()


def _chase_state(balls_remaining: int = 30) -> pd.DataFrame:
    return pd.DataFrame([{"balls_remaining": balls_remaining, "x": 20, "wickets_in_hand": 5, "scheduled2": 120}])


def test_wicket_cost_is_wp_lost_over_wp_per_run() -> None:
    cost = raw_cost(cost_parts(_LinearChase(0.01, 0.05), 2, _chase_state()))
    assert np.isclose(cost.loc[0, "c_raw"], 5.0)


def test_wicket_cost_does_not_change_when_the_state_matters_twice_as_much() -> None:
    calm = raw_cost(cost_parts(_LinearChase(0.01, 0.05), 2, _chase_state()))
    tense = raw_cost(cost_parts(_LinearChase(0.02, 0.10), 2, _chase_state()))
    assert np.isclose(calm.loc[0, "c_raw"], tense.loc[0, "c_raw"])
    assert np.isclose(tense.loc[0, "wp_lost"], 2 * calm.loc[0, "wp_lost"])


def test_a_wicket_off_the_last_ball_of_a_chase_costs_nothing_beyond_the_dot() -> None:
    parts = cost_parts(_LinearChase(0.01, 0.05), 2, _chase_state(balls_remaining=1))
    assert raw_cost(parts).loc[0, "wp_lost"] == 0


def test_settled_states_use_the_fallback_and_extreme_costs_are_capped() -> None:
    n = 400
    balls = pd.DataFrame({"state_ok": True, "innings": 1, "balls_remaining": 60, "wickets_in_hand": 5}, index=range(n))
    lost = np.full(n, 0.05)
    per_run = np.full(n, 0.01)
    per_run[0] = 0.0                                  # settled: a run is worth nothing
    lost[1] = 5.0                                     # an absurd ratio that must be capped
    parts = pd.DataFrame({"wp_dot": 0.5, "wp_dot_wicket": 0.5 - lost, "wp_run_more": 0.5 + per_run,
                          "wp_run_less": 0.5 - per_run, "run_steps": 2.0})
    cost, _, cap = wicket_cost(balls, parts, CONFIG)
    assert cost.loc[0, "used_fallback"] and np.isclose(cost.loc[0, "c"], 5.0)
    assert cost.loc[1, "capped"] and np.isclose(cost.loc[1, "c"], cap)
    assert np.isclose(cost.loc[2, "c"], 5.0) and (cost["c"] >= 0).all()


def _valued() -> pd.DataFrame:
    balls = pd.DataFrame([
        {"bat_runs": 4.0, "striker_out": 0.0, "non_striker_out": 0.0, "bowler_runs": 4.0, "bowler_wicket": 0.0},
        {"bat_runs": 0.0, "striker_out": 1.0, "non_striker_out": 0.0, "bowler_runs": 0.0, "bowler_wicket": 1.0},
        {"bat_runs": 1.0, "striker_out": 0.0, "non_striker_out": 1.0, "bowler_runs": 1.0, "bowler_wicket": 0.0},
    ])
    expected = pd.DataFrame({"xr_bat": 1.5, "xw_bat": 0.05, "xw_ns": 0.01, "xr_bowl": 1.5, "xw_bowl": 0.05}, index=balls.index)
    return add_run_value(balls, expected, pd.Series(6.0, index=balls.index))


def test_run_value_formulas_for_batter_and_bowler() -> None:
    out = _valued()
    assert np.isclose(out.loc[0, "v_bat"], 2.5 + 0.3) and np.isclose(out.loc[0, "v_bowl"], -2.5 - 0.3)
    assert np.isclose(out.loc[1, "v_bat"], -1.5 - 6 * 0.95) and np.isclose(out.loc[1, "v_bowl"], 1.5 + 6 * 0.95)


def test_non_striker_run_out_costs_the_wicket_and_nothing_else() -> None:
    out = _valued()
    assert np.isclose(out.loc[2, "v_ns"], -6 * 0.99) and np.isclose(out.loc[0, "v_ns"], 6 * 0.01)
    assert np.isclose(out.loc[2, "v_bowl"], 0.5 - 0.3)      # a run-out earns the bowler no wicket credit


def _chase(result: float, level: bool = False, runs_last: int = 4, rain: bool = False) -> pd.DataFrame:
    return pd.DataFrame({
        "match_id": 1, "innings": 2, "innings_ball_num": [1, 2], "state_ok": True, "wp": [0.4, 0.6],
        "runs_required": [8, 4], "total_runs": [4, runs_last], "balls_remaining": [10, 9], "is_legal": True,
        "wickets_in_hand": 5, "is_wicket": False, "batting_team_won": result, "level_after_regulation": level,
        "rain_affected": rain,
    })


def test_wpa_runs_to_the_result_at_a_natural_end() -> None:
    assert np.allclose(add_wpa(_chase(result=1.0))["wpa"], [0.2, 0.4])


def test_wpa_is_blank_when_the_innings_was_cut_short() -> None:
    out = add_wpa(_chase(result=0.0, runs_last=1))
    assert np.isclose(out.loc[0, "wpa"], 0.2) and np.isnan(out.loc[1, "wpa"])


def _player_frame() -> pd.DataFrame:
    n = 8
    return pd.DataFrame({
        "state_ok": True, "match_id": 1, "innings": 2, "year": 2024, "comp_type": "IPL", "involves_associate": 0,
        "striker_id": STRIKER, "striker_name": "A", "non_striker_id": PARTNER, "non_striker_name": "B",
        "bowler_id": BOWLER, "bowler_name": "C", "batting_team_id": 10, "bowling_team_id": 20, "home_team_id": 10,
        "home_team": "Home", "away_team": "Away", "is_ball_faced": [True] * 7 + [False], "is_legal": [True] * 7 + [False],
        "bat_runs": 1.0, "bowler_runs": 1.0, "striker_out": 0.0, "non_striker_out": 0.0, "bowler_wicket": 0.0,
        "v_bat": [1.0, 1.0, 3.0, 3.0, 5.0, 5.0, 7.0, 0.0], "v_bat_runs": 0.0, "v_bat_wicket": 0.0, "v_ns": 0.5,
        "v_bowl": -1.0, "v_bowl_runs": -1.0, "v_bowl_wicket": 0.0, "wpa": 0.01, "li": 1.0,
        "phase": ["powerplay"] * 4 + ["death"] * 4, "match_date": "2024-01-01",
        "hp_common": [False, False, True, True, False, False, True, False],
    }, index=range(n))


def test_non_striker_rows_add_value_but_no_balls() -> None:
    rows = player_balls(_player_frame(), "batter", "hp_common")
    partner = rows[rows["player_id"] == PARTNER]
    assert partner["count"].sum() == 0 and np.isclose(partner["value"].sum(), 4.0)
    assert rows.loc[rows["player_id"] == STRIKER, "count"].sum() == 7


def test_matched_baseline_uses_the_phase_mix_of_the_pressure_balls() -> None:
    config = {"value": {**CONFIG["value"], "baseline_min_cell_balls": 1}}
    rows = player_balls(_player_frame(), "batter", "hp_common")
    split = pressure_split(rows[rows["player_id"] == STRIKER], ["phase"], config).iloc[0]
    # Pressure: powerplay 3, 3 and death 7 -> 433.3 per 100. Ordinary: powerplay 1, 1 and death 5, 5 (a wide adds no ball).
    assert np.isclose(split["pressure_value"], 1300 / 3)
    assert np.isclose(split["usual_value"], 300.0)
    assert np.isclose(split["usual_value_matched"], (2 * 100 + 1 * 500) / 3)
    assert np.isclose(split["uplift"], 1300 / 3 - 700 / 3)


def test_calibration_curve_never_rises_with_overs_left() -> None:
    from src.metrics.wicket_cost import _monotone_curve

    overs = np.arange(1, 21, dtype=float)
    noisy = 1.6 - 0.04 * overs + np.where(overs % 2 == 0, 0.15, -0.15)
    curve = _monotone_curve(overs, noisy, np.ones(20), 20)
    assert (np.diff(curve) <= 1e-12).all()
    assert curve[0] > curve[-1]
