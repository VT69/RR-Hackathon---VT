"""The DP win-probability solver against cases with known answers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import load_config
from src.features.leverage import leverage, win_probability
from src.models.wp_dp import MAX_WICKETS, solve_chase, solve_first

CONFIG = load_config()
NAMES = [c[0] for c in CONFIG["outcome_classes"]]
N_BALLS, N_RUNS = 12, 30


def _constant_probs(mix: dict[str, float], shape: tuple[int, ...]) -> np.ndarray:
    probs = np.zeros((*shape, len(NAMES)))
    for name, p in mix.items():
        probs[..., NAMES.index(name)] = p
    return probs


def _chase(mix: dict[str, float]) -> np.ndarray:
    return solve_chase(_constant_probs(mix, (N_BALLS + 1, N_RUNS + 1, MAX_WICKETS + 1)), CONFIG)


def test_single_every_ball_is_deterministic() -> None:
    wp = _chase({"L1": 1.0})
    assert wp[6, 6, 5] == 1.0  # need 6 off 6 at a run a ball: certain win
    assert wp[6, 7, 5] == 0.5  # one short: scores level, tie
    assert wp[6, 8, 5] == 0.0


def test_wicket_every_ball_loses() -> None:
    wp = _chase({"LW0": 1.0})
    assert wp[6, 3, 2] == 0.0
    assert wp[6, 1, 1] == 0.5  # all out with scores level


def test_wides_do_not_use_up_balls() -> None:
    wp = _chase({"X1": 1.0})
    assert wp[1, 20, 1] == 1.0  # one ball left, only wides: the runs keep coming


def test_coin_flip_between_dot_and_single() -> None:
    wp = _chase({"L0": 0.5, "L1": 0.5})
    # Needing 1 means scores are level, so failing to score is a tie (half a win), not a loss.
    assert np.isclose(wp[1, 1, 5], 0.5 + 0.5 * 0.5)
    assert np.isclose(wp[2, 1, 5], 0.75 + 0.25 * 0.5)
    assert np.isclose(wp[2, 2, 5], 0.25 + 0.5 * 0.5)  # two singles win; exactly one single ties


def test_chase_wp_is_monotone() -> None:
    wp = _chase({"L0": 0.3, "L1": 0.35, "L4": 0.15, "L6": 0.05, "LW0": 0.08, "X1": 0.07})
    live = wp[1:, 1:, 1:]
    assert (np.diff(live, axis=1) <= 1e-12).all()  # more runs required never helps
    assert (np.diff(live, axis=0) >= -1e-12).all()  # more balls never hurts
    assert (np.diff(live, axis=2) >= -1e-12).all()  # more wickets never hurts


def test_first_innings_matches_expected_defence() -> None:
    probs = _constant_probs({"L1": 1.0}, (N_BALLS + 1, MAX_WICKETS + 1))
    defend = np.linspace(0, 1, N_RUNS + 1)  # chance of winning after finishing on each score
    v = solve_first(probs, defend, CONFIG)
    assert np.isclose(v[5, 10, 4], defend[15])  # five more singles from 10


class _TableModel:
    """Minimal WP model backed by one chase table, to test the leverage engine."""

    def __init__(self, table: np.ndarray) -> None:
        self.table = table

    def predict(self, innings: int, states: pd.DataFrame) -> np.ndarray:
        return self.table[
            states["balls_remaining"].to_numpy(), states["x"].to_numpy(), states["wickets_in_hand"].to_numpy()
        ]


def test_dp_win_probability_is_a_martingale_and_leverage_is_positive() -> None:
    mix = {"L0": 0.3, "L1": 0.35, "L4": 0.15, "L6": 0.05, "LW0": 0.08, "X1": 0.07}
    model = _TableModel(_chase(mix))
    states = pd.DataFrame({"balls_remaining": [6, 3, 10], "x": [9, 2, 25], "wickets_in_hand": [3, 1, 8], "scheduled2": 12})
    probs = np.tile(_constant_probs(mix, (1,))[0], (len(states), 1))
    current, lev = leverage(model, 2, states, probs, CONFIG)
    from src.features.leverage import next_state_wp

    expected = (probs * next_state_wp(model, 2, states, CONFIG)).sum(axis=1)
    assert np.allclose(current, expected)  # WP now = average WP after the ball
    assert (lev > 0).all()


def test_finished_states_follow_the_laws() -> None:
    model = _TableModel(np.full((N_BALLS + 1, N_RUNS + 1, MAX_WICKETS + 1), 0.4))
    states = pd.DataFrame(
        {"balls_remaining": [3, 0, 0, 2], "x": [0, 5, 1, 4], "wickets_in_hand": [2, 4, 4, 0], "scheduled2": 12}
    )
    assert win_probability(model, 2, states).tolist() == [1.0, 0.0, 0.5, 0.0]
