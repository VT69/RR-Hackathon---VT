"""Leverage: how much the result is expected to swing on a ball.

For a ball in state ``s`` with outcome likelihoods ``p_k``:

    leverage(s) = sum_k  p_k * | WP(next state under outcome k) - WP(s) |

It depends only on the situation before the ball. The Leverage Index (LI) is
leverage divided by its mean over all scored balls, so LI = 2 means the ball
matters twice as much as an average one.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np
import pandas as pd

# ``impact_rule`` is carried for the ablation only; no headline model reads it.
STATE_COLUMNS: list[str] = [
    "ctx", "balls_remaining", "x", "wickets_in_hand", "scheduled_balls", "scheduled2",
    "env_rpo", "is_ipl", "impact_rule", "involves_associate",
]
FULL_WICKETS = 10


class WinProbabilityModel(Protocol):
    """Anything with ``predict(innings, states) -> WP of the batting side`` for live states."""

    def predict(self, innings: int, states: pd.DataFrame) -> np.ndarray: ...


def state_frame(balls: pd.DataFrame, innings: int, extra: tuple[str, ...] = ()) -> pd.DataFrame:
    """State columns for one innings; ``x`` is the score (first innings) or runs required (chase)."""
    rows = balls[balls["innings"] == innings]
    frame = rows.assign(x=rows["score_before"] if innings == 1 else rows["runs_required"])
    if innings == 1:
        # While the first innings is being played, a later rain reduction of the chase is unknown:
        # the side batting first expects to defend over the same number of balls it is facing.
        frame = frame.assign(scheduled2=frame["scheduled_balls"])
    return frame[[*STATE_COLUMNS, *(c for c in extra if c not in STATE_COLUMNS)]]


def win_probability(model: WinProbabilityModel, innings: int, states: pd.DataFrame) -> np.ndarray:
    """WP of the batting side, with finished states resolved by the laws rather than the model."""
    b = states["balls_remaining"].to_numpy()
    x = states["x"].to_numpy()
    w = states["wickets_in_hand"].to_numpy()
    out = np.empty(len(states))
    if innings == 2:
        won = x <= 0
        dead = ~won & ((b <= 0) | (w <= 0))
        out[won] = 1.0
        out[dead] = np.where(x[dead] == 1, 0.5, 0.0)  # scores level = tie
        live = ~(won | dead)
        if live.any():
            out[live] = model.predict(2, states[live])
        return out
    over = (b <= 0) | (w <= 0)
    if (~over).any():
        out[~over] = model.predict(1, states[~over])
    if over.any():
        chase = states[over].copy()
        chase["x"] = chase["x"] + 1
        chase["balls_remaining"] = chase["scheduled2"]
        chase["scheduled_balls"] = chase["scheduled2"]
        chase["wickets_in_hand"] = FULL_WICKETS
        out[over] = 1.0 - win_probability(model, 2, chase)
    return out


def next_state_wp(
    model: WinProbabilityModel, innings: int, states: pd.DataFrame, config: dict[str, Any]
) -> np.ndarray:
    """WP after each possible outcome class: array of shape (n_states, n_classes)."""
    classes = config["outcome_classes"]
    runs = np.array([c[1] for c in classes])
    ball = np.array([c[2] for c in classes])
    wicket = np.array([c[3] for c in classes])
    n, k = len(states), len(classes)
    rep = states.loc[states.index.repeat(k)].reset_index(drop=True)
    cls = np.tile(np.arange(k), n)
    rep["balls_remaining"] = rep["balls_remaining"].to_numpy() - ball[cls]
    rep["wickets_in_hand"] = rep["wickets_in_hand"].to_numpy() - wicket[cls]
    rep["x"] = rep["x"].to_numpy() + (runs[cls] if innings == 1 else -runs[cls])
    return win_probability(model, innings, rep).reshape(n, k)


def leverage(
    model: WinProbabilityModel,
    innings: int,
    states: pd.DataFrame,
    probs: np.ndarray,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Return (current WP, leverage) for each state, given outcome likelihoods ``probs``."""
    current = win_probability(model, innings, states)
    after = next_state_wp(model, innings, states, config)
    return current, (probs * np.abs(after - current[:, None])).sum(axis=1)


def leverage_roughness(
    model: WinProbabilityModel,
    innings: int,
    states: pd.DataFrame,
    probs: np.ndarray,
    config: dict[str, Any],
) -> tuple[float, float]:
    """Noise of leverage between neighbouring states.

    Compares leverage at each state with the average of its two neighbours one run
    either side. Returns (mean absolute gap, mean leverage); their ratio is 0 for a
    locally linear surface and grows as the surface becomes jagged.
    """
    values = []
    for shift in (-1, 0, 1):
        moved = states.assign(x=states["x"] + shift)
        values.append(leverage(model, innings, moved, probs, config)[1])
    gap = np.abs(values[1] - 0.5 * (values[0] + values[2]))
    return float(gap.mean()), float(values[1].mean())
