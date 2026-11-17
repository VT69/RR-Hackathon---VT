"""Exact dynamic-programming win probability.

Chase:  WP[b, r, w] = probability the chasing side wins with ``b`` legal balls left,
        ``r`` runs required and ``w`` wickets in hand.
First:  V[b, s, w]  = probability the side batting first wins with ``b`` balls left,
        score ``s`` and ``w`` wickets in hand; its terminal value is the chance of
        defending ``s`` according to the chase table.

Both are solved by backward induction over the full state grid using the outcome
model's state-dependent transition probabilities (runs, wickets and extras). There
is no simulation: the result is deterministic and smooth in the state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.models.outcome import outcome_features, predict_outcome_probs

MAX_WICKETS = 10


def _class_arrays(config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    classes = config["outcome_classes"]
    runs = np.array([c[1] for c in classes])
    ball = np.array([c[2] for c in classes])
    wicket = np.array([c[3] for c in classes])
    return runs, ball, wicket


def solve_chase(probs: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    """Backward induction for the chase.

    ``probs[b, r, w, k]`` is the probability of outcome class ``k`` in state (b, r, w);
    only b>=1, r>=1, w>=1 are read. Returns ``WP[b, r, w]`` including terminal states:
    r == 0 means the target has been reached (win); b == 0 or w == 0 with r == 1 is a tie (0.5).
    """
    runs, ball, wicket = _class_arrays(config)
    n_balls, n_runs = probs.shape[0] - 1, probs.shape[1] - 1
    wp = np.zeros((n_balls + 1, n_runs + 1, MAX_WICKETS + 1))
    wp[0, 1, :] = 0.5
    wp[:, 1, 0] = 0.5
    wp[:, 0, :] = 1.0
    r_idx = np.arange(1, n_runs + 1)
    legal = np.flatnonzero(ball == 1)
    illegal = np.flatnonzero(ball == 0)
    for b in range(1, n_balls + 1):
        acc = np.zeros((n_runs, MAX_WICKETS))
        for k in legal:
            nxt = wp[b - 1][np.maximum(r_idx - runs[k], 0)]
            acc += probs[b, 1:, 1:, k] * nxt[:, 1 - wicket[k] : MAX_WICKETS + 1 - wicket[k]]
        # Wides and no-balls do not use up a ball: the next state is in the same ``b`` layer
        # but always needs fewer runs, so solve in increasing runs-required order.
        cur = wp[b]
        for r in range(1, n_runs + 1):
            value = acc[r - 1]
            for k in illegal:
                value = value + probs[b, r, 1:, k] * cur[max(r - runs[k], 0), 1:]
            cur[r, 1:] = value
    return wp


def solve_first(probs: np.ndarray, defend: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    """Backward induction for the first innings.

    ``probs[b, w, k]`` are transition probabilities (independent of the score);
    ``defend[s]`` is the probability of winning after finishing on ``s``.
    """
    runs, ball, wicket = _class_arrays(config)
    n_balls, n_score = probs.shape[0] - 1, len(defend) - 1
    v = np.zeros((n_balls + 1, n_score + 1, MAX_WICKETS + 1))
    v[0] = defend[:, None]
    v[:, :, 0] = defend[None, :]
    s_idx = np.arange(n_score + 1)
    legal = np.flatnonzero(ball == 1)
    illegal = np.flatnonzero(ball == 0)
    for b in range(1, n_balls + 1):
        acc = np.zeros((n_score + 1, MAX_WICKETS))
        for k in legal:
            nxt = v[b - 1][np.minimum(s_idx + runs[k], n_score)]
            acc += probs[b, 1:, k][None, :] * nxt[:, 1 - wicket[k] : MAX_WICKETS + 1 - wicket[k]]
        cur = v[b]
        p_illegal = probs[b, 1:, illegal].sum(axis=0)
        cur[n_score, 1:] = acc[n_score] / (1.0 - p_illegal)  # top of the grid refers to itself
        for s in range(n_score - 1, -1, -1):
            value = acc[s]
            for k in illegal:
                value = value + probs[b, 1:, k] * cur[min(s + runs[k], n_score), 1:]
            cur[s, 1:] = value
    return v


def _grid_frame(context: dict[str, float], n_balls: int, n_runs: int | None, per_over: int) -> pd.DataFrame:
    """All live states of the grid as model features (chase if ``n_runs`` is given)."""
    b = np.arange(1, n_balls + 1)
    w = np.arange(1, MAX_WICKETS + 1)
    if n_runs is None:
        bb, ww = np.meshgrid(b, w, indexing="ij")
        frame = pd.DataFrame({"balls_remaining": bb.ravel(), "wickets_in_hand": ww.ravel()})
    else:
        r = np.arange(1, n_runs + 1)
        bb, rr, ww = np.meshgrid(b, r, w, indexing="ij")
        frame = pd.DataFrame(
            {"balls_remaining": bb.ravel(), "runs_required": rr.ravel(), "wickets_in_hand": ww.ravel()}
        )
        frame["rrr_rel"] = frame["runs_required"] * per_over / frame["balls_remaining"] / context["env_rpo"]
    for key, value in context.items():
        frame[key] = value
    return frame


@dataclass
class DPWinProbability:
    """DP tables for every scoring context of one cross-fitting fold.

    A context is (competition type, associate flag) within a season: it fixes the
    environment run rate and the rule flags that the outcome model conditions on.
    """

    config: dict[str, Any]
    outcome_models: dict[int, Any]
    contexts: dict[int, dict[str, float]]
    drop: tuple[str, ...] = ()
    chase_tables: dict[int, np.ndarray] = field(default_factory=dict)
    first_probs: dict[int, np.ndarray] = field(default_factory=dict)
    first_tables: dict[tuple[int, int], np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        sem = self.config["ball_semantics"]
        self.n_balls: int = sem["scheduled_overs"] * sem["balls_per_over"]
        self.per_over: int = sem["balls_per_over"]
        self.n_runs: int = self.config["match_state"]["max_runs_grid"]
        self.n_classes: int = len(self.config["outcome_classes"])

    def _chase_table(self, ctx: int) -> np.ndarray:
        if ctx not in self.chase_tables:
            grid = _grid_frame(self.contexts[ctx], self.n_balls, self.n_runs, self.per_over)
            p = predict_outcome_probs(self.outcome_models[2], grid, self.n_classes)
            probs = np.zeros((self.n_balls + 1, self.n_runs + 1, MAX_WICKETS + 1, self.n_classes), dtype=np.float32)
            probs[1:, 1:, 1:, :] = p.reshape(self.n_balls, self.n_runs, MAX_WICKETS, self.n_classes)
            self.chase_tables[ctx] = solve_chase(probs, self.config)
        return self.chase_tables[ctx]

    def _first_table(self, ctx: int, scheduled2: int) -> np.ndarray:
        key = (ctx, scheduled2)
        if key not in self.first_tables:
            if ctx not in self.first_probs:
                grid = _grid_frame(self.contexts[ctx], self.n_balls, None, self.per_over)
                p = predict_outcome_probs(self.outcome_models[1], grid, self.n_classes)
                probs = np.zeros((self.n_balls + 1, MAX_WICKETS + 1, self.n_classes))
                probs[1:, 1:, :] = p.reshape(self.n_balls, MAX_WICKETS, self.n_classes)
                self.first_probs[ctx] = probs
            chase = self._chase_table(ctx)
            target = np.minimum(np.arange(self.n_runs + 1) + 1, self.n_runs)
            defend = 1.0 - chase[scheduled2, target, MAX_WICKETS]
            self.first_tables[key] = solve_first(self.first_probs[ctx], defend, self.config)
        return self.first_tables[key]

    def predict(self, innings: int, states: pd.DataFrame) -> np.ndarray:
        """WP of the batting side for live states.

        ``states`` needs ``ctx``, ``balls_remaining``, ``x`` (runs required in a chase,
        score in the first innings), ``wickets_in_hand`` and, for the first innings, ``scheduled2``.
        """
        out = np.empty(len(states))
        b = states["balls_remaining"].to_numpy().astype(int)
        x = np.clip(states["x"].to_numpy().astype(int), 0, self.n_runs)
        w = states["wickets_in_hand"].to_numpy().astype(int)
        ctx = states["ctx"].to_numpy()
        if innings == 2:
            for c in np.unique(ctx):
                rows = ctx == c
                out[rows] = self._chase_table(int(c))[b[rows], x[rows], w[rows]]
        else:
            sched2 = states["scheduled2"].to_numpy().astype(int)
            for c, s2 in {(int(a), int(d)) for a, d in zip(ctx, sched2)}:
                rows = (ctx == c) & (sched2 == s2)
                out[rows] = self._first_table(c, s2)[b[rows], x[rows], w[rows]]
        return out
