"""The run cost of a wicket in a given match state, read from the win-probability model.

    c(state) = win probability lost if a wicket falls / win probability one run is worth

Both parts are taken one ball on from the state (the ball is a dot either way), so the
only difference in the numerator is the wicket and the only difference in the
denominator is one run. Leverage cancels in the ratio: if the state mattered twice as
much, both parts would double.

Where the match is settled the denominator is too small to divide by. Those balls use
a fallback: the typical first-innings cost for the same overs left and wickets in
hand. In a first innings there is no target, so that cost reflects only the batting
resources a wicket removes - the right stand-in once the target has stopped mattering.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from sklearn.isotonic import IsotonicRegression

from src.features.leverage import WinProbabilityModel, state_frame, win_probability

PART_COLUMNS: list[str] = ["wp_dot", "wp_dot_wicket", "wp_run_more", "wp_run_less", "run_steps"]


def cost_parts(model: WinProbabilityModel, innings: int, states: pd.DataFrame) -> pd.DataFrame:
    """Win probability one ball on: after a dot, after a dot with a wicket, and one run either side."""
    dot = states.assign(balls_remaining=states["balls_remaining"] - 1)
    wicket = dot.assign(wickets_in_hand=dot["wickets_in_hand"] - 1)
    if innings == 1:                                 # x is the score: a run adds one; it cannot go below 0
        more, less = dot.assign(x=dot["x"] + 1), dot.assign(x=np.maximum(dot["x"] - 1, 0))
    else:                                            # x is the runs required: a run takes one off
        more, less = dot.assign(x=dot["x"] - 1), dot.assign(x=dot["x"] + 1)
    steps = (more["x"] - less["x"]).abs().to_numpy()
    return pd.DataFrame({
        "wp_dot": win_probability(model, innings, dot),
        "wp_dot_wicket": win_probability(model, innings, wicket),
        "wp_run_more": win_probability(model, innings, more),
        "wp_run_less": win_probability(model, innings, less),
        "run_steps": steps,
    }, index=states.index)


def crossfit_cost_parts(balls: pd.DataFrame, fold_models: dict[int, dict[str, Any]], name: str) -> pd.DataFrame:
    """Cost parts for every scored ball, each from the fold that held out the ball's season."""
    scored = balls[balls["state_ok"]]
    out = pd.DataFrame(np.nan, index=balls.index, columns=PART_COLUMNS)
    for year, fold in fold_models.items():
        for innings in (1, 2):
            rows = scored[(scored["year"] == year) & (scored["innings"] == innings)]
            if not rows.empty:
                out.loc[rows.index] = cost_parts(fold[name], innings, state_frame(rows, innings)).to_numpy()
    return out


def raw_cost(parts: pd.DataFrame) -> pd.DataFrame:
    """``wp_lost`` to a wicket, ``wp_per_run`` and their ratio (NaN where a run is worth nothing)."""
    lost = (parts["wp_dot"] - parts["wp_dot_wicket"]).clip(lower=0)
    per_run = (parts["wp_run_more"] - parts["wp_run_less"]) / parts["run_steps"]
    ratio = lost / per_run.where(per_run > 0)
    return pd.DataFrame({"wp_lost": lost, "wp_per_run": per_run, "c_raw": ratio})


def fallback_table(balls: pd.DataFrame, cost: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Typical first-innings wicket cost by overs left and wickets in hand, gaps filled by interpolation."""
    cfg = config["value"]
    per_over = config["ball_semantics"]["balls_per_over"]
    usable = balls["state_ok"] & (balls["innings"] == 1) & (cost["wp_per_run"] >= cfg["min_wp_per_run"])
    frame = pd.DataFrame({
        "overs_left": np.ceil(balls.loc[usable, "balls_remaining"] / per_over).astype(int),
        "wickets_in_hand": balls.loc[usable, "wickets_in_hand"].astype(int),
        "c": cost.loc[usable, "c_raw"],
    })
    cells = frame.groupby(["overs_left", "wickets_in_hand"])["c"].agg(["mean", "size"])
    grid = cells["mean"].where(cells["size"] >= cfg["fallback_min_cell"]).unstack("wickets_in_hand")
    overs = range(1, config["ball_semantics"]["scheduled_overs"] + 1)
    grid = grid.reindex(index=overs, columns=range(1, 11))
    grid = grid.interpolate(axis=1, limit_direction="both").interpolate(axis=0, limit_direction="both")
    return grid


def event_rows(balls: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Balls usable for measuring what a wicket actually cost: full-length, unaffected first innings.

    A first innings has no target, so the runs a side adds after a ball are the whole story.
    (In a chase the innings stops at the target, which caps the runs a wicket can cost.)
    """
    per_over = config["ball_semantics"]["balls_per_over"]
    full = config["ball_semantics"]["scheduled_overs"] * per_over
    mask = (balls["state_ok"] & (balls["innings"] == 1) & (balls["scheduled_balls"] == full)
            & ~balls["rain_affected"] & balls["is_legal"])
    d = balls.loc[mask, ["year", "balls_remaining", "wickets_in_hand", "is_wicket"]].copy()
    d["future_runs"] = balls.loc[mask, "total1"] - balls.loc[mask, "score_before"] - balls.loc[mask, "total_runs"]
    d["overs_left"] = np.ceil(d["balls_remaining"] / per_over).astype(int)
    return d


def runs_lost(rows: pd.DataFrame, train: pd.Series, min_cell: int) -> pd.Series:
    """For each wicket ball: runs added after a non-wicket ball in the same state, minus runs actually added.

    The state is balls left x wickets in hand, measured on the ``train`` rows only; a state
    with fewer than ``min_cell`` non-wicket balls uses the over instead of the exact ball.
    """
    base = rows[train & ~rows["is_wicket"]]
    fine = base.groupby(["balls_remaining", "wickets_in_hand"])["future_runs"].agg(["mean", "size"])
    coarse = base.groupby(["overs_left", "wickets_in_hand"])["future_runs"].mean()
    fell = rows[rows["is_wicket"]]
    key_fine = pd.MultiIndex.from_frame(fell[["balls_remaining", "wickets_in_hand"]])
    key_coarse = pd.MultiIndex.from_frame(fell[["overs_left", "wickets_in_hand"]])
    ok = fine["mean"].where(fine["size"] >= min_cell)
    expected = pd.Series(ok.reindex(key_fine).to_numpy(), index=fell.index)
    expected = expected.fillna(pd.Series(coarse.reindex(key_coarse).to_numpy(), index=fell.index))
    return expected - fell["future_runs"]


def _monotone_curve(overs: np.ndarray, ratio: np.ndarray, weight: np.ndarray, n_overs: int) -> np.ndarray:
    """A curve that never rises with overs left: isotonic fit, then straight lines between its block centres."""
    iso = IsotonicRegression(increasing=False, out_of_bounds="clip").fit(overs, ratio, sample_weight=weight)
    fit = iso.predict(overs)
    centres, values = [], []
    for value in pd.unique(fit):
        block = fit == value
        centres.append(np.average(overs[block], weights=weight[block]))
        values.append(value)
    order = np.argsort(centres)
    return np.interp(np.arange(1, n_overs + 1), np.array(centres)[order], np.array(values)[order])


def fit_calibration(rows: pd.DataFrame, lost: pd.Series, cost: pd.Series, n_overs: int) -> np.ndarray:
    """Factor by overs left that brings the model's wicket cost in line with the runs actually lost."""
    fell = rows.loc[lost.index].assign(lost=lost, c=cost.loc[lost.index])
    fell = fell[fell["c"] > 0]                       # last-ball wickets cost 0 by rule
    per_over = fell.groupby("overs_left").agg(lost=("lost", "sum"), c=("c", "sum"), n=("c", "size"))
    return _monotone_curve(per_over.index.to_numpy(float), (per_over["lost"] / per_over["c"]).to_numpy(),
                           per_over["n"].to_numpy(float), n_overs)


def crossfit_calibration(balls: pd.DataFrame, cost: pd.Series, config: dict[str, Any]) -> pd.DataFrame:
    """Calibration factors by overs left (rows) for each held-out season (columns), plus "all".

    For season Y both the runs-lost baseline and the factors use the other seasons only.
    ``cost`` is the uncalibrated, uncapped wicket cost per ball.
    """
    rows = event_rows(balls, config)
    n_overs = config["ball_semantics"]["scheduled_overs"]
    min_cell = config["value"]["calibration_min_cell"]
    out = {}
    for year in [*sorted(rows["year"].unique()), "all"]:
        train = pd.Series(True, index=rows.index) if year == "all" else rows["year"] != year
        lost = runs_lost(rows, train, min_cell)
        lost = lost[train.loc[lost.index]]
        out[year] = fit_calibration(rows, lost, cost, n_overs)
    return pd.DataFrame(out, index=pd.Index(range(1, n_overs + 1), name="overs_left"))


def wicket_cost(
    balls: pd.DataFrame, parts: pd.DataFrame, config: dict[str, Any], factors: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Final cost per ball: the ratio where it is reliable, the fallback elsewhere, calibrated, then capped.

    Returns (per-ball frame, fallback grid, cap). ``factors`` (from ``crossfit_calibration``)
    rescales the cost by overs left using the factors of the ball's held-out season; the
    same factors apply to both innings. The cap is the configured quantile of the reliable
    (calibrated) ratios, so a handful of last-wicket states cannot dominate a player's value.
    On the last ball of an innings a wicket costs nothing beyond the dot ball, so c = 0.
    """
    cfg = config["value"]
    per_over = config["ball_semantics"]["balls_per_over"]
    cost = raw_cost(parts)
    grid = fallback_table(balls, cost, config)
    scored = balls["state_ok"]
    last_ball = scored & (balls["balls_remaining"] <= 1)
    settled = scored & ~last_ball & ~(cost["wp_per_run"] >= cfg["min_wp_per_run"])
    overs_left = np.ceil(balls["balls_remaining"].clip(lower=1) / per_over).clip(upper=grid.index.max())
    wickets = balls["wickets_in_hand"].clip(lower=1, upper=10)
    lookup = grid.stack()
    keys = pd.MultiIndex.from_arrays([overs_left.fillna(1).astype(int), wickets.fillna(1).astype(int)])
    cost["c_fallback"] = lookup.reindex(keys).to_numpy()
    cost["used_fallback"] = settled
    cost["last_ball"] = last_ball
    chosen = cost["c_raw"].where(~settled, cost["c_fallback"]).where(~last_ball, 0.0)
    cost["factor"] = 1.0
    if factors is not None:
        years = balls["year"].where(balls["year"].isin(factors.columns), "all")
        keys = pd.MultiIndex.from_arrays([overs_left.fillna(1).astype(int), years])
        cost["factor"] = factors.stack().reindex(keys).to_numpy()
    chosen = chosen * cost["factor"]
    cost["c_uncapped"] = chosen.where(scored)
    cap = float(chosen[scored & ~settled & ~last_ball].quantile(cfg["cost_cap_quantile"]))
    cost["c"] = chosen.clip(upper=cap).where(scored)
    cost["capped"] = scored & (chosen > cap)
    return cost, grid, cap
