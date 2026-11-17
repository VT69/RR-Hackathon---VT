"""League-average expectations for a delivery, given only the situation.

Five expectations, each cross-fitted by season (a ball in season Y is judged by
models that never saw season Y):

    xr_bat   expected runs off the bat on a ball faced
    xw_bat   chance the striker is dismissed
    xw_ns    chance the non-striker is dismissed (run-outs)
    xr_bowl  expected runs charged to the bowler
    xw_bowl  chance of a bowler-credited wicket

Batting and bowling expectations use the SAME features, so a bowler gets no credit
merely for bowling at a new batter, and a batter none for walking in late.
Nothing here identifies a player, a team, a venue or a match.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SITUATION: list[str] = [
    "balls_remaining", "wickets_in_hand", "in_powerplay", "striker_balls_faced", "free_hit",
    "env_rpo", "is_ipl", "involves_associate",
]
FEATURES: dict[int, list[str]] = {
    1: [*SITUATION, "runs_vs_env"],                  # first innings: pace against the environment
    2: [*SITUATION, "runs_required", "rrr_rel"],     # chase: the task and its required rate
}


@dataclass(frozen=True)
class Expectation:
    """One expectation: its target column, whether it is a probability, and which rows it covers."""

    target: str
    binary: bool
    faced_only: bool = False


EXPECTATIONS: dict[str, Expectation] = {
    "xr_bat": Expectation("bat_runs", binary=False, faced_only=True),
    "xw_bat": Expectation("striker_out", binary=True),
    "xw_ns": Expectation("non_striker_out", binary=True),
    "xr_bowl": Expectation("bowler_runs", binary=False),
    "xw_bowl": Expectation("bowler_wicket", binary=True),
}


def expectation_features(innings: int, drop: tuple[str, ...] = ()) -> list[str]:
    """Feature list for one innings, optionally without some features (ablations)."""
    return [f for f in FEATURES[innings] if f not in drop]


def _rows(balls: pd.DataFrame, spec: Expectation, innings: int) -> pd.Series:
    mask = balls["state_ok"] & (balls["innings"] == innings)
    return mask & balls["is_ball_faced"] if spec.faced_only else mask


def _estimator(spec: Expectation, config: dict[str, Any]) -> Any:
    cls = lgb.LGBMClassifier if spec.binary else lgb.LGBMRegressor
    return cls(
        random_state=config["seed"], n_jobs=config["models"]["n_jobs"], deterministic=True,
        force_row_wise=True, subsample_freq=1, verbose=-1,
        **config["value"]["expected_wickets" if spec.binary else "expected_runs"],
    )


def _predict(model: Any, spec: Expectation, features: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(features)[:, 1] if spec.binary else model.predict(features)


def crossfit_expectations(balls: pd.DataFrame, config: dict[str, Any], drop: tuple[str, ...] = ()) -> pd.DataFrame:
    """Held-out expectations for every scored ball (NaN where the state is unusable).

    ``xr_bat`` is 0 on wides: the striker did not face the ball.
    """
    out = pd.DataFrame(np.nan, index=balls.index, columns=list(EXPECTATIONS))
    for year in sorted(balls["year"].unique()):
        train, test = balls[balls["year"] != year], balls[balls["year"] == year]
        for name, spec in EXPECTATIONS.items():
            for innings in (1, 2):
                cols = expectation_features(innings, drop)
                fit_rows = train[_rows(train, spec, innings)]
                model = _estimator(spec, config).fit(fit_rows[cols], fit_rows[spec.target])
                rows = test[_rows(test, spec, innings)]
                out.loc[rows.index, name] = _predict(model, spec, rows[cols])
        logger.info("Expectation models: held out %s", year)
    wides = balls["state_ok"] & ~balls["is_ball_faced"]
    out.loc[wides, "xr_bat"] = 0.0
    return out


def recalibrate_by_season(balls: pd.DataFrame, expected: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rescale each expectation so that it matches the actual total in every season and competition type.

    Tree models cannot extrapolate: a held-out season that scores faster than any season
    in training is under-predicted, which would tilt run value by season. One factor per
    (season, competition type, expectation) fixes the level and leaves the shape across
    match states untouched. It is the same descriptive use of the season's scoring level
    as the scoring-environment feature (DECISIONS.md D10), and it identifies no match.
    Returns (recalibrated expectations, the factors).
    """
    out = expected.copy()
    scored = balls["state_ok"]
    factors = []
    for (year, comp), part in balls[scored].groupby(["year", "comp_type"]):
        row: dict[str, Any] = {"year": year, "comp_type": comp, "balls": len(part)}
        for name, spec in EXPECTATIONS.items():
            rows = part[part["is_ball_faced"]] if spec.faced_only else part
            factor = rows[spec.target].sum() / expected.loc[rows.index, name].sum()
            out.loc[part.index, name] = expected.loc[part.index, name] * factor
            row[name] = factor
        factors.append(row)
    return out, pd.DataFrame(factors)


def baseline_expectations(balls: pd.DataFrame) -> pd.DataFrame:
    """The traditional yardstick: the average for the same competition type, innings and phase.

    Also held out by season, so it competes with the models on equal terms.
    """
    out = pd.DataFrame(np.nan, index=balls.index, columns=list(EXPECTATIONS))
    cell = ["comp_type", "innings", "phase"]
    for year in sorted(balls["year"].unique()):
        train, test = balls[(balls["year"] != year) & balls["state_ok"]], balls[(balls["year"] == year) & balls["state_ok"]]
        for name, spec in EXPECTATIONS.items():
            source = train[train["is_ball_faced"]] if spec.faced_only else train
            means = source.groupby(cell)[spec.target].mean().rename("value").reset_index()
            rows = test[test["is_ball_faced"]] if spec.faced_only else test
            merged = rows[cell].merge(means, on=cell, how="left")
            out.loc[rows.index, name] = merged["value"].to_numpy()
    return out


def fingerprint_audit(balls: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Can any feature identify a match? (The Phase 2b leak was a feature nearly unique per match.)

    A feature that is constant within a match is safe only if many matches share each
    of its values. A feature that varies ball by ball describes the state, not the match.
    """
    scored = balls[balls["state_ok"]]
    need = max(2, int(config["value"]["fingerprint_min_match_share"] * scored["match_id"].nunique()))
    rows = []
    for feature in sorted({f for cols in FEATURES.values() for f in cols}):
        values = scored[["match_id", feature]].dropna()
        per_match = values.groupby("match_id")[feature].nunique()
        constant = bool((per_match == 1).all())
        distinct = int(values[feature].round(6).nunique())
        row: dict[str, Any] = {"feature": feature, "constant_within_match": constant, "distinct_values": distinct}
        if constant:
            sharing = values.drop_duplicates("match_id")[feature].round(6).value_counts()
            row["fewest_matches_sharing_a_value"] = int(sharing.min())
            row["verdict"] = "safe: a season-level value" if sharing.min() >= need else "RISK: near-unique per match"
        else:
            row["fewest_matches_sharing_a_value"] = np.nan
            row["verdict"] = "safe: changes ball by ball (state, not identity)"
        rows.append(row)
    return pd.DataFrame(rows)
