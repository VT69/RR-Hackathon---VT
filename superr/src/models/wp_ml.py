"""Fitted win-probability models: logistic regression, XGBoost and LightGBM.

All three see the same information. The two tree models carry monotonic constraints:
win probability can only rise with wickets in hand and balls remaining, fall as runs
required rise (chase), and rise with the score (first innings).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

CONTEXT: list[str] = ["env_rpo", "is_ipl", "involves_associate"]
TREE_FEATURES: dict[int, list[str]] = {
    1: ["score", "balls_remaining", "wickets_in_hand", "runs_vs_env", *CONTEXT],
    2: ["runs_required", "balls_remaining", "wickets_in_hand", "rrr_rel", *CONTEXT],
}
MONOTONE: dict[int, dict[str, int]] = {
    1: {"score": 1, "balls_remaining": 1, "wickets_in_hand": 1, "runs_vs_env": 1},
    2: {"runs_required": -1, "balls_remaining": 1, "wickets_in_hand": 1, "rrr_rel": -1},
}
LOGIT_FEATURES: dict[int, list[str]] = {
    1: ["pace_z", "runs_vs_env", "wickets_in_hand", "balls_remaining", "wkt_x_balls", *CONTEXT],
    2: ["pace_z", "rrr_rel_capped", "wickets_in_hand", "balls_remaining", "pace_x_wkt", *CONTEXT],
}
RRR_CAP = 4.0


def state_features(innings: int, states: pd.DataFrame, per_over: int) -> pd.DataFrame:
    """Model features from a state frame (``balls_remaining``, ``x``, ``wickets_in_hand`` + context)."""
    b = states["balls_remaining"].to_numpy(dtype=float)
    x = states["x"].to_numpy(dtype=float)
    w = states["wickets_in_hand"].to_numpy(dtype=float)
    env = states["env_rpo"].to_numpy(dtype=float)
    f = pd.DataFrame({"balls_remaining": b, "wickets_in_hand": w}, index=states.index)
    for col in CONTEXT:
        f[col] = states[col].to_numpy()
    if innings == 2:
        f["runs_required"] = x
        f["rrr_rel"] = x * per_over / b / env
        f["rrr_rel_capped"] = np.minimum(f["rrr_rel"], RRR_CAP)
        # Runs short of environment pace, scaled by the spread that grows with sqrt(balls).
        f["pace_z"] = np.clip((x - env * b / per_over) / np.sqrt(b), -10, 10)
        f["pace_x_wkt"] = f["pace_z"] * w
    else:
        bowled = states["scheduled_balls"].to_numpy(dtype=float) - b
        f["score"] = x
        f["runs_vs_env"] = x - env * bowled / per_over
        f["pace_z"] = f["runs_vs_env"] / np.sqrt(bowled + 1.0)
        f["wkt_x_balls"] = w * b
    return f


@dataclass
class FittedWinProbability:
    """One fitted model per innings, sharing the ``predict(innings, states)`` interface."""

    name: str
    models: dict[int, Any]
    features: dict[int, list[str]]
    per_over: int

    def predict(self, innings: int, states: pd.DataFrame) -> np.ndarray:
        feats = state_features(innings, states, self.per_over)
        for col in self.features[innings]:
            if col not in feats.columns:  # ablation-only extras are read straight from the state frame
                feats[col] = states[col].to_numpy()
        return self.models[innings].predict_proba(feats[self.features[innings]])[:, 1]


def _train_frame(train: pd.DataFrame, innings: int) -> pd.DataFrame:
    rows = train[(train["innings"] == innings) & train["wp_train_row"]].copy()
    rows["x"] = rows["score_before"] if innings == 1 else rows["runs_required"]
    return rows


def _make_estimator(name: str, innings: int, features: list[str], config: dict[str, Any]) -> Any:
    seed, jobs, params = config["seed"], config["models"]["n_jobs"], config["models"]
    mono = [MONOTONE[innings].get(f, 0) for f in features]
    if name == "logistic":
        return make_pipeline(StandardScaler(), LogisticRegression(**params["logistic"]))
    if name == "xgboost":
        return xgb.XGBClassifier(
            monotone_constraints=tuple(mono), tree_method="hist", random_state=seed, n_jobs=jobs,
            eval_metric="logloss", **params["xgboost"],
        )
    if name == "lightgbm":
        return lgb.LGBMClassifier(
            monotone_constraints=mono, monotone_constraints_method="advanced", random_state=seed,
            n_jobs=jobs, deterministic=True, force_row_wise=True, subsample_freq=1, verbose=-1,
            **params["lightgbm"],
        )
    raise ValueError(f"Unknown model: {name}")


def fit_wp_model(
    name: str,
    train: pd.DataFrame,
    config: dict[str, Any],
    drop: tuple[str, ...] = (),
    extra: tuple[str, ...] = (),
) -> FittedWinProbability:
    """Fit ``name`` on clean, decided matches. ``drop``/``extra`` support feature ablations."""
    per_over = config["ball_semantics"]["balls_per_over"]
    base = LOGIT_FEATURES if name == "logistic" else TREE_FEATURES
    models: dict[int, Any] = {}
    features: dict[int, list[str]] = {}
    for innings in (1, 2):
        rows = _train_frame(train, innings)
        feats = state_features(innings, rows, per_over)
        for col in extra:
            feats[col] = rows[col].to_numpy()
        features[innings] = [f for f in base[innings] if f not in drop] + list(extra)
        est = _make_estimator(name, innings, features[innings], config)
        est.fit(feats[features[innings]], rows["batting_team_won"].astype(int))
        models[innings] = est
    return FittedWinProbability(name=name, models=models, features=features, per_over=per_over)


@dataclass
class PlattCalibrated:
    """Logistic recalibration of another model's WP: sigmoid(a + b * logit(wp)), per innings.

    It is monotone and smooth, so it corrects over- or under-confidence without
    adding steps to the win-probability surface.
    """

    base: Any
    params: dict[int, tuple[float, float]]
    clip: float

    def predict(self, innings: int, states: pd.DataFrame) -> np.ndarray:
        raw = np.clip(self.base.predict(innings, states), self.clip, 1 - self.clip)
        a, b = self.params[innings]
        return 1.0 / (1.0 + np.exp(-(a + b * np.log(raw / (1 - raw)))))


def fit_platt(raw: np.ndarray, won: np.ndarray, clip: float) -> tuple[float, float]:
    """Fit sigmoid(a + b * logit(raw)) to outcomes; returns (a, b)."""
    p = np.clip(raw, clip, 1 - clip)
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    model = LogisticRegression(C=1e6, max_iter=1000).fit(logit, won.astype(int))
    return float(model.intercept_[0]), float(model.coef_[0, 0])
