"""Season-wise cross-fitting of every win-probability candidate.

For each season, all models are trained on the other seasons only, then applied to
that season. No win probability or leverage value comes from a model that saw the
result of its own match (or of any match in its season).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.features.leverage import leverage, leverage_roughness, state_frame
from src.models.outcome import fit_outcome_model, outcome_features, predict_outcome_probs
from src.models.wp_dp import DPWinProbability
from src.models.wp_ml import PlattCalibrated, fit_platt, fit_wp_model

logger = logging.getLogger(__name__)

FITTED_MODELS: tuple[str, ...] = ("logistic", "xgboost", "lightgbm")
CANDIDATES: tuple[str, ...] = (*FITTED_MODELS, "dp", "dp_platt")


def add_context_code(balls: pd.DataFrame) -> pd.DataFrame:
    """``ctx`` = season x (IPL / T20I full members / T20I with an associate)."""
    group = np.where(balls["is_ipl"] == 1, 0, 1 + balls["involves_associate"])
    return balls.assign(ctx=balls["year"] * 10 + group)


def _contexts(rows: pd.DataFrame) -> dict[int, dict[str, float]]:
    agg = rows.groupby("ctx")[["env_rpo", "is_ipl", "impact_rule", "involves_associate"]].mean()
    return {int(ctx): {k: float(v) for k, v in rec.items()} for ctx, rec in agg.iterrows()}


@dataclass
class CrossFit:
    """Models per held-out season plus cross-fitted outcome likelihoods for every scored ball."""

    balls: pd.DataFrame
    models: dict[int, dict[str, Any]]
    outcome_probs: np.ndarray
    config: dict[str, Any]
    outcome_models: dict[int, dict[int, Any]]

    def scored(self) -> pd.DataFrame:
        """Balls with a usable state."""
        return self.balls[self.balls["state_ok"]]


def _raw_dp_wp(fold_models: dict[str, Any], rows: pd.DataFrame) -> np.ndarray:
    out = np.empty(len(rows))
    for innings in (1, 2):
        mask = (rows["innings"] == innings).to_numpy()
        out[mask] = fold_models["dp"].predict(innings, state_frame(rows, innings))
    return out


def run_crossfit(
    balls: pd.DataFrame,
    config: dict[str, Any],
    drop: tuple[str, ...] = (),
    fitted: tuple[str, ...] = FITTED_MODELS,
    extra: tuple[str, ...] = (),
    with_dp: bool = True,
) -> CrossFit:
    """Fit all candidates leaving out each season in turn.

    ``drop`` removes context features from every model; ``extra`` adds ablation-only
    columns (for the DP they must be constant within a context, e.g. the Impact Player
    flag); ``with_dp=False`` skips the outcome model and the DP (quick ablations of the
    fitted models).
    """
    balls = add_context_code(balls).reset_index(drop=True)
    n_classes = len(config["outcome_classes"])
    clip = config["models"]["prob_clip"]
    probs = np.full((len(balls), n_classes), np.nan)
    models: dict[int, dict[str, Any]] = {}
    outcome_models: dict[int, dict[int, Any]] = {}
    for year in sorted(balls["year"].unique()):
        train = balls[balls["year"] != year]
        test = balls[(balls["year"] == year) & balls["state_ok"]]
        fold: dict[str, Any] = {name: fit_wp_model(name, train, config, drop, extra) for name in fitted}
        if with_dp:
            outcome = {inn: fit_outcome_model(train, inn, config, drop, extra) for inn in (1, 2)}
            for inn in (1, 2):
                rows = test[test["innings"] == inn]
                probs[rows.index] = predict_outcome_probs(outcome[inn], rows[outcome_features(inn, drop, extra)], n_classes)
            fold["dp"] = DPWinProbability(config, outcome, _contexts(test), drop)
            outcome_models[int(year)] = outcome
        models[int(year)] = fold
        logger.info("Cross-fit fold %s: trained on %d balls, held out %d", year, len(train), len(test))
    if not with_dp:
        return CrossFit(balls=balls, models=models, outcome_probs=probs, config=config, outcome_models={})

    # Platt recalibration of the DP, fitted on the other seasons' held-out DP predictions.
    clean = balls[balls["wp_train_row"]]
    raw = pd.Series(np.nan, index=clean.index)
    for year, fold in models.items():
        rows = clean[clean["year"] == year]
        raw[rows.index] = _raw_dp_wp(fold, rows)
    for year, fold in models.items():
        params = {}
        for inn in (1, 2):
            other = clean[(clean["year"] != year) & (clean["innings"] == inn)]
            params[inn] = fit_platt(raw[other.index].to_numpy(), other["batting_team_won"].to_numpy(), clip)
        fold["dp_platt"] = PlattCalibrated(fold["dp"], params, clip)
        logger.info("Platt (intercept, slope) for held-out %s by innings: %s", year, params)
    return CrossFit(balls=balls, models=models, outcome_probs=probs, config=config, outcome_models=outcome_models)


def leverage_at(cf: CrossFit, name: str, year: int, innings: int, states: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """WP and raw leverage of one candidate at arbitrary states, using the fold that held out ``year``.

    ``states`` needs the columns of ``leverage.STATE_COLUMNS``; outcome likelihoods are
    predicted for those states by the same fold's outcome model.
    """
    per_over = cf.config["ball_semantics"]["balls_per_over"]
    feats = states.copy()
    if innings == 2:
        feats["runs_required"] = feats["x"]
        feats["rrr_rel"] = feats["x"] * per_over / feats["balls_remaining"] / feats["env_rpo"]
    probs = predict_outcome_probs(cf.outcome_models[year][innings], feats, len(cf.config["outcome_classes"]))
    return leverage(cf.models[year][name], innings, states, probs, cf.config)


def score_candidate(cf: CrossFit, name: str, extra: tuple[str, ...] = ()) -> pd.DataFrame:
    """Cross-fitted WP and raw leverage of one candidate for every scored ball."""
    scored = cf.scored()
    out = pd.DataFrame(index=scored.index, columns=["wp", "leverage"], dtype=float)
    for year, fold in cf.models.items():
        for innings in (1, 2):
            rows = scored[(scored["year"] == year) & (scored["innings"] == innings)]
            if rows.empty:
                continue
            wp, lev = leverage(
                fold[name], innings, state_frame(rows, innings, extra), cf.outcome_probs[rows.index], cf.config
            )
            out.loc[rows.index, "wp"] = wp
            out.loc[rows.index, "leverage"] = lev
    return out


def predict_candidate(cf: CrossFit, name: str, rows: pd.DataFrame, extra: tuple[str, ...] = ()) -> np.ndarray:
    """Cross-fitted WP only (no leverage) for the given balls."""
    out = pd.Series(np.nan, index=rows.index)
    for year, fold in cf.models.items():
        for innings in (1, 2):
            part = rows[(rows["year"] == year) & (rows["innings"] == innings)]
            if not part.empty:
                out[part.index] = fold[name].predict(innings, state_frame(part, innings, extra))
    return out.to_numpy()


def roughness_candidate(cf: CrossFit, name: str) -> dict[str, float]:
    """Leverage roughness of one candidate on a fixed random sample of balls, by innings."""
    scored = cf.scored()
    sample = scored.sample(n=cf.config["models"]["smoothness_sample"], random_state=cf.config["seed"])
    result: dict[str, float] = {}
    for innings in (1, 2):
        gaps, levels, weights = [], [], []
        for year, fold in cf.models.items():
            rows = sample[(sample["year"] == year) & (sample["innings"] == innings)]
            rows = rows[rows["runs_required"] >= 2] if innings == 2 else rows[rows["score_before"] >= 1]
            if rows.empty:
                continue
            gap, level = leverage_roughness(
                fold[name], innings, state_frame(rows, innings), cf.outcome_probs[rows.index], cf.config
            )
            gaps.append(gap)
            levels.append(level)
            weights.append(len(rows))
        result[f"roughness_inn{innings}"] = float(
            np.average(gaps, weights=weights) / np.average(levels, weights=weights)
        )
    return result
