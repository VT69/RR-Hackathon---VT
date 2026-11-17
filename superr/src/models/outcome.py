"""League-average delivery-outcome model: P(outcome class | match state).

"League-average" is deliberate: these likelihoods describe the situation, not the
players in it. They are used (a) as the transition probabilities of the DP
win-probability model and (b) as the weights in the leverage calculation.
"""

from __future__ import annotations

from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

# The Impact Player flag was tested and dropped (DECISIONS.md D15); it survives only as an ablation extra.
CONTEXT_FEATURES: list[str] = ["env_rpo", "is_ipl", "involves_associate"]
# First innings: transitions depend on resources only (balls, wickets) plus context.
FIRST_FEATURES: list[str] = ["balls_remaining", "wickets_in_hand", *CONTEXT_FEATURES]
# Chase: batting sides adapt to the task, so the required rate enters as well.
CHASE_FEATURES: list[str] = [*FIRST_FEATURES, "runs_required", "rrr_rel"]


def outcome_features(innings: int, drop: tuple[str, ...] = (), extra: tuple[str, ...] = ()) -> list[str]:
    """Feature list for the given innings; ``drop`` removes and ``extra`` adds context features (ablations)."""
    base = FIRST_FEATURES if innings == 1 else CHASE_FEATURES
    return [f for f in base if f not in drop] + [f for f in extra if f not in base]


def fit_outcome_model(
    train: pd.DataFrame, innings: int, config: dict[str, Any], drop: tuple[str, ...] = (), extra: tuple[str, ...] = ()
) -> lgb.LGBMClassifier:
    """Fit the multiclass outcome model on usable-state balls of one innings."""
    params = config["models"]["outcome"]
    rows = train[(train["innings"] == innings) & train["state_ok"]]
    model = lgb.LGBMClassifier(
        objective="multiclass",
        random_state=config["seed"],
        n_jobs=config["models"]["n_jobs"],
        deterministic=True,
        force_row_wise=True,
        subsample_freq=1,
        verbose=-1,
        **params,
    )
    model.fit(rows[outcome_features(innings, drop, extra)], rows["outcome_class"])
    return model


def predict_outcome_probs(model: lgb.LGBMClassifier, features: pd.DataFrame, n_classes: int) -> np.ndarray:
    """Class probabilities as an (n, n_classes) array, zero-filled for classes unseen in training."""
    cols = list(model.feature_name_)
    raw = model.predict_proba(features[cols])
    out = np.zeros((len(features), n_classes), dtype=np.float64)
    out[:, model.classes_.astype(int)] = raw
    return out / out.sum(axis=1, keepdims=True)
